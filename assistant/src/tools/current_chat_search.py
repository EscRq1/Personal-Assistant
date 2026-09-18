from __future__ import annotations

import json
import re
import sqlite3
import threading
from typing import Iterable


_TOKEN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_+.#-]*")

MAX_QUERY_TERMS = 24
MAX_MATCHES = 12
NEIGHBOR_RADIUS = 2
MAX_CONTEXT_CHARS = 16000


def _extract_terms(query: str) -> list[str]:
    terms: list[str] = []

    for raw in _TOKEN_RE.findall(query or ""):
        term = raw.strip()

        if len(term) < 2:
            continue

        if term not in terms:
            terms.append(term)

        if len(terms) >= MAX_QUERY_TERMS:
            break

    return terms


def _fts_query(query: str) -> str:
    """
    Build a conservative SQLite FTS5 OR query.

    The tokenizer handles punctuation/model names better than a raw
    MATCH string, while quoting each term prevents query syntax from
    leaking into the FTS expression.
    """
    terms = _extract_terms(query)

    if not terms:
        return ""

    return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)


def _turn_text(turn: list[dict]) -> str:
    parts: list[str] = []

    for message in turn:
        if not isinstance(message, dict):
            continue

        role = str(message.get("role") or "").strip()
        content = message.get("content")

        if not isinstance(content, str) or not content.strip():
            continue

        if role:
            parts.append(f"{role.title()}: {content}")
        else:
            parts.append(content)

    return "\n".join(parts).strip()


def _normalize_turns(
    turns: Iterable[dict],
) -> list[dict]:
    """
    Validate and normalize the canonical indexed representation.

    The returned objects retain the original `index` and nested `turn`
    structure. This is important: the canonical representation is never
    replaced by a flat message list.
    """
    normalized: list[dict] = []

    for item in turns or ():
        if not isinstance(item, dict):
            continue

        turn_index = item.get("index")
        turn = item.get("turn")

        if not isinstance(turn_index, int):
            continue

        if not isinstance(turn, list):
            continue

        normalized.append(
            {
                "index": turn_index,
                "turn": list(turn),
            }
        )

    normalized.sort(key=lambda entry: entry["index"])
    return normalized


class CurrentChatIndex:
    """
    Ephemeral FTS5 index for one current conversation.

    A caller refreshes it with the authoritative canonical turn list.
    Search results never leave this object's process memory.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._turns: list[dict] = []
        self._db = sqlite3.connect(
            ":memory:",
            check_same_thread=False,
        )
        self._db.execute(
            """
            CREATE VIRTUAL TABLE current_chat_fts
            USING fts5(
                turn_pos UNINDEXED,
                turn_index UNINDEXED,
                text,
                tokenize='porter unicode61'
            )
            """
        )
        self._db.commit()

    def replace(self, turns: Iterable[dict]) -> None:
        normalized = _normalize_turns(turns)

        with self._lock:
            self._turns = normalized

            self._db.execute("DELETE FROM current_chat_fts")

            rows = []
            for position, entry in enumerate(normalized):
                text = _turn_text(entry["turn"])
                if not text:
                    continue

                rows.append(
                    (
                        position,
                        entry["index"],
                        text,
                    )
                )

            if rows:
                self._db.executemany(
                    """
                    INSERT INTO current_chat_fts(
                        turn_pos,
                        turn_index,
                        text
                    )
                    VALUES (?, ?, ?)
                    """,
                    rows,
                )

            self._db.commit()

    def clear(self) -> None:
        with self._lock:
            self._turns.clear()
            self._db.execute("DELETE FROM current_chat_fts")
            self._db.commit()

    def search(
        self,
        query: str,
        *,
        max_matches: int = MAX_MATCHES,
        neighbor_radius: int = NEIGHBOR_RADIUS,
        max_context_chars: int = MAX_CONTEXT_CHARS,
    ) -> dict:
        query = (query or "").strip()

        if not query:
            return {
                "query": "",
                "found": False,
                "message": "Empty current-chat search request.",
            }

        fts = _fts_query(query)

        if not fts:
            return {
                "query": query,
                "found": False,
                "message": "The search request contained no usable lexical terms.",
            }

        with self._lock:
            if not self._turns:
                return {
                    "query": query,
                    "found": False,
                    "message": "The current chat is empty.",
                }

            rows = self._db.execute(
                """
                SELECT turn_pos, turn_index, bm25(current_chat_fts) AS score
                FROM current_chat_fts
                WHERE current_chat_fts MATCH ?
                ORDER BY score ASC
                LIMIT ?
                """,
                (fts, max(1, min(max_matches, MAX_MATCHES))),
            ).fetchall()

            if not rows:
                return {
                    "query": query,
                    "found": False,
                    "message": "No matching turns were found in the current chat.",
                }

            # Search hits are expanded by neighboring positions. Using
            # positions rather than arithmetic on turn IDs keeps this correct
            # even if the canonical turn indices contain gaps.
            selected_positions: set[int] = set()
            matched_indices: list[int] = []

            turn_count = len(self._turns)

            for position, turn_index, _score in rows:
                matched_indices.append(turn_index)

                start = max(0, position - max(0, neighbor_radius))
                end = min(
                    turn_count,
                    position + max(0, neighbor_radius) + 1,
                )

                selected_positions.update(range(start, end))

            selected_positions = set(
                sorted(selected_positions)
            )

            selected_turns = [
                self._turns[position]
                for position in sorted(selected_positions)
            ]

            # The merge above naturally combines overlapping windows.
            # Keep the final representation canonical and chronological.
            selected_turns.sort(key=lambda entry: entry["index"])

            context: list[dict] = []
            total_chars = 0

            for entry in selected_turns:
                serialized = json.dumps(
                    entry,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                extra = len(serialized)

                if context and total_chars + extra > max_context_chars:
                    break

                if not context and extra > max_context_chars:
                    serialized_turn = {
                        "index": entry["index"],
                        "turn": [],
                    }

                    used = 0
                    for message in entry["turn"]:
                        if not isinstance(message, dict):
                            continue

                        message_copy = dict(message)
                        message_text = json.dumps(
                            message_copy,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )

                        if used + len(message_text) > max_context_chars:
                            break

                        serialized_turn["turn"].append(message_copy)
                        used += len(message_text)

                    entry = serialized_turn
                    extra = used

                context.append(entry)
                total_chars += extra

            return {
                "query": query,
                "found": bool(context),
                "matched_turns": matched_indices,
                "neighbor_radius": max(0, neighbor_radius),
                "turns": context,
            }


# One process-local index is enough for the active conversation.
current_chat_index = CurrentChatIndex()


def set_current_chat(turns: Iterable[dict]) -> None:
    """
    Replace the ephemeral index with the authoritative current-chat turns.

    This is the integration point used by the application layer. It does
    not persist the data and does not modify the canonical turn objects.
    """
    current_chat_index.replace(turns)


def clear_current_chat() -> None:
    current_chat_index.clear()


def search_current_chat(request: str) -> str:
    """
    Model-facing tool implementation.

    Search only the active chat. No Chroma, no Observer, no web access, and
    no long-term-memory writes are involved.
    """

    result = current_chat_index.search(request)

    return json.dumps(
        result,
        ensure_ascii=False,
    )