from __future__ import annotations

import json
import re
import sqlite3
import threading
from typing import Iterable
from .estimate_tokens import estimate_tokens


_TOKEN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_+.#-]*")

MAX_QUERY_TERMS = 24
MAX_MATCHES = 12
NEIGHBOR_RADIUS = 2
MAX_CONTEXT_TOKENS = 16000


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
        max_context_tokens: int,
        max_matches: int = MAX_MATCHES,
        neighbor_radius: int = NEIGHBOR_RADIUS,
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

        # Sanitize the caller-provided retrieval budget while keeping a
        # process-level safety ceiling.
        max_context_tokens = max(
            1,
            min(int(max_context_tokens), MAX_CONTEXT_TOKENS),
        )
        max_matches = max(1, min(int(max_matches), MAX_MATCHES))
        neighbor_radius = max(0, int(neighbor_radius))

        with self._lock:
            if not self._turns:
                return {
                    "query": query,
                    "found": False,
                    "message": "The current chat is empty.",
                }

            rows = self._db.execute(
                """
                SELECT
                    turn_pos,
                    turn_index,
                    bm25(current_chat_fts) AS score
                FROM current_chat_fts
                WHERE current_chat_fts MATCH ?
                ORDER BY score ASC
                LIMIT ?
                """,
                (fts, max_matches),
            ).fetchall()

            if not rows:
                return {
                    "query": query,
                    "found": False,
                    "message": "No matching turns were found in the current chat.",
                }

            turn_count = len(self._turns)

            # ------------------------------------------------------------
            # 1. Separate actual FTS hits from surrounding context.
            #
            # Matched turns always have priority. Neighbors are considered
            # only after the actual search hits have been considered.
            # ------------------------------------------------------------
            matched_positions: list[int] = []
            matched_indices: list[int] = []
            neighbor_positions: set[int] = set()

            for position, turn_index, _score in rows:
                matched_positions.append(position)
                matched_indices.append(turn_index)

                start = max(0, position - neighbor_radius)
                end = min(
                    turn_count,
                    position + neighbor_radius + 1,
                )

                neighbor_positions.update(range(start, end))

            # A matched turn should never also be classified as a neighbor.
            neighbor_positions.difference_update(matched_positions)

            # FTS rows are already ordered by relevance. Preserve that order
            # for matches so the most relevant results get budget priority.
            # Neighbors are chronological.
            priority_positions = (
                matched_positions
                + sorted(neighbor_positions)
            )

            # ------------------------------------------------------------
            # 2. Pack candidates into the requested token budget.
            #
            # A candidate that does not fit must NOT terminate the search.
            # It is skipped (or partially included when possible), and the
            # next candidate is still considered.
            # ------------------------------------------------------------
            selected_turns: list[dict] = []
            total_tokens = 0
            truncated = False
            skipped_indices: list[int] = []

            for position in priority_positions:
                entry = self._turns[position]

                serialized = json.dumps(
                    entry,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )

                entry_tokens = estimate_tokens(serialized)

                # Complete turn fits.
                if total_tokens + entry_tokens <= max_context_tokens:
                    selected_turns.append(entry)
                    total_tokens += entry_tokens
                    continue

                # We cannot fit the complete candidate. Mark the result as
                # truncated because some searchable context was omitted.
                truncated = True

                remaining_tokens = max_context_tokens - total_tokens

                if remaining_tokens <= 0:
                    skipped_indices.append(entry["index"])
                    continue

                # Try to preserve useful message-level context from this
                # oversized turn. We only append whole messages; the
                # canonical turn shape remains intact.
                partial_turn = {
                    "index": entry["index"],
                    "turn": [],
                }

                partial_tokens = 0

                for message in entry["turn"]:
                    if not isinstance(message, dict):
                        continue

                    message_copy = dict(message)

                    message_text = json.dumps(
                        message_copy,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )

                    message_tokens = estimate_tokens(message_text)

                    if partial_tokens + message_tokens > remaining_tokens:
                        break

                    partial_turn["turn"].append(message_copy)
                    partial_tokens += message_tokens

                if partial_turn["turn"]:
                    selected_turns.append(partial_turn)
                    total_tokens += partial_tokens

                    # If only part of the turn was returned, remember that
                    # the full turn was not available.
                    if partial_tokens < entry_tokens:
                        skipped_indices.append(entry["index"])
                else:
                    skipped_indices.append(entry["index"])

            # ------------------------------------------------------------
            # 3. Return selected turns chronologically.
            #
            # Selection was priority-based, but chronological ordering is
            # the most useful representation for the model to read.
            # ------------------------------------------------------------
            selected_turns.sort(
                key=lambda entry: entry["index"]
            )

            returned_indices = [
                entry["index"]
                for entry in selected_turns
            ]

            return {
                "query": query,
                "found": bool(selected_turns),
                "matched_turns": matched_indices,
                "returned_turns": returned_indices,
                "skipped_turns": skipped_indices,
                "neighbor_radius": neighbor_radius,
                "truncated": truncated,
                "requested_tokens": max_context_tokens,
                "estimated_tokens": total_tokens,
                "turns": selected_turns,
            }

    def get_turn(self, turn_index: int, max_tokens: int) -> dict:
        """
        Direct primary-key lookup of a single turn by its semantic index.

        This is the escape-hatch counterpart to `search`: when the FTS5
        rank-ordered budget allocation skips a relevant turn, the model can
        fetch it directly without competing for shared retrieval budget.

        The lookup scans `self._turns` linearly. The list is bounded to a
        single conversation and is small enough that an O(n) scan is cheaper
        than maintaining a secondary index.

        If the full serialized turn exceeds `max_tokens`, messages are
        included greedily until the budget is exhausted. The first valid
        message is always included (with content clipped at character level
        if necessary) to guarantee the model receives useful output.
        """
        max_tokens = max(1, min(int(max_tokens), MAX_CONTEXT_TOKENS))

        with self._lock:
            if not self._turns:
                return {
                    "turn_index": turn_index,
                    "found": False,
                    "message": "The current chat is empty.",
                }

            # Locate the entry by its semantic `index` field (not list position).
            entry: dict | None = None
            for candidate in self._turns:
                if candidate["index"] == turn_index:
                    entry = candidate
                    break

            if entry is None:
                return {
                    "turn_index": turn_index,
                    "found": False,
                    "message": (
                        f"Turn with index {turn_index} was not found "
                        f"in the current chat."
                    ),
                }

            # Serialize the full turn to measure its token cost.
            serialized = json.dumps(
                entry, ensure_ascii=False, separators=(",", ":"),
            )
            full_tokens = estimate_tokens(serialized)

            # ------------------------------------------------------------------
            # Case 1: The entire turn fits within the budget.
            # ------------------------------------------------------------------
            if full_tokens <= max_tokens:
                result: dict = {
                    "turn_index": turn_index,
                    "found": True,
                    "turn": entry["turn"],
                    "token_count": full_tokens,
                    "truncated": False,
                    "skipped_messages": 0,
                }

                messages = entry["turn"]
                if len(messages) == 1 and isinstance(messages[0], dict):
                    result["role"] = messages[0].get("role", "")
                    result["content"] = messages[0].get("content", "")

                return result

            # ------------------------------------------------------------------
            # Case 2: The turn exceeds the budget. Truncate at message level.
            # ------------------------------------------------------------------
            included_messages: list[dict] = []
            used_tokens = 0
            skipped_messages = 0
            note: str | None = None

            total_messages = len(entry["turn"])

            for i, message in enumerate(entry["turn"]):
                if not isinstance(message, dict):
                    skipped_messages += 1
                    continue

                message_copy = dict(message)
                message_text = json.dumps(
                    message_copy, ensure_ascii=False, separators=(",", ":"),
                )
                message_tokens = estimate_tokens(message_text)

                if used_tokens + message_tokens > max_tokens:
                    if not included_messages:
                        # --- Force-include the first valid message ---
                        # Clip content at character level so the model always
                        # receives at least partial information.
                        char_budget = max_tokens * 4  # inverse of len//4 heuristic
                        content = message_copy.get("content", "")
                        if isinstance(content, str) and len(content) > char_budget:
                            message_copy["content"] = content[:char_budget] + "\n…[truncated]"
                            note = (
                                f"First message content clipped to {char_budget} "
                                f"characters (original: {len(content)} chars). "
                                f"Increase max_tokens for full content."
                            )
                        included_messages.append(message_copy)
                        used_tokens = estimate_tokens(
                            json.dumps(message_copy, ensure_ascii=False, separators=(",", ":")),
                        )
                        # Remaining messages after index i are all skipped.
                        skipped_messages += total_messages - i - 1
                    else:
                        # Current message and all subsequent ones do not fit.
                        skipped_messages += total_messages - i
                    break

                included_messages.append(message_copy)
                used_tokens += message_tokens

            result = {
                "turn_index": turn_index,
                "found": True,
                "turn": included_messages,
                "token_count": used_tokens,
                "truncated": True,
                "skipped_messages": skipped_messages,
            }

            if note:
                result["note"] = note

            # Shortcut fields for single-message results.
            if len(included_messages) == 1 and isinstance(included_messages[0], dict):
                result["role"] = included_messages[0].get("role", "")
                result["content"] = included_messages[0].get("content", "")

            return result


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


def search_current_chat(request: str, tokens: int) -> str:
    """
    Model-facing tool implementation.

    Search only the active chat. No Chroma, no Observer, no web access, and
    no long-term-memory writes are involved.
    """

    result = current_chat_index.search(
        request,
        max_context_tokens=tokens,
    )

    return json.dumps(
        result,
        ensure_ascii=False,
    )


def get_chat_turn(turn_index: int, max_tokens: int = 8000) -> str:
    """
    Model-facing tool implementation.

    Retrieve a specific turn from the active chat by its semantic index.
    This bypasses FTS5 ranking and budget competition, serving as an
    escape hatch when `search_current_chat` skips a relevant turn due to
    rank-ordered token allocation exhausting the shared budget before
    reaching lower-ranked results.

    No Chroma, no Observer, no web access, and no long-term-memory writes
    are involved.
    """

    result = current_chat_index.get_turn(
        turn_index=turn_index,
        max_tokens=max_tokens,
    )

    return json.dumps(
        result,
        ensure_ascii=False,
    )