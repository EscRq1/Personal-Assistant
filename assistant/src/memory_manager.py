import os
import json
import uuid
import time
import sqlite3
import hashlib
import re


from resource_manager import resource_manager
from colorama import Fore, init
import chromadb

from rag_gates import (
    RAGGate,
    MEMORY_ANCHORS,
    CONVERSATIONAL_ANCHORS,
    CONTEXTUAL_FOLLOWUP_ANCHORS,
)

init(autoreset=True)

MAXIMUM_TOKENS = 2048
ALLOWED_PROFILE_FIELDS = {
    "Name",
    "Education",
    "Field_of_Study",
    "Career_Goal",
    "Current_Status",
    "Preferences",
    "Location",
}


FEEDBACK_ONLY = {
    "thanks",
    "thank you",
    "thx",
    "ty",
    "ok",
    "okay",
    "got it",
    "nice",
    "cool",
    "great",
    "yes",
    "yep",
    "no",
    "nope",
    "sure",
}


class MemoryManager:
    def __init__(self, db_path=None):
        if db_path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(base, "../memory_db")

        self.persist_directory = db_path

        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        self.history = self.client.get_or_create_collection(
            name="chat_history"
        )

        # Retrieval-optimized turn-level index. The legacy chat_history
        # collection remains the canonical archive/UI store.
        self.memory_turns = self.client.get_or_create_collection(
            name="memory_turns"
        )


        self.fts_path = os.path.join(
            self.persist_directory,
            "memory_turns_fts.sqlite3",
        )

        self._init_fts()

        self.profile_col = self.client.get_or_create_collection(
            name="user_profile"
        )

        self.intent_col = self.client.get_or_create_collection(
            name="rag_intent_anchors"
        )

        self.assistant_notebook = self.client.get_or_create_collection(
            name="assistant_notebook"
        )
        

        self.profile_cache = self._load_profile()
        self.last_profile_update_ts = 0

        self.resource_manager = resource_manager
        self.rag_gate = RAGGate()

        # Jina is lazy-loaded through ResourceManager.
        self.reranker = None

        # Ensure semantic intent anchors exist.
        self._ensure_intent_anchors()

        # One-time migration of the existing whole-chat archive into the
        # retrieval-optimized turn index.
        self._ensure_turn_index()

    def write_to_notebook(self, notes: str):
        token_count = len(notes) / 4

        if token_count > MAXIMUM_TOKENS:
            return {
                "error": (
                    f"Notebook exceeds the {MAXIMUM_TOKENS}-token limit."
                )
            }

        self.assistant_notebook.upsert(
            ids=["assistant_notebook"],
            documents=[notes],
        )

        return {
            "success": True,
            "tokens_estimate": int(token_count),
        }

    # ============================================================
    # Turn-level / hybrid retrieval index
    # ============================================================

    def get_notebook(self) -> str:
        try:
            data = self.assistant_notebook.get(
                ids=["assistant_notebook"],
                include=["documents"],
            )

            documents = data.get("documents") or []

            if not documents:
                return ""

            return documents[0] or ""

        except Exception as exc:
            print(
                f"{Fore.YELLOW}"
                f"[NOTEBOOK] retrieval failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return ""

    def _init_fts(self):
        with sqlite3.connect(self.fts_path) as conn:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_turns_fts
                USING fts5(
                    turn_id UNINDEXED,
                    chat_id UNINDEXED,
                    text,
                    tokenize='porter unicode61'
                )
                """
            )
            conn.commit()

    @staticmethod
    def _turn_id(chat_id: str, turn_index: int, text: str) -> str:
        payload = f"{chat_id}:{turn_index}:{text}".encode("utf-8")
        return hashlib.sha1(payload).hexdigest()

    @staticmethod
    def _split_turns(document: str) -> list[str]:
        """Convert a canonical JSON message list into searchable turn text."""
        if not document or not isinstance(document, str):
            return []

        try:
            messages = json.loads(document)
        except json.JSONDecodeError:
            return []

        if not isinstance(messages, list):
            return []

        turns = []
        pending_user = None

        for message in messages:
            if not isinstance(message, dict):
                continue

            role = message.get("role")
            content = message.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                continue

            if role == "user":
                if pending_user is not None:
                    turns.append(f"User: {pending_user}")
                pending_user = content
                continue

            if pending_user is None:
                turns.append(f"Assistant: {content}")
            else:
                turns.append(f"User: {pending_user}\nAssistant: {content}")
                pending_user = None

        if pending_user is not None:
            turns.append(f"User: {pending_user}")

        return [turn.strip() for turn in turns if turn.strip()]

    def _delete_turn_index_for_chat(self, chat_id: str):
        if not chat_id:
            return

        try:
            existing = self.memory_turns.get(
                where={"chat_id": chat_id},
                include=[],
            )
            ids = existing.get("ids") or []
            if ids:
                self.memory_turns.delete(ids=ids)
        except Exception as exc:
            print(
                f"{Fore.YELLOW}"
                f"[MEMORY INDEX] Chroma cleanup failed: {exc}"
            )

        with sqlite3.connect(self.fts_path) as conn:
            conn.execute(
                "DELETE FROM memory_turns_fts WHERE chat_id = ?",
                (chat_id,),
            )
            conn.commit()

    def sync_turn_index(
        self,
        chat_id: str,
        document: str,
        metadata: dict | None = None,
    ) -> int:
        """Rebuild the search index for one archived conversation."""
        chat_id = str(chat_id or "").strip()
        if not chat_id:
            return 0

        turns = self._split_turns(document)
        self._delete_turn_index_for_chat(chat_id)

        if not turns:
            return 0

        metadata = dict(metadata or {})
        title = str(metadata.get("title") or "")
        timestamp = str(metadata.get("timestamp") or "")
        pinned = int(bool(metadata.get("pinned", 0)))

        ids = []
        documents = []
        metadatas = []

        for turn_index, turn in enumerate(turns):
            turn_id = self._turn_id(
                chat_id,
                turn_index,
                turn,
            )
            ids.append(turn_id)
            documents.append(turn)
            metadatas.append({
                "chat_id": chat_id,
                "turn_index": turn_index,
                "title": title,
                "timestamp": timestamp,
                "pinned": pinned,
            })

        self.memory_turns.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        with sqlite3.connect(self.fts_path) as conn:
            conn.executemany(
                "INSERT INTO memory_turns_fts(turn_id, chat_id, text) VALUES (?, ?, ?)",
                [
                    (turn_id, chat_id, turn)
                    for turn_id, turn in zip(ids, turns)
                ],
            )
            conn.commit()

        print(
            f"{Fore.CYAN}"
            f"[MEMORY INDEX] indexed {len(turns)} turns for chat {chat_id}"
        )
        return len(turns)

    def _ensure_turn_index(self):
        """Migrate legacy chat-level records the first time turn indexing is enabled."""
        try:
            if self.memory_turns.count() > 0:
                return

            existing = self.history.get(
                include=["documents", "metadatas"],
            )
            ids = existing.get("ids") or []
            documents = existing.get("documents") or []
            metadatas = existing.get("metadatas") or []

            if not ids:
                return

            print(
                f"{Fore.CYAN}"
                f"[MEMORY INDEX] migrating {len(ids)} legacy chats..."
            )

            for chat_id, document, metadata in zip(
                ids,
                documents,
                metadatas,
            ):
                self.sync_turn_index(
                    chat_id,
                    document,
                    metadata or {},
                )

            print(
                f"{Fore.GREEN}"
                "[MEMORY INDEX] migration complete"
            )

        except Exception as exc:
            print(
                f"{Fore.YELLOW}"
                f"[MEMORY INDEX] migration failed: {exc}"
            )

    @staticmethod
    def _lexical_query(query: str) -> str:
        terms = re.findall(
            r"[A-Za-z0-9_][A-Za-z0-9_+.#-]*",
            query or "",
        )
        terms = [term for term in terms if len(term) >= 2]
        if not terms:
            return ""

        safe_terms = [
            term.replace('"', '')
            for term in terms[:24]
        ]
        return " OR ".join(
            f'"{term}"'
            for term in safe_terms
        )

    def _lexical_candidates(
        self,
        query: str,
        limit: int = 30,
    ) -> list[dict]:
        match_query = self._lexical_query(query)
        if not match_query:
            return []

        try:
            with sqlite3.connect(self.fts_path) as conn:
                rows = conn.execute(
                    """
                    SELECT turn_id, chat_id, text,
                           bm25(memory_turns_fts) AS score
                    FROM memory_turns_fts
                    WHERE memory_turns_fts MATCH ?
                    ORDER BY score ASC
                    LIMIT ?
                    """,
                    (match_query, limit),
                ).fetchall()

            return [
                {
                    "turn_id": row[0],
                    "chat_id": row[1],
                    "text": row[2],
                    "score": float(row[3]),
                }
                for row in rows
            ]
        except Exception as exc:
            print(
                f"{Fore.YELLOW}"
                f"[MEMORY INDEX] lexical search failed: {exc}"
            )
            return []

    def _hybrid_candidates(
        self,
        query: str,
        vector_limit: int = 30,
        lexical_limit: int = 30,
        fusion_limit: int = 40,
        exclude_chat_id: str | None = None,
    ) -> list[dict]:
        """Hybrid vector + lexical retrieval with Reciprocal Rank Fusion."""
        fused = {}

        try:
            count = self.memory_turns.count()
            if count > 0:
                results = self.memory_turns.query(
                    query_texts=[query],
                    n_results=min(vector_limit, count),
                    include=["documents", "metadatas", "distances"],
                )

                ids = (results.get("ids") or [[]])[0] or []
                documents = (results.get("documents") or [[]])[0] or []
                metadatas = (results.get("metadatas") or [[]])[0] or []

                for rank, (turn_id, document, metadata) in enumerate(
                    zip(ids, documents, metadatas),
                    start=1,
                ):
                    metadata = metadata or {}
                    chat_id = str(metadata.get("chat_id") or "")
                    if exclude_chat_id and chat_id == exclude_chat_id:
                        continue

                    fused.setdefault(
                        turn_id,
                        {
                            "turn_id": turn_id,
                            "chat_id": chat_id,
                            "text": document,
                            "metadata": metadata,
                            "rrf": 0.0,
                        },
                    )
                    fused[turn_id]["rrf"] += 1.0 / (60.0 + rank)

        except Exception as exc:
            print(
                f"{Fore.YELLOW}"
                f"[MEMORY INDEX] vector retrieval failed: {exc}"
            )

        # Lexical retrieval is especially useful for rare identifiers,
        # versions, filenames, model names, and technical terms.
        for rank, item in enumerate(
            self._lexical_candidates(query, lexical_limit),
            start=1,
        ):
            chat_id = item["chat_id"]
            if exclude_chat_id and chat_id == exclude_chat_id:
                continue

            turn_id = item["turn_id"]
            fused.setdefault(
                turn_id,
                {
                    "turn_id": turn_id,
                    "chat_id": chat_id,
                    "text": item["text"],
                    "metadata": {"chat_id": chat_id},
                    "rrf": 0.0,
                },
            )
            # Slightly favor lexical matches for exact technical terms.
            fused[turn_id]["rrf"] += 1.15 / (60.0 + rank)

        return sorted(
            fused.values(),
            key=lambda item: item["rrf"],
            reverse=True,
        )[:fusion_limit]

    # ============================================================
    # Jina
    # ============================================================

    def _get_reranker(self):
        if self.resource_manager is None:
            raise RuntimeError(
                "MemoryManager requires a ResourceManager"
            )

        self.resource_manager.acquire("jina")
        return self.resource_manager.get_jina()

    # ============================================================
    # RAG semantic intent
    # ============================================================

    def _ensure_intent_anchors(self):
        """
        Ensure all current semantic intent anchors exist.

        This is incremental so adding new anchors later does not
        require deleting the existing Chroma collection.
        """
        existing = self.intent_col.get(
            include=["documents", "metadatas"],
        )

        existing_ids = set(existing.get("ids") or [])

        ids = []
        documents = []
        metadatas = []

        for i, text in enumerate(MEMORY_ANCHORS):
            anchor_id = f"memory_{i}"
            if anchor_id in existing_ids:
                continue

            ids.append(anchor_id)
            documents.append(text)
            metadatas.append({"intent": "memory"})

        for i, text in enumerate(CONVERSATIONAL_ANCHORS):
            anchor_id = f"social_{i}"
            if anchor_id in existing_ids:
                continue

            ids.append(anchor_id)
            documents.append(text)
            metadatas.append({"intent": "social"})

        for i, text in enumerate(CONTEXTUAL_FOLLOWUP_ANCHORS):
            anchor_id = f"followup_{i}"
            if anchor_id in existing_ids:
                continue

            ids.append(anchor_id)
            documents.append(text)
            metadatas.append({"intent": "followup"})

        if not ids:
            print(
                f"{Fore.CYAN}"
                "[RAG GATE] semantic anchors already up to date"
            )
            return

        self.intent_col.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        print(
            f"{Fore.CYAN}"
            f"[RAG GATE] added {len(ids)} semantic anchors"
        )

    def _semantic_intent_scores(self, query: str):
        try:
            count = self.intent_col.count()

            if count == 0:
                print(
                    f"{Fore.YELLOW}"
                    "[RAG GATE] semantic anchor "
                    "collection is empty"
                )
                return 0.0, 0.0, 0.0

            results = self.intent_col.query(
                query_texts=[query],
                n_results=min(12, count),
                include=["metadatas", "distances"],
            )

            metadatas = (
                results.get("metadatas") or [[]]
            )[0] or []

            distances = (
                results.get("distances") or [[]]
            )[0] or []

            best_memory = 0.0
            best_social = 0.0
            best_followup = 0.0

            for metadata, distance in zip(metadatas, distances):
                if distance is None:
                    continue

                similarity = 1.0 / (1.0 + float(distance))
                intent = metadata.get("intent")

                if intent == "memory":
                    best_memory = max(best_memory, similarity)
                elif intent == "social":
                    best_social = max(best_social, similarity)
                elif intent == "followup":
                    best_followup = max(best_followup, similarity)

            return best_memory, best_social, best_followup

        except Exception as e:
            print(
                f"{Fore.YELLOW}"
                f"[RAG GATE] semantic anchor lookup failed: {e}"
            )
            return 0.0, 0.0, 0.0

    def _history_relevance_gate(self, distances):
        """
        Classify Chroma's best match into quality zones.

        Chroma always returns nearest neighbours. A neighbour is not
        automatically a relevant memory, so this stage is deliberately
        conservative before we spend VRAM on Jina.
        """
        if not distances:
            return False, 0.0, "weak"

        values = sorted(
            float(d)
            for d in distances
            if d is not None
        )

        if not values:
            return False, 0.0, "weak"

        best = values[0]

        STRONG_DISTANCE = 1.20
        AMBIGUOUS_DISTANCE = 1.50

        if best <= STRONG_DISTANCE:
            relevance = max(
                0.0,
                min(
                    1.0,
                    1.0 - best / STRONG_DISTANCE,
                ),
            )

            print(
                f"{Fore.GREEN}"
                f"[RAG CHROMA] strong: "
                f"best_distance={best:.4f} "
                f"relevance={relevance:.3f}"
            )

            return True, relevance, "strong"

        if best <= AMBIGUOUS_DISTANCE:
            relevance = max(
                0.0,
                min(
                    1.0,
                    1.0 - best / AMBIGUOUS_DISTANCE,
                ),
            )

            print(
                f"{Fore.YELLOW}"
                f"[RAG CHROMA] ambiguous: "
                f"best_distance={best:.4f} "
                f"relevance={relevance:.3f}"
            )

            return True, relevance, "ambiguous"

        print(
            f"{Fore.YELLOW}"
            f"[RAG CHROMA] weak: "
            f"best_distance={best:.4f} "
            f"> ambiguous threshold={AMBIGUOUS_DISTANCE:.4f}"
        )

        return False, 0.0, "weak"

    # ============================================================
    # Profile helpers
    # ============================================================

    @staticmethod
    def normalize_pin_value(value) -> int:
        """Normalize legacy bool/string/number values to exactly 0 or 1."""
        if isinstance(value, bool):
            return int(value)

        if isinstance(value, (int, float)):
            return 1 if int(value) == 1 else 0

        return (
            1
            if str(value).strip().lower()
            in {"1", "true", "yes", "on"}
            else 0
        )

    def _load_profile(self) -> dict:
        """Load JSON profile from Chroma (single doc)."""
        try:
            data = self.profile_col.get(
                ids=["user_core_profile"]
            )

            if data and data.get("documents"):
                raw = data["documents"][0]
                return json.loads(raw) if raw else {}

        except Exception:
            pass

        return {}

    def _save_profile(self):
        """Save JSON profile to Chroma (single doc)."""
        try:
            payload = json.dumps(
                self.profile_cache,
                ensure_ascii=False,
            )

            self.profile_col.upsert(
                documents=[payload],
                ids=["user_core_profile"],
            )

        except Exception as e:
            print(
                f"{Fore.RED}"
                f"[Profile Save Error]: {e}"
            )

    # ============================================================
    # History
    # ============================================================

    def add_history(self, text: str, summary: dict, compacted_memory: str = None):
        doc_id = str(uuid.uuid4())

        metadata = dict(summary or {})
        metadata["pinned"] = self.normalize_pin_value(
            metadata.get("pinned", 0)
        )
        if compacted_memory is not None:
            metadata["compacted_data"] = compacted_memory

        self.history.add(
            documents=[text],
            ids=[doc_id],
            metadatas=[metadata],
        )

        self.sync_turn_index(
            chat_id=doc_id,
            document=text,
            metadata=metadata,
        )

        return doc_id

    def update_history(
        self,
        chat_id: str,
        text: str,
        metadata: dict | None = None,
    ) -> bool:
        """Update one canonical chat and rebuild its retrieval index."""
        chat_id = str(chat_id or "").strip()
        if not chat_id or not isinstance(text, str):
            return False

        try:
            existing = self.history.get(
                ids=[chat_id],
                include=["metadatas"],
            )
            if not existing.get("ids"):
                return False

            metadata = dict(metadata or {})
            metadata["pinned"] = self.normalize_pin_value(
                metadata.get("pinned", 0)
            )

            self.history.update(
                ids=[chat_id],
                documents=[text],
                metadatas=[metadata],
            )

            self.sync_turn_index(
                chat_id=chat_id,
                document=text,
                metadata=metadata,
            )
            return True

        except Exception as exc:
            print(
                f"{Fore.RED}"
                f"[Update History Error]: {exc}"
            )
            return False

    def get_history_metadata(
        self,
        chat_id: str,
    ) -> dict | None:
        """Return a copy of one chat's metadata."""
        chat_id = (chat_id or "").strip()

        if not chat_id:
            return None

        data = self.history.get(
            ids=[chat_id],
            include=["metadatas"],
        )

        if not data.get("ids"):
            return None

        metadatas = data.get("metadatas") or []

        if not metadatas:
            return {}

        return dict(metadatas[0] or {})

    def delete_history(self, chat_id: str) -> bool:
        chat_id = (chat_id or "").strip()

        if not chat_id:
            return False

        try:
            data = self.history.get(
                ids=[chat_id],
                include=[],
            )

            if not data.get("ids"):
                return False

            self.history.delete(
                ids=[chat_id]
            )
            self._delete_turn_index_for_chat(chat_id)

            return True

        except Exception as e:
            print(
                f"{Fore.RED}"
                f"[Delete History Error]: {e}"
            )
            return False

    def rename_history(
        self,
        chat_id: str,
        new_title: str,
    ) -> bool:
        chat_id = (chat_id or "").strip()
        new_title = (new_title or "").strip()

        if not chat_id or not new_title:
            return False

        try:
            meta = self.get_history_metadata(
                chat_id
            )

            if meta is None:
                return False

            meta["title"] = new_title

            meta["pinned"] = self.normalize_pin_value(
                meta.get("pinned", 0)
            )

            self.history.update(
                ids=[chat_id],
                metadatas=[meta],
            )

            return True

        except Exception as e:
            print(
                f"{Fore.RED}"
                f"[Rename History Error]: {e}"
            )
            return False

    def toggle_pin_history(
        self,
        chat_id: str,
    ) -> int | None:
        """
        Toggle one chat's pin state.

        Returns:
            1: now pinned
            0: now unpinned
            None: invalid/missing chat
        """
        chat_id = (chat_id or "").strip()

        if not chat_id:
            return None

        meta = self.get_history_metadata(
            chat_id
        )

        if meta is None:
            return None

        current = self.normalize_pin_value(
            meta.get("pinned", 0)
        )

        new_value = (
            0 if current == 1 else 1
        )

        meta["pinned"] = new_value

        self.history.update(
            ids=[chat_id],
            metadatas=[meta],
        )

        return new_value

    def prune_memories(self):
        """Delete very short junk history items and their search indexes."""
        existing = self.history.get(
            include=["documents"]
        )

        if not existing.get("ids"):
            return

        documents = existing.get(
            "documents"
        ) or []

        ids_to_delete = []

        for index, chat_id in enumerate(
            existing["ids"]
        ):
            document = (
                documents[index]
                if index < len(documents)
                else ""
            ) or ""

            if len(document) < 15:
                ids_to_delete.append(
                    chat_id
                )

        if not ids_to_delete:
            return

        self.history.delete(
            ids=ids_to_delete
        )

        for chat_id in ids_to_delete:
            self._delete_turn_index_for_chat(chat_id)

        print(
            f"{Fore.RED}"
            f"Cleaned "
            f"{len(ids_to_delete)} "
            f"junk history items."
        )

    # ============================================================
    # Profile merge
    # ============================================================

    def merge_profile_fact(
        self,
        field: str,
        value: str,
        confidence: float = 0.7,
    ) -> bool:
        field = (field or "").strip()
        value = (value or "").strip()

        if field not in ALLOWED_PROFILE_FIELDS:
            return False

        if len(value) < 2 or value.endswith("?"):
            return False

        if confidence < 0.6:
            return False

        if value.lower().strip() in FEEDBACK_ONLY:
            return False

        now = time.time()
        existing = self.profile_cache.get(field)

        last_field_update = (
            float(existing.get("updated_at", 0.0))
            if existing
            else 0.0
        )

        if (
            last_field_update
            and now - last_field_update < 180
            and existing
            and float(confidence)
                < float(
                    existing.get("confidence", 0.0)
                ) + 0.05
        ):
            return False

        if (
            existing
            and float(confidence)
                < float(
                    existing.get("confidence", 0.0)
                ) - 0.10
        ):
            return False

        if not existing:
            self.profile_cache[field] = {
                "value": value,
                "confidence": float(confidence),
                "alternatives": [],
                "updated_at": now,
            }

            self._save_profile()
            self.last_profile_update_ts = now

            return True

        if existing.get("value") == value:
            old_conf = float(
                existing.get(
                    "confidence",
                    0.0,
                )
            )

            new_conf = max(
                old_conf,
                float(confidence),
            )

            if new_conf == old_conf:
                return False

            existing["confidence"] = new_conf
            existing["updated_at"] = now

            self.profile_cache[field] = existing
            self._save_profile()
            self.last_profile_update_ts = now

            return True

        old_conf = float(
            existing.get(
                "confidence",
                0.0,
            )
        )

        if float(confidence) >= old_conf + 0.05:
            alternatives = [
                item
                for item in existing.get(
                    "alternatives",
                    [],
                )
                if item.get("value") != value
            ]

            if (
                existing.get("value")
                and existing.get("value") != value
            ):
                alternatives.append({
                    "value": existing["value"],
                    "confidence": old_conf,
                })

            existing = {
                "value": value,
                "confidence": float(confidence),
                "alternatives": alternatives[-5:],
                "updated_at": now,
            }

            self.profile_cache[field] = existing
            self._save_profile()
            self.last_profile_update_ts = now

            return True

        alternatives = existing.get(
            "alternatives",
            [],
        )

        for alternative in alternatives:
            if alternative.get("value") != value:
                continue

            old_alt_conf = float(
                alternative.get(
                    "confidence",
                    0.0,
                )
            )

            new_alt_conf = max(
                old_alt_conf,
                float(confidence),
            )

            if new_alt_conf == old_alt_conf:
                return False

            alternative["confidence"] = new_alt_conf
            existing["alternatives"] = alternatives
            existing["updated_at"] = now

            self.profile_cache[field] = existing
            self._save_profile()
            self.last_profile_update_ts = now

            return True

        alternatives.append({
            "value": value,
            "confidence": float(confidence),
        })

        existing["alternatives"] = alternatives[-5:]
        existing["updated_at"] = now

        self.profile_cache[field] = existing
        self._save_profile()
        self.last_profile_update_ts = now

        return True

    def get_profile_summary(
        self,
        include_alternatives: bool = False,
    ) -> str:
        """Return a clean profile summary for the system prompt."""
        if not self.profile_cache:
            return ""

        lines = []

        order = [
            "Name",
            "Education",
            "Field_of_Study",
            "Career_Goal",
            "Current_Status",
            "Preferences",
            "Location",
        ]

        for key in order:
            if key not in self.profile_cache:
                continue

            entry = self.profile_cache[key]
            main_value = entry.get("value", "")

            if not main_value:
                continue

            lines.append(
                f"{key}: {main_value}"
            )

            if include_alternatives:
                alternatives = entry.get(
                    "alternatives",
                    [],
                )

                top_alternatives = sorted(
                    alternatives,
                    key=lambda item: item.get(
                        "confidence",
                        0,
                    ),
                    reverse=True,
                )[:2]

                for alternative in top_alternatives:
                    value = alternative.get(
                        "value"
                    )

                    if value:
                        lines.append(
                            f"  - alt: {value}"
                        )

        return "\n".join(lines)

    # ============================================================
    # Context retrieval
    # ============================================================

    def get_raw_context(
        self,
        query: str,
        k: int = 20,
        top_n: int = 8,
        max_chars: int = 12000,
        exclude_chat_id: str | None = None,
    ):
        """
        Automatic long-term retrieval.

        Routing is separated from retrieval quality:
          1. RAGGate decides whether long-term memory is appropriate.
          2. Hybrid turn-level retrieval maximizes recall.
          3. Jina performs final precision reranking.
          4. Only a small bounded context is returned to the main model.
        """
        t0 = time.perf_counter()

        basic = self.rag_gate.evaluate_basic(query)

        print(
            f"[RAG GATE] route={basic.get('route', 'none')} "
            f"reason={basic.get('reason', '')} "
            f"memory={basic.get('memory', 0.0):.3f} "
            f"social={basic.get('social', 0.0):.3f} "
            f"followup={basic.get('followup', 0.0):.3f}"
        )

        empty = {
            "history": [],
            "profile_summary": self.get_profile_summary(
                include_alternatives=False
            ),
        }

        if not basic.get("use_rag", False):
            return empty

        # --------------------------------------------------------
        # Stage 1: broad hybrid turn retrieval
        # --------------------------------------------------------
        candidates = self._hybrid_candidates(
            query=query,
            vector_limit=max(30, k),
            lexical_limit=30,
            fusion_limit=50,
            exclude_chat_id=exclude_chat_id,
        )

        t1 = time.perf_counter()
        print(
            f"[TIMING] hybrid retrieval {t1 - t0:.3f}s "
            f"({len(candidates)} candidates)"
        )

        if not candidates:
            return empty

        # --------------------------------------------------------
        # Stage 2: Jina precision reranking
        # --------------------------------------------------------
        relevant_turns = []

        try:
            MAX_JINA_CANDIDATES = 50
            MIN_RELEVANCE = 0.20
            RELATIVE_THRESHOLD = 0.75

            candidates = candidates[:MAX_JINA_CANDIDATES]

            rerank_documents = []
            for index, candidate in enumerate(candidates):
                text = candidate["text"]
                if len(text) <= 7000:
                    excerpt = text
                else:
                    excerpt = (
                        text[:3500]
                        + "\n...[middle omitted for reranking]...\n"
                        + text[-3500:]
                    )

                rerank_documents.append(
                    f"[CANDIDATE {index}]\n{excerpt}"
                )

            reranker = self._get_reranker()
            results = reranker.rerank(
                query,
                rerank_documents,
                top_n=min(top_n, len(rerank_documents)),
            )

            if results:
                top_score = float(
                    results[0].get("relevance_score", 0.0)
                )

                for result in results:
                    score = float(
                        result.get("relevance_score", 0.0)
                    )

                    if score < MIN_RELEVANCE:
                        continue

                    if (
                        top_score > 0.0
                        and score < top_score * RELATIVE_THRESHOLD
                    ):
                        break

                    text = result.get("document", "")
                    try:
                        index = int(
                            text.split("\n", 1)[0]
                            .removeprefix("[CANDIDATE ")
                            .removesuffix("]")
                        )
                        relevant_turns.append(
                            candidates[index]["text"]
                        )
                    except (ValueError, IndexError):
                        continue

                    if len(relevant_turns) >= top_n:
                        break

        except Exception as exc:
            print(
                f"{Fore.RED}"
                f"Jina reranking failed: {type(exc).__name__}: {exc}"
            )

        t2 = time.perf_counter()
        print(
            f"[TIMING] reranking {t2 - t1:.3f}s "
            f"selected={len(relevant_turns)}"
        )

        if not relevant_turns:
            return empty

        # --------------------------------------------------------
        # Stage 3: final context budget
        # --------------------------------------------------------
        selected = []
        current_chars = 0

        for turn in relevant_turns:
            remaining = max_chars - current_chars
            if remaining <= 0:
                break

            clipped = turn[:remaining]
            selected.append(clipped)
            current_chars += len(clipped)

        return {
            "history": selected,
            "profile_summary": self.get_profile_summary(
                include_alternatives=False
            ),
        }

    def get_memory_candidates(
        self,
        query: str,
        conversation_limit: int = 5,
        turns_per_conversation: int = 6,
        exclude_chat_id: str | None = None,
    ) -> list[str]:
        """
        Candidate generation for explicit memory_call.

        Hybrid retrieval provides broad recall. Observer does final
        semantic selection, so Jina is intentionally not used here.
        """
        query = (query or "").strip()
        if not query:
            return []

        try:
            candidates = self._hybrid_candidates(
                query=query,
                vector_limit=max(20, conversation_limit * turns_per_conversation),
                lexical_limit=max(20, conversation_limit * turns_per_conversation),
                fusion_limit=30,
                exclude_chat_id=exclude_chat_id,
            )

            texts = [
                item["text"]
                for item in candidates[:30]
            ]

            print(
                f"[MEMORY TOOL] prepared {len(texts)} hybrid candidates"
            )
            return texts

        except Exception as exc:
            print(
                f"{Fore.RED}"
                f"[MEMORY TOOL] candidate retrieval failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return []

    


memory_manager = MemoryManager()