import os
import json
import uuid
import time

# Environment setup must happen before importing Hugging Face / Chroma.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_OFF"] = "True"
os.environ["POSTHOG_DISABLED"] = "1"

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
            db_path = os.path.join(base, "memory_db")

        self.persist_directory = db_path

        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        self.history = self.client.get_or_create_collection(
            name="chat_history"
        )

        self.profile_col = self.client.get_or_create_collection(
            name="user_profile"
        )

        self.intent_col = self.client.get_or_create_collection(
            name="rag_intent_anchors"
        )

        self.profile_cache = self._load_profile()
        self.last_profile_update_ts = 0

        self.resource_manager = resource_manager
        self.rag_gate = RAGGate()

        # Jina is lazy-loaded through ResourceManager.
        self.reranker = None

        # Ensure semantic intent anchors exist.
        self._ensure_intent_anchors()

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

    def add_history(self, text: str, summary: dict):
        doc_id = str(uuid.uuid4())

        metadata = dict(summary or {})

        metadata["pinned"] = self.normalize_pin_value(
            metadata.get("pinned", 0)
        )

        self.history.add(
            documents=[text],
            ids=[doc_id],
            metadatas=[metadata],
        )

        return doc_id

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
        """Delete very short junk history items."""
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

        if ids_to_delete:
            self.history.delete(
                ids=ids_to_delete
            )

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
        k: int = 5,
        top_n: int = 8,
        max_chars: int = 12000,
        exclude_chat_id: str | None = None,
    ):
        t0 = time.perf_counter()

        # ========================================================
        # Stage 0: cheap lexical / fuzzy gate
        # ========================================================

        basic = self.rag_gate.evaluate_basic(query)

        print(
            f"[RAG GATE] basic: "
            f"use_rag={basic['use_rag']} "
            f"confidence={basic['confidence']:.3f} "
            f"reason={basic['reason']} "
            f"followup={basic.get('followup', 0.0):.3f}"
        )

        if basic["reason"] in {
            "exact_conversational_anchor",
            "fuzzy_conversational_anchor",
        }:
            return {
                "history": [],
                "profile_summary": self.get_profile_summary(
                    include_alternatives=False
                ),
            }

        # ========================================================
        # Stage 1: semantic intent
        # ========================================================

        (
            semantic_memory,
            semantic_social,
            semantic_followup,
        ) = self._semantic_intent_scores(query)

        print(
            f"[RAG GATE] semantic: "
            f"memory={semantic_memory:.3f} "
            f"social={semantic_social:.3f} "
            f"followup={semantic_followup:.3f}"
        )

        memory_intent = max(
            basic.get("memory", 0.0),
            semantic_memory,
        )

        social_intent = max(
            basic.get("social", 0.0),
            semantic_social,
        )

        followup_intent = max(
            basic.get("followup", 0.0),
            semantic_followup,
        )

        # Avoid even touching Chroma for clear social queries.
        if (
            social_intent >= 0.60
            and social_intent > memory_intent + 0.10
            and followup_intent < 0.55
        ):
            print(
                "[RAG GATE] rejected: "
                "semantic conversational intent"
            )

            return {
                "history": [],
                "profile_summary": self.get_profile_summary(
                    include_alternatives=False
                ),
            }

        # ========================================================
        # Stage 2: cheap Chroma history retrieval
        # ========================================================

        history_docs = []
        history_found = False
        history_relevance = 0.0
        history_zone = "weak"

        try:
            history_count = self.history.count()

            if history_count > 0:
                fetch_n = min(
                    k + (3 if exclude_chat_id else 0),
                    history_count,
                )

                results = self.history.query(
                    query_texts=[query],
                    n_results=fetch_n,
                    include=[
                        "documents",
                        "metadatas",
                        "distances",
                    ],
                )

                docs = (
                    results.get("documents") or [[]]
                )[0] or []

                ids = (
                    results.get("ids") or [[]]
                )[0] or []

                distances = (
                    results.get("distances") or [[]]
                )[0] or []

                print(
                    "[RAG CHROMA]",
                    list(zip(distances, ids)),
                )

                (
                    history_found,
                    history_relevance,
                    history_zone,
                ) = self._history_relevance_gate(distances)

                for doc_id, document in zip(ids, docs):
                    if (
                        exclude_chat_id
                        and doc_id == exclude_chat_id
                    ):
                        continue

                    if document:
                        history_docs.append(document)

                    if len(history_docs) >= k:
                        break

        except Exception as e:
            print(
                f"{Fore.RED}"
                f"Error fetching history: {e}"
            )

            history_found = False
            history_relevance = 0.0
            history_zone = "weak"

        t1 = time.perf_counter()

        print(
            f"[TIMING] history collections "
            f"fetching {t1 - t0:.3f}s"
        )

        # ========================================================
        # Stage 3: hard Chroma gate
        # ========================================================

        if not history_found:
            print(
                f"{Fore.YELLOW}"
                "[RAG GATE] rejected: "
                "no sufficiently relevant history"
            )

            return {
                "history": [],
                "profile_summary": self.get_profile_summary(
                    include_alternatives=False
                ),
            }

        # ========================================================
        # Stage 4: explicit-memory vs contextual-follow-up route
        # ========================================================

        margin = memory_intent - social_intent

        memory_route = (
            memory_intent >= 0.55
            and margin >= 0.08
            and history_zone in {"strong", "ambiguous"}
        )

        followup_route = (
            followup_intent >= 0.55
            and history_zone == "strong"
            and social_intent < 0.75
        )

        ambiguous_followup_route = (
            followup_intent >= 0.65
            and memory_intent >= 0.40
            and history_zone == "ambiguous"
            and social_intent < 0.75
        )

        use_rag = (
            memory_route
            or followup_route
            or ambiguous_followup_route
        )

        print(
            f"[RAG GATE] final signals: "
            f"memory={memory_intent:.3f}, "
            f"social={social_intent:.3f}, "
            f"followup={followup_intent:.3f}, "
            f"margin={margin:.3f}, "
            f"history={history_relevance:.3f}, "
            f"zone={history_zone}"
        )

        routes = []
        if memory_route:
            routes.append("memory")
        if followup_route:
            routes.append("followup")
        if ambiguous_followup_route:
            routes.append("ambiguous_followup")

        print(
            "[RAG GATE] route: "
            + ("+".join(routes) if routes else "none")
        )

        if not use_rag or not history_docs:
            print(
                f"{Fore.YELLOW}"
                "[RAG GATE] rejected: "
                "insufficient retrieval evidence"
            )

            return {
                "history": [],
                "profile_summary": self.get_profile_summary(
                    include_alternatives=False
                ),
            }

        # ========================================================
        # Stage 5: extract ALL turns from selected conversations
        # ========================================================

        candidates = []
        per_conversation_limit = 50

        for document in history_docs:
            turns = [
                turn.strip()
                for turn in document.split("[TURN_END]")
                if turn.strip()
            ]

            candidates.extend(
                turns[:per_conversation_limit]
            )

        if not candidates:
            return {
                "history": [],
                "profile_summary": self.get_profile_summary(
                    include_alternatives=False
                ),
            }

        t2 = time.perf_counter()

        print(
            f"[TIMING] preparing "
            f"{len(candidates)} candidates "
            f"{t2 - t1:.3f}s"
        )

        # ========================================================
        # Stage 6: Jina reranking
        # ========================================================

        relevant_turns = []

        try:
            MAX_JINA_CANDIDATES = 150
            MIN_RELEVANCE = 0.20
            RELATIVE_THRESHOLD = 0.75

            candidates = candidates[:MAX_JINA_CANDIDATES]

            rerank_documents = []

            for index, candidate in enumerate(candidates):
                if len(candidate) <= 7000:
                    excerpt = candidate
                else:
                    excerpt = (
                        candidate[:3500]
                        + "\n...[middle omitted for reranking]...\n"
                        + candidate[-3500:]
                    )

                rerank_documents.append(
                    f"[CANDIDATE {index}]\n{excerpt}"
                )

            reranker = self._get_reranker()

            results = reranker.rerank(
                query,
                rerank_documents,
                top_n=min(8, len(rerank_documents)),
            )

            if results:
                top_score = float(
                    results[0].get(
                        "relevance_score",
                        0.0,
                    )
                )

                for result in results:
                    score = float(
                        result.get(
                            "relevance_score",
                            0.0,
                        )
                    )

                    if score < MIN_RELEVANCE:
                        continue

                    if (
                        top_score > 0.0
                        and score < top_score * RELATIVE_THRESHOLD
                    ):
                        break

                    text = result.get(
                        "document",
                        "",
                    )

                    try:
                        index = int(
                            text.split("\n", 1)[0]
                            .removeprefix("[CANDIDATE ")
                            .removesuffix("]")
                        )

                        relevant_turns.append(
                            candidates[index]
                        )

                    except (ValueError, IndexError):
                        continue

                    if len(relevant_turns) >= top_n:
                        break

        except Exception as e:
            print(
                f"{Fore.RED}"
                f"Jina reranking failed: {e}"
            )

        t3 = time.perf_counter()

        print(
            f"[TIMING] reranking "
            f"{t3 - t2:.3f}s"
        )

        # ========================================================
        # Stage 7: final context budget
        # ========================================================

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

