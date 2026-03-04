import os
import json
import uuid
import time
from colorama import Fore, Style, init
import chromadb
from chromadb.config import Settings

# Environment setup for offline/privacy mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_OFF"] = "True"
os.environ["POSTHOG_DISABLED"] = "1"

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
    "thanks", "thank you", "thx", "ty",
    "ok", "okay", "got it", "nice", "cool", "great",
    "yes", "yep", "no", "nope", "sure"
}


class MemoryManager:
    def __init__(self, db_path=None):
        if db_path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(base, "memory_db")

        self.client = chromadb.Client(Settings(
            persist_directory=db_path,
            chroma_db_impl="duckdb+parquet",
            anonymized_telemetry=False
        ))

        # Raw chat memory (high-volume)
        self.history = self.client.get_or_create_collection(name="chat_history")

        # Profile memory (low-volume, structured)
        self.profile_col = self.client.get_or_create_collection(name="user_profile")

        # In-memory cache of profile as dict
        self.profile_cache = self._load_profile()
        
        self._persist()
        self.last_profile_update_ts = 0

    # ---------- Persistence helpers ----------
    def _persist(self):
        """Persist if available. Some Chroma versions manage persistence automatically."""
        try:
            if hasattr(self.client, "persist"):
                self.client.persist()
        except Exception:
            pass

    def _load_profile(self) -> dict:
        """Load JSON profile from Chroma (single doc)."""
        try:
            data = self.profile_col.get(ids=["user_core_profile"])
            if data and data.get("documents"):
                raw = data["documents"][0]
                return json.loads(raw) if raw else {}
        except Exception:
            pass
        return {}

    def _save_profile(self):
        """Save JSON profile to Chroma (single doc)."""
        try:
            payload = json.dumps(self.profile_cache, ensure_ascii=False)
            self.profile_col.upsert(
                documents=[payload],
                ids=["user_core_profile"]
            )
            self._persist()
        except Exception as e:
            print(f"{Fore.RED}[Profile Save Error]: {e}")

    # ---------- History ----------
    def add_history(self, text: str, summary: dict):
        doc_id = str(uuid.uuid4())
        self.history.add(documents=[text], ids=[doc_id], metadatas=[summary])
        self._persist()
        return doc_id
    def delete_history(self, chat_id: str) -> bool:
        chat_id = (chat_id or "").strip()
        if not chat_id:
            return False
        try:
            self.history.delete(ids=[chat_id])
            self._persist()
            return True
        except Exception as e:
            print(f"{Fore.RED}[Delete History Error]: {e}")
            return False
        
    def rename_history(self, chat_id: str, new_title: str) -> bool:
        chat_id = (chat_id or "").strip()
        new_title = (new_title or "").strip()
        if not chat_id or not new_title:
            return False

        data = self.history.get(ids=[chat_id], include=["documents", "metadatas"])
        if not data.get("ids"):
            return False

        doc = data["documents"][0]
        meta = data["metadatas"][0] or {}

        meta["title"] = new_title

        # upsert is safer than update across Chroma versions
        self.history.upsert(ids=[chat_id], documents=[doc], metadatas=[meta])
        self._persist()
        return True


    def toggle_pin_history(self, chat_id: str) -> int | None:
        chat_id = (chat_id or "").strip()
        if not chat_id:
            return None

        data = self.history.get(ids=[chat_id], include=["documents", "metadatas"])
        if not data.get("ids"):
            return None

        doc = data["documents"][0]
        meta = data["metadatas"][0] or {}

        # Chroma metadata can’t store bool; use 0/1
        current = int(meta.get("pinned", 0))
        meta["pinned"] = 0 if current == 1 else 1

        self.history.upsert(ids=[chat_id], documents=[doc], metadatas=[meta])
        self._persist()
        return meta["pinned"]

    def prune_memories(self):
        """Delete very short junk history items."""
        existing = self.history.get()
        if not existing.get("ids"):
            return

        ids_to_delete = [
            _id for i, _id in enumerate(existing["ids"])
            if len(existing["documents"][i]) < 15
        ]
        if ids_to_delete:
            self.history.delete(ids=ids_to_delete)
            print(f"{Fore.RED}Cleaned {len(ids_to_delete)} junk history items.")
            self._persist()

    # ---------- Profile merge (the primary writer) ----------
    def merge_profile_fact(self, field: str, value: str, confidence: float = 0.7) -> bool:
        field = (field or "").strip()
        value = (value or "").strip()

        # Validate
        if field not in ALLOWED_PROFILE_FIELDS:
            return False
        if len(value) < 2 or value.endswith("?"):
            return False
        if confidence < 0.6:
            return False

        vlow = value.lower().strip()
        if vlow in FEEDBACK_ONLY:
            return False

        # Cooldown (prevents repeated writes)
        now = time.time()
        if getattr(self, "last_profile_update_ts", 0.0) and (now - self.last_profile_update_ts) < 180:
            return False

        existing = self.profile_cache.get(field)
        if existing and float(confidence) < float(existing.get("confidence", 0.0)) - 0.10:
            # new info is much less reliable than what we already believe
            return False

        # New field
        if not existing:
            self.profile_cache[field] = {
                "value": value, 
                "confidence": float(confidence), 
                "alternatives": []
            }
            self._save_profile()
            self.last_profile_update_ts = now
            return True

        # Same value: only update if confidence increased
        if existing.get("value") == value:
            old_conf = float(existing.get("confidence", 0.0))
            new_conf = max(old_conf, float(confidence))
            if new_conf == old_conf:
                return False
            existing["confidence"] = new_conf
            self.profile_cache[field] = existing
            self._save_profile()
            self.last_profile_update_ts = now
            return True

        # Different value -> alternatives (dedup)
        alts = existing.get("alternatives", [])
        for alt in alts:
            if alt.get("value") == value:
                old_conf = float(alt.get("confidence", 0.0))
                new_conf = max(old_conf, float(confidence))
                if new_conf == old_conf:
                    return False
                alt["confidence"] = new_conf
                existing["alternatives"] = alts
                self.profile_cache[field] = existing
                self._save_profile()
                self.last_profile_update_ts = now
                return True

        alts.append({"value": value, "confidence": float(confidence)})
        existing["alternatives"] = alts
        self.profile_cache[field] = existing
        self._save_profile()
        self.last_profile_update_ts = now
        return True

    def get_profile_summary(self, include_alternatives: bool = False) -> str:
        """
        Returns a clean human-readable summary for the system prompt.
        """
        if not self.profile_cache:
            return ""

        lines = []
        # fixed ordering (nice UX)
        order = [
            "Name", "Education", "Field_of_Study",
            "Career_Goal", "Current_Status", "Preferences", "Location"
        ]

        for key in order:
            if key not in self.profile_cache:
                continue
            entry = self.profile_cache[key]
            main_val = entry.get("value", "")
            if not main_val:
                continue

            lines.append(f"{key}: {main_val}")

            if include_alternatives:
                alts = entry.get("alternatives", [])
                if alts:
                    # show top 2 alternatives
                    top_alts = sorted(alts, key=lambda x: x.get("confidence", 0), reverse=True)[:2]
                    for a in top_alts:
                        v = a.get("value")
                        if v:
                            lines.append(f"  - alt: {v}")

        return "\n".join(lines)

    # ---------- Context retrieval ----------
    def get_raw_context(self, query: str, k: int = 15):
        """
        Retrieve relevant history snippets + profile summary.
        """
        history_docs = []
        try:
            hist_count = self.history.count()
            if hist_count > 0:
                results = self.history.query(
                    query_texts=[query],
                    n_results=min(k, hist_count)
                )
                history_docs = results["documents"][0]
        except Exception as e:
            print(f"{Fore.RED}Error fetching history: {e}")

        profile_summary = self.get_profile_summary(include_alternatives=False)

        return {
            "history": history_docs,
            "profile_summary": profile_summary
        }