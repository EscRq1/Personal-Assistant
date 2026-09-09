import re
from rapidfuzz import fuzz


CONVERSATIONAL_ANCHORS = {
    "hi", "hello", "hey", "hey there", "yo", "hiya",
    "morning", "good morning", "good afternoon",
    "good evening", "good night",
    "how are you", "how is it going", "how's it going",
    "what's up", "whats up",
    "thanks", "thank you", "thx", "ty",
    "you're welcome", "you are welcome",
    "ok", "okay", "got it", "gotcha", "sure", "yeah", "yep",
    "yes", "no", "nope", "cool", "nice", "great", "awesome",
    "bye", "goodbye", "see you", "see ya",
}


MEMORY_ANCHORS = (
    "remember",
    "do you remember",
    "what did i tell you",
    "what did i say",
    "what did we decide",
    "what did we choose",
    "what was decided",
    "why did we decide",
    "what was my",
    "what were my",
    "when did i",
    "where did i",
    "i told you",
    "i mentioned",
    "you mentioned",
    "we discussed",
    "we talked about",
    "we've discussed",
    "we've talked about",
    "we went over",
    "we covered",
    "had a conversation about",
    "continue from",
    "continue where",
    "last time",
    "earlier you said",
    "you said before",
    "previous conversation",
    "previous chat",
    "previously",
    "as before",
    "like before",
    "that issue",
    "that problem",
    "that thing",
)


TEMPORAL_ANCHORS = (
    "yesterday",
    "today",
    "last time",
    "last week",
    "last month",
    "earlier",
    "before",
    "previously",
    "recently",
)


PERSONAL_ANCHORS = (
    "my ",
    "i'm ",
    "im ",
    "i am ",
    "i have ",
    "i had ",
    "i want ",
    "i need ",
    "for me",
    "about me",
    "for my ",
)


# ------------------------------------------------------------
# Explicit requests to use the retrieval/memory subsystem.
#
# These should bypass the normal semantic gate entirely.
# ------------------------------------------------------------

EXPLICIT_RAG_ANCHORS = (
    "use rag",
    "use the rag",
    "use memory",
    "use my memory",
    "use history",
    "automatic rag",
    "rag pipeline",
    "rag search",
    "rag retrieval",
    "retrieve from rag",
    "retrieve from memory",
    "retrieve from history",
    "search my memory",
    "search memory",
    "search my memories",
    "search previous conversations",
    "search previous conversation",
    "search conversation history",
    "search chat history",
    "look through my memory",
    "look through memory",
    "look through previous conversations",
    "look through previous chats",
    "look through chat history",
    "get information from memory",
    "get info from memory",
    "get information from history",
    "get info from history",
    "get information from rag",
    "get info from rag",
    "retrieve relevant memories",
    "retrieve relevant history",
)


# ------------------------------------------------------------
# Topics that can plausibly correspond to stored personal context.
#
# These DO NOT force RAG by themselves.
# They only become useful when combined with a personal signal.
# ------------------------------------------------------------

PROFILE_TOPIC_ANCHORS = (
    "masters degree",
    "master's degree",
    "masters",
    "master's",
    "master program",
    "master's program",
    "graduate program",
    "graduate school",
    "university",
    "universities",
    "ai/ml",
    "ai ml",
    "machine learning",
    "data science",
    "career",
    "career goal",
    "education",
    "degree",
    "study",
    "studies",
    "application",
    "applications",
    "admission",
    "admissions",
    "gpa",
    "ielts",
    "toefl",
    "scholarship",
    "scholarships",
    "internship",
    "internships",
    "research",
)


GENERAL_KNOWLEDGE_ANCHORS = (
    "what is ",
    "what are ",
    "define ",
    "explain ",
    "how does ",
    "how do ",
    "why does ",
    "why is ",
    "difference between ",
    "compare ",
)


CONTEXTUAL_FOLLOWUP_ANCHORS = (
    "what about",
    "how about",
    "and what about",
    "and how about",
    "what then",
    "then what",
    "what happens then",
    "and then",
    "what about that",
    "what about the other",
    "what about the second",
    "what about the first",
    "what about this",
    "what about that one",
    "how does that work",
    "how would that work",
    "and this",
    "and that",
    "what if instead",
    "what about the alternative",
)


def normalize_query(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



class RAGGate:
    """
    CPU-side retrieval router.

    Routing is deliberately separated from retrieval quality:
      - trivial/social conversation -> no long-term RAG
      - current-turn follow-up -> rely on current chat context
      - explicit memory/RAG request -> retrieve
      - personal historical/profile query -> retrieve
      - generic knowledge -> no automatic RAG

    The gate decides *whether* long-term retrieval is worth doing.
    Chroma/hybrid retrieval + Jina decide *what* is relevant.
    """

    def __init__(self):
        self._social_anchors = tuple(
            normalize_query(x) for x in CONVERSATIONAL_ANCHORS
        )
        self._followup_anchors = tuple(
            normalize_query(x) for x in CONTEXTUAL_FOLLOWUP_ANCHORS
        )
        self._memory_anchors = tuple(
            normalize_query(x) for x in MEMORY_ANCHORS
        )
        self._temporal_anchors = tuple(
            normalize_query(x) for x in TEMPORAL_ANCHORS
        )
        self._personal_anchors = tuple(
            normalize_query(x) for x in PERSONAL_ANCHORS
        )
        self._profile_topic_anchors = tuple(
            normalize_query(x) for x in PROFILE_TOPIC_ANCHORS
        )
        self._explicit_rag_anchors = tuple(
            normalize_query(x) for x in EXPLICIT_RAG_ANCHORS
        )

    def _fuzzy_social_score(self, query: str) -> float:
        if not query:
            return 1.0

        return max(
            fuzz.ratio(query, anchor) / 100.0
            for anchor in self._social_anchors
        )

    def _contextual_followup_score(self, query: str) -> float:
        words = query.split()
        if len(words) < 3:
            return 0.0

        best = 0.0
        for anchor in self._followup_anchors:
            if query.startswith(anchor):
                best = max(best, 1.0)
                continue

            # Avoid fuzzy partial matching on very short utterances.
            if len(words) >= 5:
                best = max(
                    best,
                    fuzz.partial_ratio(query, anchor) / 100.0,
                )

        return best

    def _explicit_memory_score(self, query: str) -> float:
        score = 0.0

        if any(anchor in query for anchor in self._memory_anchors):
            score += 0.75

        if any(anchor in query for anchor in self._temporal_anchors):
            score += 0.20

        if any(anchor in query for anchor in self._personal_anchors):
            score += 0.15

        return min(score, 1.0)

    def _profile_topic_score(self, query: str) -> float:
        matched = sum(
            1
            for anchor in self._profile_topic_anchors
            if anchor in query
        )

        if matched == 0:
            return 0.0

        return min(
            1.0,
            0.45 + 0.15 * min(matched - 1, 3),
        )

    def _explicit_rag_score(self, query: str) -> float:
        if any(anchor in query for anchor in self._explicit_rag_anchors):
            return 1.0
        return 0.0

    def _is_trivial_social(self, query: str) -> bool:
        """Reject greetings/acknowledgements before semantic retrieval."""
        if query in self._social_anchors:
            return True

        words = query.split()
        if len(words) <= 4:
            return self._fuzzy_social_score(query) >= 0.88

        # Allow short variants such as "hi there" / "hey man" but don't
        # classify "hey, what did we decide about X" as trivial.
        trivial_prefixes = (
            "hi ",
            "hello ",
            "hey ",
            "yo ",
            "hiya ",
            "good morning ",
            "good afternoon ",
            "good evening ",
            "how are you ",
            "how's it going ",
            "how is it going ",
            "what's up ",
            "whats up ",
        )

        if any(query.startswith(prefix) for prefix in trivial_prefixes):
            remainder = query.split(" ", 1)[1] if " " in query else ""
            return len(remainder.split()) <= 3

        return False

    def evaluate_basic(self, text: str) -> dict:
        query = normalize_query(text)

        base = {
            "query": query,
            "memory": 0.0,
            "social": 0.0,
            "followup": 0.0,
            "explicit_rag": 0.0,
            "profile_topic": 0.0,
        }

        if not query:
            return {
                **base,
                "route": "none",
                "use_rag": False,
                "force_rag": False,
                "confidence": 0.0,
                "reason": "empty_query",
            }

        # --------------------------------------------------------
        # Stage 0: hard trivial/social stop.
        # --------------------------------------------------------
        fuzzy_social = self._fuzzy_social_score(query)
        if self._is_trivial_social(query):
            return {
                **base,
                "route": "none",
                "use_rag": False,
                "force_rag": False,
                "confidence": 0.0,
                "reason": "trivial_conversation",
                "social": max(0.95, fuzzy_social),
            }

        memory = self._explicit_memory_score(query)
        followup = self._contextual_followup_score(query)
        explicit_rag = self._explicit_rag_score(query)
        profile_topic = self._profile_topic_score(query)
        personal = any(
            anchor in query
            for anchor in self._personal_anchors
        )

        # --------------------------------------------------------
        # Explicit retrieval always wins.
        # --------------------------------------------------------
        if explicit_rag >= 1.0:
            return {
                **base,
                "route": "memory",
                "use_rag": True,
                "force_rag": True,
                "confidence": 1.0,
                "reason": "explicit_rag_request",
                "memory": max(memory, 1.0),
                "social": fuzzy_social,
                "followup": followup,
                "explicit_rag": explicit_rag,
                "profile_topic": profile_topic,
            }

        # --------------------------------------------------------
        # Current-context follow-up is NOT long-term RAG.
        # Qwen already receives the current chat turns.
        # --------------------------------------------------------
        if (
            followup >= 0.80
            and memory < 0.75
            and not personal
        ):
            return {
                **base,
                "route": "current_context",
                "use_rag": False,
                "force_rag": False,
                "confidence": followup,
                "reason": "current_context_followup",
                "memory": memory,
                "social": fuzzy_social,
                "followup": followup,
                "explicit_rag": explicit_rag,
                "profile_topic": profile_topic,
            }

        # --------------------------------------------------------
        # Personal topics with personal language are eligible for RAG.
        # Generic questions like "what is a master's degree?" are not.
        # --------------------------------------------------------
        if (
            profile_topic >= 0.45
            and personal
            and (memory >= 0.25 or followup >= 0.50)
        ):
            confidence = max(
                profile_topic,
                memory,
                followup,
            )

            return {
                **base,
                "route": "memory",
                "use_rag": True,
                "force_rag": False,
                "confidence": confidence,
                "reason": "personal_profile_topic",
                "memory": memory,
                "social": fuzzy_social,
                "followup": followup,
                "explicit_rag": explicit_rag,
                "profile_topic": profile_topic,
            }

        # --------------------------------------------------------
        # Explicit historical-language requests.
        # --------------------------------------------------------
        if (
            memory >= 0.55
            and memory > fuzzy_social + 0.08
        ):
            return {
                **base,
                "route": "memory",
                "use_rag": True,
                "force_rag": False,
                "confidence": memory,
                "reason": "memory_intent",
                "memory": memory,
                "social": fuzzy_social,
                "followup": followup,
                "explicit_rag": explicit_rag,
                "profile_topic": profile_topic,
            }

        # Everything else stays local. General knowledge should not
        # automatically spend VRAM on historical retrieval.
        return {
            **base,
            "route": "none",
            "use_rag": False,
            "force_rag": False,
            "confidence": 0.0,
            "reason": "no_memory_route",
            "memory": memory,
            "social": fuzzy_social,
            "followup": followup,
            "explicit_rag": explicit_rag,
            "profile_topic": profile_topic,
        }

