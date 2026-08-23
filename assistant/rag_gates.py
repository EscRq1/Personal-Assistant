import re
from rapidfuzz import fuzz


CONVERSATIONAL_ANCHORS = {
    "hi",
    "hello",
    "hey",
    "hey there",
    "yo",
    "hiya",
    "morning",
    "good morning",
    "good afternoon",
    "good evening",
    "good night",
    "how are you",
    "how is it going",
    "how's it going",
    "what's up",
    "whats up",

    "thanks",
    "thank you",
    "thx",
    "ty",
    "you're welcome",
    "you are welcome",

    "ok",
    "okay",
    "got it",
    "gotcha",
    "sure",
    "yeah",
    "yep",
    "yes",
    "no",
    "nope",
    "cool",
    "nice",
    "great",
    "awesome",

    "bye",
    "goodbye",
    "see you",
    "see ya",
}

# Strong indications that the user is asking about something
# from previous conversation(s).
MEMORY_ANCHORS = (
    "remember",
    "do you remember",
    "what did i tell you",
    "what did i say",
    "what was my",
    "what were my",
    "when did i",
    "where did i",
    "i told you",
    "i mentioned",
    "we discussed",
    "we talked about",
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


def normalize_query(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class RAGGate:
    """
    Cheap CPU-side decision layer.

    It intentionally does NOT try to determine whether useful memory
    definitely exists. It only estimates whether paying the cost of
    semantic memory retrieval + Jina is justified.
    """

    def __init__(self):
        self._social_anchors = tuple(
            normalize_query(x) for x in CONVERSATIONAL_ANCHORS
        )

    def _fuzzy_social_score(self, query: str) -> float:
        if not query:
            return 1.0

        best = 0.0

        for anchor in self._social_anchors:
            score = fuzz.ratio(query, anchor) / 100.0
            best = max(best, score)

        return best

    def _explicit_memory_score(self, query: str) -> float:
        score = 0.0

        if any(anchor in query for anchor in MEMORY_ANCHORS):
            score += 0.75

        if any(anchor in query for anchor in TEMPORAL_ANCHORS):
            score += 0.20

        if any(anchor in query for anchor in PERSONAL_ANCHORS):
            score += 0.15

        return min(score, 1.0)

    def _general_knowledge_penalty(self, query: str) -> float:
        if any(query.startswith(anchor) for anchor in GENERAL_KNOWLEDGE_ANCHORS):
            return 0.15

        return 0.0

    def evaluate_basic(self, text: str) -> dict:
        query = normalize_query(text)

        if not query:
            return {
                "use_rag": False,
                "confidence": 0.0,
                "reason": "empty_query",
                "query": query,
            }

        words = query.split()

        # Exact social utterance.
        if query in self._social_anchors:
            return {
                "use_rag": False,
                "confidence": 0.0,
                "reason": "exact_conversational_anchor",
                "query": query,
            }

        fuzzy_social = self._fuzzy_social_score(query)

        # Very short fuzzy matches are strong evidence of casual
        # conversation and shouldn't wake Jina.
        if len(words) <= 4 and fuzzy_social >= 0.84:
            return {
                "use_rag": False,
                "confidence": 0.02,
                "reason": "fuzzy_conversational_anchor",
                "query": query,
                "social": fuzzy_social,
            }

        memory = self._explicit_memory_score(query)
        penalty = self._general_knowledge_penalty(query)

        confidence = memory - penalty

        # Longer queries are less likely to be a simple greeting.
        if len(words) >= 8:
            confidence += 0.05

        # A social-looking query gets a modest penalty,
        # but does not automatically kill RAG.
        if fuzzy_social >= 0.75:
            confidence -= 0.15

        confidence = max(0.0, min(confidence, 1.0))

        return {
            "use_rag": confidence >= 0.25,
            "confidence": confidence,
            "reason": "basic_gate",
            "query": query,
            "memory": memory,
            "social": fuzzy_social,
        }

    def combine(
        self,
        basic: dict,
        semantic_memory: float,
        semantic_social: float,
        history_relevance: float,
    ) -> dict:
        query = basic["query"]

        memory_intent = max(
            basic.get("memory", 0.0),
            semantic_memory,
        )

        social_intent = max(
            basic.get("social", 0.0),
            semantic_social,
        )

        margin = max(-1.0, min(
            memory_intent - social_intent,
            1.0,
        ))

        # The three signals have different jobs:
        #
        # memory_intent:
        #   does the query sound like a request for past context?
        #
        # margin:
        #   does it sound more like memory retrieval than conversation?
        #
        # history_relevance:
        #   does Chroma actually have something resembling the query?
        #
        confidence = (
            0.45 * memory_intent
            + 0.20 * max(margin, 0.0)
            + 0.35 * history_relevance
        )

        confidence = max(0.0, min(confidence, 1.0))

        # Require independent evidence.
        #
        # This is intentionally conservative:
        # semantic intent alone isn't enough.
        use_rag = (
            memory_intent >= 0.45
            and margin >= 0.08
            and history_relevance >= 0.12
            and confidence >= 0.40
        )

        reason = (
            f"memory={memory_intent:.3f}, "
            f"social={social_intent:.3f}, "
            f"margin={margin:.3f}, "
            f"history={history_relevance:.3f}, "
            f"confidence={confidence:.3f}"
        )

        return {
            "use_rag": use_rag,
            "confidence": confidence,
            "reason": reason,
            "query": query,
            "memory_intent": memory_intent,
            "social_intent": social_intent,
            "intent_margin": margin,
            "history_relevance": history_relevance,
        }