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
    Cheap CPU-side intent classifier.

    It does not decide whether useful history exists. Chroma does that.
    It estimates whether a query is conversational, explicit-memory,
    contextual-followup, or ambiguous.
    """

    def __init__(self):
        self._social_anchors = tuple(
            normalize_query(x) for x in CONVERSATIONAL_ANCHORS
        )
        self._followup_anchors = tuple(
            normalize_query(x) for x in CONTEXTUAL_FOLLOWUP_ANCHORS
        )

    def _fuzzy_social_score(self, query: str) -> float:
        if not query:
            return 1.0

        return max(
            fuzz.ratio(query, anchor) / 100.0
            for anchor in self._social_anchors
        )

    def _contextual_followup_score(self, query: str) -> float:
        if not query:
            return 0.0

        words = query.split()
        if len(words) < 3:
            return 0.0

        best = 0.0

        for anchor in self._followup_anchors:
            # Strongest signal: query literally starts with the anchor.
            if query.startswith(anchor):
                best = max(best, 1.0)
                continue

            # For longer queries, partial matching handles prefixes such as
            # "ok then what about ..." without making "hi there" a follow-up.
            if len(words) >= 4:
                score = fuzz.partial_ratio(
                    query,
                    anchor,
                ) / 100.0
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
                "memory": 0.0,
                "social": 0.0,
                "followup": 0.0,
            }

        words = query.split()

        if query in self._social_anchors:
            return {
                "use_rag": False,
                "confidence": 0.0,
                "reason": "exact_conversational_anchor",
                "query": query,
                "memory": 0.0,
                "social": 1.0,
                "followup": 0.0,
            }

        fuzzy_social = self._fuzzy_social_score(query)

        if len(words) <= 4 and fuzzy_social >= 0.84:
            return {
                "use_rag": False,
                "confidence": 0.02,
                "reason": "fuzzy_conversational_anchor",
                "query": query,
                "memory": 0.0,
                "social": fuzzy_social,
                "followup": 0.0,
            }

        memory = self._explicit_memory_score(query)
        followup = self._contextual_followup_score(query)
        penalty = self._general_knowledge_penalty(query)

        confidence = memory - penalty

        if len(words) >= 8:
            confidence += 0.05

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
            "followup": followup,
        }

