import json
import re

def estimate_tokens(text, tokenizer=None) -> int:
    if text is None:
        return 0

    if not isinstance(text, str):
        try:
            text = json.dumps(
                text,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            text = str(text)

    if not text:
        return 0

    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass

    # Rough fallback:
    # ~1 token per 4 characters for ordinary text,
    # with a little extra room for whitespace/punctuation.
    char_estimate = (len(text) + 3) // 4

    whitespace_count = len(re.findall(r"\s+", text))
    punctuation_count = len(re.findall(r"[^\w\s]", text, flags=re.UNICODE))

    structural_padding = (
        whitespace_count // 2 +
        punctuation_count // 4
    )

    return max(1, char_estimate + structural_padding)