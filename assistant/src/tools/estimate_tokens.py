import json
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
            return len(tokenizer.encode(text))

        # Conservative fallback
        return max(1, (len(text) + 2) // 3)