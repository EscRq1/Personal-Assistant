def format_memories(memories):
    """
    Format a list of memory strings for better readability in the prompt.
    Adds a dash and line break for each memory.
    """
    if not memories:
        return "No relevant memories found."
    return "\n".join([f"- {m}" for m in memories])

def clean_text(text):
    return " ".join(text.strip().split())