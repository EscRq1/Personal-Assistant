from typing import Callable, Any


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_call",
            "description": (
                "Retrieve current or externally verifiable information "
                "from the public web."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string",
                        "description": (
                            "Describe the information you need from the web. "
                            "Do not formulate the search-engine query yourself."
                        )
                    }
                },
                "required": ["request"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "memory_call",
            "description": (
                "Search previous conversations and persistent user memory for "
                "information that is NOT already available in the current conversation "
                "or provided context.\n\n"
                "Use this tool ONLY when the information needed to answer the user's "
                "request is missing, incomplete, ambiguous, or insufficient in the "
                "current context.\n\n"
                "Do NOT use this tool when the required information is already present "
                "in the current conversation, system-provided context, retrieved RAG "
                "context, tool results, or the user's latest message.\n\n"
                "Do NOT use this tool merely because the topic is personal or relates "
                "to something the user may have discussed previously.\n\n"
                "If the provided RAG context already contains the requested information, "
                "use that information directly and do not call memory_call.\n\n"
                "If the provided information is partially sufficient but an important "
                "detail is missing, you may use memory_call specifically to retrieve "
                "that missing detail."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string",
                        "description": (
                            "Describe the specific missing historical information you need. "
                            "Ask only for information that is not already available in "
                            "the current context."
                        )
                    }
                },
                "required": ["request"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "vision_call",
            "description": (
                "Ask the Observer to inspect a user-provided image and "
                "return the relevant visual information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": (
                            "What visual information should be extracted."
                        )
                    }
                },
                "required": ["instruction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "Return a simple test response.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string"
                    }
                },
                "required": ["message"]
            }
        }
    },
]


TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {}


def register_tool(name: str, function: Callable[..., Any]) -> None:
    TOOL_FUNCTIONS[name] = function


def get_tool(name: str) -> Callable[..., Any] | None:
    return TOOL_FUNCTIONS.get(name)

def test_tool(message: str) -> str:
    return f"TEST TOOL RESULT: {message}"

# register_tool("test_tool", test_tool)

def get_registered_tools() -> list[dict]:
    return [
        tool
        for tool in TOOLS
        if tool["function"]["name"] in TOOL_FUNCTIONS
    ]

