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
            "name": "high_thinking",
            "description": (
                "Transfer control to coding mode using high reasoning effort. "
                "Use this for programming, debugging, code analysis, implementation, "
                "algorithm design, or software architecture tasks. Use this when the "
                "task requires deeper technical reasoning than normal assistant mode. "
                "Coding mode becomes responsible for solving the task and may use "
                "web_call, memory_call, and vision_call when needed. "
                "Do not use this merely because the request contains code; use it "
                "when dedicated coding reasoning would improve the result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "The exact programming or technical problem to solve."
                        )
                    },
                    "context": {
                        "type": "string",
                        "description": (
                            "Relevant code, logs, requirements, constraints, "
                            "and useful information gathered during the current turn."
                        )
                    }
                },
                "required": ["task", "context"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "xhigh_thinking",
            "description": (
                "Escalate the current coding task to extreme reasoning effort. "
                "Use this ONLY when coding mode has already attempted the problem "
                "with high reasoning and the problem remains unresolved, ambiguous, "
                "or requires unusually deep multi-step analysis. "
                "Do NOT use this for ordinary coding tasks, simple implementations, "
                "routine debugging, or tasks that can reasonably be solved with high "
                "reasoning. This is an expensive last-resort escalation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "The unresolved coding problem that requires deeper "
                            "reasoning."
                        )
                    },
                    "context": {
                        "type": "string",
                        "description": (
                            "The relevant code, logs, previous reasoning results, "
                            "tool results, attempted solutions, and constraints "
                            "already considered during high-reasoning coding mode."
                        )
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "Explain briefly why high-reasoning coding mode was "
                            "insufficient and why extreme reasoning is justified."
                        )
                    }
                },
                "required": ["task", "context", "reason"]
            }
        }
    },
]


TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {}

CONTROL_TOOLS = {
    "high_thinking",
    "xhigh_thinking"

}

def register_tool(name: str, function: Callable[..., Any]) -> None:
    TOOL_FUNCTIONS[name] = function


def register_control_tool(name: str) -> None:
    CONTROL_TOOLS.add(name)


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
        or tool["function"]["name"] in CONTROL_TOOLS
    ]

