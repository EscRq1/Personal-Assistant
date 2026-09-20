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
    # preserved for better times !!!DO NOT DELETE!!!
    # { 
    #     "type": "function",
    #     "function": {
    #         "name": "high_thinking",
    #         "description": (
    #             "Transfer control to coding mode using high reasoning effort. "
    #             "Use this for programming, debugging, code analysis, implementation, "
    #             "algorithm design, or software architecture tasks. Use this when the "
    #             "task requires deeper technical reasoning than normal assistant mode. "
    #             "Coding mode becomes responsible for solving the task and may use "
    #             "web_call, memory_call, and vision_call when needed. "
    #             "Do not use this merely because the request contains code; use it "
    #             "when dedicated coding reasoning would improve the result."
    #         ),
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "task": {
    #                     "type": "string",
    #                     "description": (
    #                         "The exact programming or technical problem to solve."
    #                     )
    #                 },
    #                 "context": {
    #                     "type": "string",
    #                     "description": (
    #                         "Relevant code, logs, requirements, constraints, "
    #                         "and useful information gathered during the current turn."
    #                     )
    #                 }
    #             },
    #             "required": ["task", "context"]
    #         }
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "xhigh_thinking",
    #         "description": (
    #             "Escalate the current coding task to extreme reasoning effort. "
    #             "Use this ONLY when coding mode has already attempted the problem "
    #             "with high reasoning and the problem remains unresolved, ambiguous, "
    #             "or requires unusually deep multi-step analysis. "
    #             "Do NOT use this for ordinary coding tasks, simple implementations, "
    #             "routine debugging, or tasks that can reasonably be solved with high "
    #             "reasoning. This is an expensive last-resort escalation."
    #         ),
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "task": {
    #                     "type": "string",
    #                     "description": (
    #                         "The unresolved coding problem that requires deeper "
    #                         "reasoning."
    #                     )
    #                 },
    #                 "context": {
    #                     "type": "string",
    #                     "description": (
    #                         "The relevant code, logs, previous reasoning results, "
    #                         "tool results, attempted solutions, and constraints "
    #                         "already considered during high-reasoning coding mode."
    #                     )
    #                 },
    #                 "reason": {
    #                     "type": "string",
    #                     "description": (
    #                         "Explain briefly why high-reasoning coding mode was "
    #                         "insufficient and why extreme reasoning is justified."
    #                     )
    #                 }
    #             },
    #             "required": ["task", "context", "reason"]
    #         }
    #     }
    # },
    {
        "type": "function",
        "function": {
            "name": "write_notebook",
            "description": (
                "Update the assistant's persistent global notebook. "
                "The notebook is a small, long-term memory maintained by the "
                "assistant itself for information that may remain useful across "
                "future conversations.\n\n"

                "Provide the COMPLETE replacement contents of the notebook, "
                "not an individual note or an append operation. "
                "The previous notebook contents will be replaced entirely.\n\n"

                "Only store information that is likely to remain useful across "
                "future conversations. Do not store temporary reasoning, "
                "speculation, unsupported assumptions, transient task details, "
                "or information that is already adequately represented by "
                "user_profile or conversational memory.\n\n"

                "Keep the notebook concise and information-dense. "
                "Prefer stable facts, important project context, architectural "
                "decisions, persistent preferences, recurring constraints, and "
                "other information that would materially improve future responses.\n\n"

                "HARD LIMIT: The complete notebook MUST NOT exceed 2048 tokens. "
                "If the existing notebook is already near the limit, rewrite it "
                "by removing outdated or lower-value information rather than "
                "adding more content. Never intentionally exceed this limit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "notes": {
                        "type": "string",
                        "description": (
                            "The complete replacement contents of the persistent "
                            "assistant notebook. Keep it concise, information-dense, "
                            "and strictly within the 2048-token hard limit."
                        )
                    }
                },
                "required": ["notes"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "call_coder",
            "description": (
                "Delegate programming and software-engineering work to the "
                "dedicated coding specialist. IMPORTANT: If the user's request "
                "involves writing, modifying, debugging, analyzing, explaining, "
                "reviewing, refactoring, optimizing, or designing code or a "
                "software system, prefer calling this tool rather than solving "
                "the coding task yourself. This includes code errors, stack "
                "traces, compiler/build errors, runtime errors, API integration, "
                "architecture, algorithms, implementation details, performance "
                "problems, configuration issues, and code-related technical "
                "questions. The task does NOT need to be difficult or complex "
                "to justify delegation. Call the coder whenever the core of the "
                "user's request is a programming or software-engineering problem. "
                "Do not call it for purely conceptual questions unrelated to "
                "implementing or debugging software, or for ordinary non-technical "
                "requests."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "The exact programming or software-engineering task "
                            "the coding specialist should solve. State what needs "
                            "to be implemented, debugged, analyzed, changed, or "
                            "explained and what the expected result is."
                        ),
                    },
                    "context": {
                        "type": "string",
                        "description": (
                            "All relevant technical context available to the "
                            "specialist, including source code, error messages, "
                            "stack traces, logs, environment details, APIs, "
                            "requirements, constraints, attempted solutions, "
                            "and relevant previous discussion."
                        ),
                    },
                },
                "required": ["task"],
            },
        },
    },
    {
    "type": "function",
    "function": {
        "name": "search_current_chat",
        "description": (
            "Search the complete current conversation for earlier turns "
            "that are relevant to the request. Use this when information "
            "was discussed earlier in THIS chat but is outside the visible "
            "recent context. The tool searches the current chat only; it "
            "does not search previous conversations or persistent memory. "
            "Use a concise natural-language description of what you need. "
            "The result preserves chronological turn boundaries and turn "
            "indices. You control how much context to retrieve by choosing "
            "max_tokens. Prefer a smaller budget for a specific lookup and "
            "a larger budget when you need broader surrounding discussion. "
            "If the result is truncated, you can call the tool again with "
            "a larger max_tokens value or with a more specific request."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string",
                        "description": (
                            "Describe the specific information, decision, "
                            "discussion, or earlier detail you need to find "
                            "in the current conversation."
                        )
                    },
                    "tokens": {
                        "type": "integer",
                        "description": (
                            "Maximum number of tokens of conversation context to return. "
                            "Choose the budget based on how much context you expect to need."
                        ),
                        "minimum": 500,
                        "maximum": 30000
                    }
                },
                "required": ["request", "tokens"],
            }
        }
    },
    {
    "type": "function",
    "function": {
        "name": "get_chat_turn",
        "description": (
            "Retrieve a specific turn from the current conversation by its "
            "index. Use this when search_current_chat identified a relevant "
            "turn index (via matched_turns or skipped_turns) but did not "
            "return its full content. The index is 0-based and refers to "
            "the turn position in the current chat."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "turn_index": {
                        "type": "integer",
                        "description": (
                            "The 0-based index of the turn to retrieve. "
                            "This index comes from search_current_chat results."
                        )
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": (
                            "Maximum number of tokens to return. "
                            "If the turn is longer, it will be truncated. "
                            "Default is 8000."
                        ),
                        "minimum": 500,
                        "maximum": 30000,
                        "default": 8000
                    }
                },
                "required": ["turn_index"]
            }
        }
    },
]

TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {}

CONTROL_TOOLS = {
    # "high_thinking",
    # "xhigh_thinking",
    "call_coder",

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

