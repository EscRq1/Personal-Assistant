from typing import Callable, Any
from sandbox_tools import SANDBOX_EXEC_SCHEMA, SANDBOX_INSTALL_SCHEMA

SUMMARY_PROMPT = {
        "summary": {
            "type": "string",
            "description": "A short, human-readable description of what this specific tool call is about to do. Describe the concrete action and its target, not the generic purpose of the tool. Use a concise present-participle phrase (3–12 words), such as 'Searching the web for NASA satellite datasets', 'Retrieving previous discussion about GPU configuration', 'Executing Python code to analyze CSV data', or 'Delegating database schema debugging to the coder'. Avoid vague descriptions, internal tool names, unnecessary details, and claims that the action has already succeeded."
        }
    }


TOOLS = [
    SANDBOX_EXEC_SCHEMA,
    SANDBOX_INSTALL_SCHEMA,
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
                "detail is missing, use memory_call specifically to retrieve that detail.\n\n"

                "You control the maximum amount of selected historical memory returned "
                "by choosing max_memory_tokens. Choose the smallest budget that is likely "
                "to contain enough information. Use a small budget for a specific fact "
                "or decision, a medium budget for broader discussion or technical context, "
                "and a large budget when full implementation details, long code, or "
                "multiple historical turns may be needed. If the result is insufficient "
                "or truncated, call memory_call again with a larger budget or a more "
                "specific request."
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
                    },
                    "max_memory_tokens": {
                        "type": "integer",
                        "description": (
                            "Maximum token budget for the selected historical memory "
                            "returned by this tool. Choose based on how much historical "
                            "detail is needed. Smaller values are appropriate for specific "
                            "facts or decisions; larger values are appropriate for long "
                            "technical discussions, source code, or several relevant turns."
                        ),
                        "minimum": 500,
                        "maximum": 30000,
                        "default": 10000
                    }
                },
                "required": ["request"],
                "additionalProperties": False
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
                "Delegate a coding or technical task to the coding specialist. "
                "The coder receives the parent assistant's existing conversation context, "
                "including relevant user messages, previous discussion, tool calls, "
                "tool results, source code, and logs. "
                "Do not use the delegation fields to repeat information the coder can already see. "
                "The parent assistant should provide the exact task and only any additional "
                "handoff guidance that is genuinely useful. "
                "The coder can use its available tools to retrieve additional context "
                "when the inherited context is insufficient."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "A concise statement of the exact task the coding specialist "
                            "must perform."
                        ),
                    },
                    "context": {
                        "type": "string",
                        "description": (
                            "A concise handoff with only additional context the coder needs "
                            "to understand or execute the task. "
                            "Do not paraphrase or repeat the user's request, source code, logs, "
                            "conversation history, or tool results—the coder can already see them. "
                            "Do not manually include tool results in this field; tool calls and "
                            "their results are already available to the coder. "
                            "If the inherited context is insufficient, the coder can use the "
                            "available tools to retrieve additional context. "
                            "Use this field only for important interpretation, constraints, "
                            "decisions, or guidance that is not already available to the coder. "
                            "Leave empty when no additional context is needed."
                        ),
                    },
                },
                "required": [
                    "task",
                    "context",
                ],
                "additionalProperties": False,
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
                        "maximum": 30000,
                        "default": 10000,
                    }
                },
                "required": ["request"],
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

for tool in TOOLS:
    parameters = tool["function"]["parameters"]

    parameters.setdefault("properties", {}).update(SUMMARY_PROMPT)

    required = parameters.setdefault("required", [])
    if "summary" not in required:
        required.append("summary")

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

