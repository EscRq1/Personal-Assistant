import json
from typing import Any

from colorama import Fore

from .tool_registry import get_tool


# Maximum number of tool calls the model may request
# in a single generation phase.
MAX_TOOL_CALLS_PER_ITERATION = 8


def parse_arguments(
    raw_arguments: str,
) -> tuple[dict[str, Any], str | None]:
    """
    Parse the JSON arguments emitted by the model.

    Returns:
        ({...}, None) on success
        ({}, "error message") on failure
    """

    if not isinstance(raw_arguments, str):
        raw_arguments = str(raw_arguments or "")

    if not raw_arguments.strip():
        return {}, None

    try:
        arguments = json.loads(raw_arguments)

    except json.JSONDecodeError as exc:
        return {}, f"Invalid JSON tool arguments: {exc}"

    if not isinstance(arguments, dict):
        return {}, "Tool arguments must decode to a JSON object."

    return arguments, None


def execute_tool(
    name: str,
    arguments: dict[str, Any],
) -> str:
    """
    Execute one registered tool.

    The actual implementation lives in the registered function.
    This layer only dispatches to it and normalizes the result.
    """

    tool = get_tool(name)

    if tool is None:
        print(
            f"{Fore.RED}"
            f"[TOOL] unknown tool: {name}"
        )
        return json.dumps(
            {
                "error": f"Unknown tool: {name}"
            },
            ensure_ascii=False,
        )

    print(
        f"{Fore.MAGENTA}"
        f"[TOOL] executing: {name}"
    )

    print(
        f"{Fore.MAGENTA}"
        f"[TOOL] arguments: {arguments}"
    )

    try:
        result = tool(**arguments)

    except Exception as exc:
        print(
            f"{Fore.RED}"
            f"[TOOL] {name} failed: "
            f"{type(exc).__name__}: {exc}"
        )

        return json.dumps(
            {
                "error": (
                    f"{type(exc).__name__}: {exc}"
                )
            },
            ensure_ascii=False,
        )

    # Tools may directly return prepared text/JSON.
    if isinstance(result, str):
        return result

    # Otherwise serialize structured results.
    try:
        return json.dumps(
            result,
            ensure_ascii=False,
            default=str,
        )

    except Exception as exc:
        print(
            f"{Fore.RED}"
            f"[TOOL] failed to serialize result "
            f"from {name}: {exc}"
        )

        return json.dumps(
            {
                "error": (
                    "Failed to serialize tool result: "
                    f"{type(exc).__name__}: {exc}"
                )
            },
            ensure_ascii=False,
        )


def execute_tool_call(
    tool_call: dict,
) -> dict:
    """
    Execute one fully reconstructed OpenAI-compatible tool call.

    Returns a normalized result that can be converted into:

        {
            "role": "tool",
            "tool_call_id": "...",
            "name": "...",
            "content": "..."
        }
    """

    call_id = tool_call.get("id", "")

    function = tool_call.get("function") or {}

    name = function.get("name", "")

    raw_arguments = function.get(
        "arguments",
        "{}",
    )

    # --------------------------------------------------------
    # Validate function name.
    # --------------------------------------------------------

    if not name:
        return {
            "tool_call_id": call_id,
            "name": "",
            "content": json.dumps(
                {
                    "error": (
                        "Tool call has no function name."
                    )
                },
                ensure_ascii=False,
            ),
        }

    # --------------------------------------------------------
    # Parse arguments.
    # --------------------------------------------------------

    arguments, parse_error = parse_arguments(
        raw_arguments
    )

    if parse_error:
        print(
            f"{Fore.RED}"
            f"[TOOL] {name}: {parse_error}"
        )

        return {
            "tool_call_id": call_id,
            "name": name,
            "content": json.dumps(
                {
                    "error": parse_error
                },
                ensure_ascii=False,
            ),
        }

    # --------------------------------------------------------
    # Execute.
    # --------------------------------------------------------

    content = execute_tool(
        name,
        arguments,
    )

    return {
        "tool_call_id": call_id,
        "name": name,
        "content": content,
    }


def execute_tool_calls(
    tool_calls: list[dict],
) -> list[dict]:
    """
    Execute all tool calls requested by one model generation.

    Tool calls are intentionally executed sequentially for now.

    That gives us:
        - predictable ResourceManager behavior
        - easier debugging
        - deterministic ordering
        - no accidental concurrent GPU ownership

    We can introduce selective parallelism later.
    """

    if not tool_calls:
        return []

    if len(tool_calls) > MAX_TOOL_CALLS_PER_ITERATION:
        print(
            f"{Fore.YELLOW}"
            f"[TOOL] limiting "
            f"{len(tool_calls)} calls to "
            f"{MAX_TOOL_CALLS_PER_ITERATION}"
        )

        tool_calls = tool_calls[
            :MAX_TOOL_CALLS_PER_ITERATION
        ]

    results = []

    for tool_call in tool_calls:
        results.append(
            execute_tool_call(tool_call)
        )

    return results


def build_tool_message(
    result: dict,
) -> dict:
    """
    Convert an execution result into the OpenAI-compatible
    role='tool' message sent back to the model.
    """

    return {
        "role": "tool",
        "tool_call_id": result.get(
            "tool_call_id",
            "",
        ),
        "name": result.get(
            "name",
            "",
        ),
        "content": result.get(
            "content",
            "",
        ),
    }