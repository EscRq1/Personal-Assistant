"""The sole model-facing sandbox capability. Context is supplied by the backend."""
from __future__ import annotations
import logging
import threading
from typing import Any, Protocol
from sandbox import SandboxManager
from sandbox.result import cap_text

log = logging.getLogger(__name__)
_manager: SandboxManager | None = None
_manager_lock = threading.Lock()


class ToolContext(Protocol):
    cancel_event: threading.Event
    chat_id: str | None
    sandbox_context_id: str | None


SANDBOX_EXEC_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "sandbox_exec",
        "description": (
            "Run a Bash command in the current chat's copied /workspace. Use ls/find/grep/cat, "
            "edit files, run Python/pytest or git. Files persist between calls; each command "
            "gets a fresh non-root container, so shell variables, /tmp and background processes "
            "do not persist. No network or host file access. Only files imported by the backend "
            "are available. Shell composition is supported; default timeout is 30 seconds."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Bash command; cwd is /workspace."},
                "timeout": {"type": "integer", "minimum": 1, "maximum": 300},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}


def get_manager() -> SandboxManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = SandboxManager()
        return _manager


def bind_saved_chat(previous_context: str, chat_id: str) -> None:
    # Saving a chat that has never used the sandbox must not require Docker/setup.
    with _manager_lock:
        if _manager is not None:
            _manager.bind_context(previous_context, chat_id)


def sandbox_exec(args: dict[str, Any], ctx: ToolContext | None) -> dict[str, Any]:
    if not isinstance(args, dict) or set(args) - {"command", "timeout"}:
        return {"error": "invalid_arguments", "message": "Only command and timeout are accepted."}
    if ctx is None or not (ctx.sandbox_context_id or ctx.chat_id):
        return {"error": "missing_context", "message": "No backend chat context is attached."}
    if ctx.cancel_event.is_set():
        return {"cancelled": True, "exit_code": None}
    try:
        manager = get_manager()
        # Validate BEFORE session creation or other side effects.
        command = args.get("command")
        if not isinstance(command, str) or not command.strip() or "\0" in command:
            raise ValueError("command must be a nonempty string without NUL")
        if len(command.encode()) > manager.config.max_command_bytes:
            raise ValueError("Command is too long")
        timeout = manager.config.timeout(args.get("timeout"))
        session = manager.get_or_create_for_context(ctx.sandbox_context_id or ctx.chat_id, ctx.cancel_event)
        result = manager.exec(session, command, timeout, ctx.cancel_event, caller="assistant/coder")
        payload = result.to_dict()
        for field in ("stdout", "stderr"):
            payload[field], truncated = cap_text(payload[field], manager.config.model_output_bytes // 2)
            payload[field + "_truncated"] |= truncated
        return payload
    except ValueError as exc:
        return {"error": "invalid_arguments", "message": str(exc)}
    except Exception:
        log.exception("sandbox_exec failed")
        # Do not put Docker errors or backend paths into the model context.
        return {"error": "sandbox_unavailable", "message": "Sandbox unavailable; consult backend logs."}
