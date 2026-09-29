"""Model-facing execution and offline packages. Backend supplies all context/paths."""
from __future__ import annotations
import logging
import threading
import shlex
import secrets
import time
from typing import Any, Protocol
from sandbox import SandboxManager
from sandbox.result import cap_text
from sandbox.package_cache import PackageError, validate_requests

log = logging.getLogger(__name__)
_manager: SandboxManager | None = None
_manager_lock = threading.Lock()


MANUAL_APPROVAL_TTL_SECONDS = 15 * 60

_manual_approvals_lock = threading.Lock()
_manual_approvals: dict[str, dict[str, Any]] = {}


def _purge_manual_approvals_locked(now: float) -> None:
    expired = [
        token
        for token, approval in _manual_approvals.items()
        if (
            approval["expires_at"] <= now
            and not approval["in_use"]
        )
    ]

    for token in expired:
        _manual_approvals.pop(token, None)


def create_manual_package_approval(
    session_id: str,
    packages: list[str],
) -> str:
    token = secrets.token_urlsafe(32)
    now = time.monotonic()

    with _manual_approvals_lock:
        _purge_manual_approvals_locked(now)

        _manual_approvals[token] = {
            "session_id": session_id,
            "packages": list(packages),
            "expires_at": now + MANUAL_APPROVAL_TTL_SECONDS,
            "in_use": False,
        }

    return token


def claim_manual_package_approval(
    token: str,
    session_id: str,
) -> list[str] | None:
    if (
        not isinstance(token, str)
        or not token
        or len(token) > 256
        or not isinstance(session_id, str)
        or not session_id
    ):
        return None

    now = time.monotonic()

    with _manual_approvals_lock:
        _purge_manual_approvals_locked(now)

        approval = _manual_approvals.get(token)

        if approval is None:
            return None

        if approval["expires_at"] <= now:
            _manual_approvals.pop(token, None)
            return None

        if approval["session_id"] != session_id:
            return None

        if approval["in_use"]:
            return None

        approval["in_use"] = True

        return list(approval["packages"])


def release_manual_package_approval(
    token: str,
    session_id: str,
) -> None:
    with _manual_approvals_lock:
        approval = _manual_approvals.get(token)

        if (
            approval is not None
            and approval["session_id"] == session_id
        ):
            approval["in_use"] = False


def consume_manual_package_approval(
    token: str,
    session_id: str,
) -> None:
    with _manual_approvals_lock:
        approval = _manual_approvals.get(token)

        if (
            approval is not None
            and approval["session_id"] == session_id
        ):
            _manual_approvals.pop(token, None)


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
            "are available. Shell composition is supported; default timeout is 30 seconds. "
            "Use sandbox_install for approved Python packages; runtime pip is offline. "
            "After installation python uses /workspace/.venv; use python -m pytest for tests."
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

SANDBOX_INSTALL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "sandbox_install",
        "description": (
            "Install approved Python wheels into this chat's persistent /workspace/.venv. "
            "Runs offline in the same isolated sandbox; no internet downloads. Packages must "
            "already be in the operator's approved cache. Names, extras and version constraints "
            "are accepted, for example requests or requests>=2. Use sandbox_exec afterwards. "
            "If package_not_approved is returned, stop installation and show the user "
            "the provided session_id and manual_script exactly as returned. Ask the user "
            "to run that script. Do not attempt another download or network method yourself. "
            "Reset removes this environment. Default installation timeout is 120 seconds."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "packages": {"type": "array", "minItems": 1, "maxItems": 20,
                             "items": {"type": "string", "maxLength": 200}},
                "timeout": {"type": "integer", "minimum": 1, "maximum": 300},
            },
            "required": ["packages"],
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

def build_manual_package_script(
    packages: list[str],
    session_id: str,
    approval_token: str,
) -> str:
    quoted_packages = " ".join(
        shlex.quote(package)
        for package in packages
    )

    quoted_session = shlex.quote(session_id)
    quoted_token = shlex.quote(approval_token)

    return f"""#!/usr/bin/env bash
set -euo pipefail

SESSION_ID={quoted_session}
APPROVAL_TOKEN={quoted_token}

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

PIP_CONFIG_FILE=/dev/null \\
python3 -m pip download \\
    --disable-pip-version-check \\
    --no-cache-dir \\
    --index-url https://pypi.org/simple \\
    --only-binary=:all: \\
    --dest "$TMPDIR" \\
    {quoted_packages}

CURL_ARGS=()

while IFS= read -r -d '' wheel; do
    CURL_ARGS+=(
        -F
        "wheel=@${{wheel}};type=application/octet-stream"
    )
done < <(
    find "$TMPDIR" \\
        -maxdepth 1 \\
        -type f \\
        -name '*.whl' \\
        -print0
)

curl \\
    --fail \\
    --silent \\
    --show-error \\
    -H "X-Sandbox-Approval: $APPROVAL_TOKEN" \\
    "${{CURL_ARGS[@]}}" \\
    "http://127.0.0.1:5000/sandbox/manual-packages/$SESSION_ID"

echo
echo "Package installation completed for sandbox session: $SESSION_ID"
echo "Return to the chat and tell the assistant the installation completed."
"""

def sandbox_install(
    args: dict[str, Any],
    ctx: ToolContext | None,
) -> dict[str, Any]:

    # ------------------------------------------------------------
    # 1. Validate model-facing arguments
    # ------------------------------------------------------------

    if (
        not isinstance(args, dict)
        or set(args) - {"packages", "timeout"}
    ):
        return {
            "error": "invalid_arguments",
            "message": (
                "Only packages and timeout are accepted."
            ),
        }

    if (
        ctx is None
        or not (
            ctx.sandbox_context_id
            or ctx.chat_id
        )
    ):
        return {
            "error": "missing_context",
            "message": (
                "No backend chat context is attached."
            ),
        }

    if ctx.cancel_event.is_set():
        return {
            "cancelled": True,
            "exit_code": None,
        }

    packages: list[str] | None = None
    session = None

    try:
        # --------------------------------------------------------
        # 2. Validate package requirement strings
        # --------------------------------------------------------

        packages = validate_requests(
            args.get("packages")
        )

        manager = get_manager()

        timeout = args.get("timeout")

        if timeout is not None:
            # Validation only.
            manager.config.timeout(timeout)

        # --------------------------------------------------------
        # 3. Resolve the real persistent sandbox session
        # --------------------------------------------------------

        context_id = (
            ctx.sandbox_context_id
            or ctx.chat_id
        )

        session = manager.get_or_create_for_context(
            context_id,
            ctx.cancel_event,
        )

        # --------------------------------------------------------
        # 4. Try normal approved-cache installation
        # --------------------------------------------------------

        result = manager.install(
            session,
            packages,
            timeout,
            ctx.cancel_event,
            caller="assistant/coder:packages",
        )

        payload = result.to_dict()

        # Keep model-facing tool output bounded.
        for field in ("stdout", "stderr"):
            payload[field], truncated = cap_text(
                payload[field],
                manager.config.model_output_bytes // 2,
            )

            payload[field + "_truncated"] |= truncated

        payload.update(
            requested=packages,
            environment="/workspace/.venv",
            session_id=session.id,
        )

        return payload

    # ------------------------------------------------------------
    # 5. Safe package/cache failures
    # ------------------------------------------------------------

    except PackageError as exc:

        # The package itself is syntactically valid, but is absent from
        # the immutable operator-approved cache.
        #
        # Instead of giving the model network access, create a temporary
        # one-time user approval and return a host-side download/upload
        # script for the assistant to show to the user.
        if (
            exc.code == "package_not_approved"
            and session is not None
            and packages is not None
        ):
            approval_token = (
                create_manual_package_approval(
                    session.id,
                    packages,
                )
            )

            script = build_manual_package_script(
                packages,
                session.id,
                approval_token,
            )

            return {
                "error": "package_not_approved",

                "message": (
                    "The requested package/version is not "
                    "in the approved package cache. "
                    "The user may manually approve it for "
                    "this sandbox session."
                ),

                "requested": packages,

                # This is the actual sandbox session ID,
                # not the chat/context identifier.
                "session_id": session.id,

                "environment": "/workspace/.venv",

                "manual_script": script,

                "approval_expires_in_seconds": (
                    MANUAL_APPROVAL_TTL_SECONDS
                ),

                "assistant_instruction": (
                    "The requested Python dependency is not "
                    "in the approved package cache. "
                    "Tell the user that manual approval is "
                    "required. Show the session_id and "
                    "manual_script exactly as provided. "
                    "Ask the user to run the script on the "
                    "host and then return to the chat. "
                    "Do not attempt another package download, "
                    "do not enable sandbox networking, and do "
                    "not invent another installation method."
                ),
            }

        # Other PackageError values remain ordinary safe
        # model-facing failures.
        return {
            "error": exc.code,
            "message": str(exc),
        }

    # ------------------------------------------------------------
    # 6. Other validation failures
    # ------------------------------------------------------------

    except ValueError as exc:
        return {
            "error": "invalid_arguments",
            "message": str(exc),
        }

    # ------------------------------------------------------------
    # 7. Unexpected backend failure
    # ------------------------------------------------------------

    except Exception:
        log.exception(
            "sandbox_install failed"
        )

        return {
            "error": "sandbox_unavailable",
            "message": (
                "Package installation unavailable; "
                "consult backend logs."
            ),
        }


def delete_context(chat_id: str) -> str | None:
    """Backend-only: permanently delete the sandbox session bound to a chat.

    Safe to call for every chat deletion: returns None when the chat never
    owned a sandbox or it was already destroyed. Raises RuntimeError only if
    a command is still executing inside it (the caller should retry after
    stopping generation).
    """
    with _manager_lock:
        if _manager is None:
            return None
        return _manager.delete_for_context(chat_id)