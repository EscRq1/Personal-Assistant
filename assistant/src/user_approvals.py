"""Request-scoped user decisions. Capabilities go only to the UI SSE stream."""
import secrets
import threading
import time
from dataclasses import dataclass, field

APPROVAL_SECONDS = 300


def has_desktop_access(ctx):
    return bool(ctx is not None and getattr(ctx, 'desktop_access', False)
                and not ctx.cancel_event.is_set())


TOOL_ACCESS = {
    "web": ({"web_call"}, "ask_web_access"),
    "memory": ({"memory_call"}, "ask_memory_access"),
    "desktop": ({"desktop_act", "desktop_observe"}, "ask_desktop_access"),
}


def has_tool_access(ctx, kind):
    return bool(ctx is not None and getattr(ctx, kind + "_access", False)
                and not ctx.cancel_event.is_set())


def filter_desktop_tools(tools, ctx):
    result = list(tools)
    for kind, (actual, request) in TOOL_ACCESS.items():
        allowed = has_tool_access(ctx, kind)
        result = [tool for tool in result
                  if (tool['function']['name'] not in actual or allowed)
                  and (tool['function']['name'] != request or not allowed)]
    return result


@dataclass
class Pending:
    ctx: object
    token: str
    expires: float
    event: threading.Event = field(default_factory=threading.Event)
    decision: bool | None = None


class ApprovalBroker:
    def __init__(self):
        self.lock = threading.Lock()
        self.pending = {}

    def request(self, ctx, kind, details):
        if ctx is None or ctx.cancel_event.is_set():
            return 'cancelled'
        seconds = min(APPROVAL_SECONDS, getattr(ctx, 'approval_timeout_seconds', APPROVAL_SECONDS))
        if seconds <= 0:
            return 'expired'
        request_id = secrets.token_urlsafe(24)
        pending = Pending(ctx, secrets.token_urlsafe(32), time.monotonic() + seconds)
        with self.lock:
            self.pending[request_id] = pending
        ctx.stream_queue.put(('token', {'approval_request': {
            'id': request_id, 'token': pending.token, 'kind': kind,
            'details': details, 'expires_in_seconds': seconds,
        }}))
        status = 'expired'
        try:
            while time.monotonic() < pending.expires:
                if ctx.cancel_event.is_set():
                    status = 'cancelled'
                    break
                if pending.event.wait(0.05):
                    status = 'approved' if pending.decision else 'denied'
                    break
            if ctx.cancel_event.is_set():
                status = 'cancelled'
            return status
        finally:
            with self.lock:
                self.pending.pop(request_id, None)
            ctx.stream_queue.put(('token', {'approval_resolved': {'id': request_id, 'status': status}}))

    def decide(self, request_id, token, allow):
        if not isinstance(request_id, str) or not isinstance(token, str) or type(allow) is not bool:
            return False
        with self.lock:
            pending = self.pending.get(request_id)
            if (pending is None or pending.event.is_set() or pending.ctx.cancel_event.is_set()
                    or time.monotonic() >= pending.expires
                    or not secrets.compare_digest(pending.token, token)):
                return False
            pending.decision = allow
            pending.event.set()
            return True


approvals = ApprovalBroker()

ASK_DESKTOP_SCHEMA = {
    'type': 'function', 'function': {
        'name': 'ask_desktop_access',
        'description': 'Ask the user to allow desktop observation and input for this turn. Waits for their decision. A denial does not authorize desktop use. Do not repeat a denied request unless the user asks.',
        'parameters': {'type': 'object', 'properties': {
            'reason': {'type': 'string', 'description': 'Explain the intended desktop task to the user.'}
        }, 'required': ['reason'], 'additionalProperties': False},
    },
}


def ask_desktop_access(ctx, reason):
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
        return {'error': 'invalid_arguments', 'message': 'Provide a reason of 1–1000 characters.'}
    if has_desktop_access(ctx):
        return {'status': 'approved', 'scope': 'current_turn'}
    status = approvals.request(ctx, 'desktop', {'reason': reason, 'scope': 'current_turn'})
    if status == 'approved' and not ctx.cancel_event.is_set():
        ctx.desktop_access = True
    return {'status': status, 'scope': 'current_turn',
            'message': 'Desktop tools are now available.' if has_desktop_access(ctx)
            else 'Desktop access was not granted. Do not use desktop tools.'}


def access_schema(kind):
    return {"type": "function", "function": {
        "name": TOOL_ACCESS[kind][1],
        "description": f"Ask the user to allow {kind} access for this turn. Waits for approval. Do not repeat a denied request unless the user asks.",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string", "description": "Explain why this capability is needed."}
        }, "required": ["reason"], "additionalProperties": False},
    }}


ASK_WEB_SCHEMA = access_schema("web")
ASK_MEMORY_SCHEMA = access_schema("memory")


def ask_tool_access(ctx, kind, reason):
    if kind not in TOOL_ACCESS or not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
        return {"error": "invalid_arguments", "message": "Provide a reason of 1–1000 characters."}
    if has_tool_access(ctx, kind):
        return {"status": "approved", "scope": "current_turn"}
    status = approvals.request(ctx, kind, {"reason": reason, "scope": "current_turn"})
    if status == "approved" and ctx is not None and not ctx.cancel_event.is_set():
        setattr(ctx, kind + "_access", True)
    return {"status": status, "scope": "current_turn",
            "message": f"{kind.title()} tools are now available." if has_tool_access(ctx, kind)
            else f"{kind.title()} access was not granted. Do not use these tools."}
