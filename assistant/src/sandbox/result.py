from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any

TRUNCATION_MARKER = "\n[OUTPUT TRUNCATED]"


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int | None
    duration_ms: int
    timed_out: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    cancelled: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BoundedOutput:
    """Store at most limit bytes; callers continue draining/discarding the pipe."""
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.data = bytearray()
        self.truncated = False

    def feed(self, chunk: bytes) -> None:
        available = max(0, self.limit - len(self.data))
        self.data.extend(chunk[:available])
        self.truncated |= len(chunk) > available

    def text(self) -> str:
        return self.data.decode("utf-8", errors="replace") + (
            TRUNCATION_MARKER if self.truncated else ""
        )


def cap_text(text: str, limit: int) -> tuple[str, bool]:
    data = text.encode("utf-8")
    if len(data) <= limit:
        return text, False
    marker = TRUNCATION_MARKER.encode()
    return data[:max(0, limit - len(marker))].decode("utf-8", "ignore") + TRUNCATION_MARKER, True
