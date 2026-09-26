from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class SessionState(str, Enum):
    READY = "ready"
    RUNNING = "running"
    BROKEN = "broken"
    DESTROYED = "destroyed"


@dataclass
class SandboxSession:
    id: str
    workspace: Path
    created_at: float = field(default_factory=time.time)
    state: SessionState = SessionState.READY

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex
