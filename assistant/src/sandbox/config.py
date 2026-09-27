"""Trusted backend configuration; none of these settings are tool arguments."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SandboxConfig:
    image: str = "assistant-sandbox:latest"
    docker_binary: str = "docker"
    memory_limit: str = "2g"
    cpu_limit: float = 2.0
    pids_limit: int = 128
    tmpfs_size: str = "256m"
    default_timeout: int = 30
    max_timeout: int = 300
    control_timeout: int = 15
    max_stdout_bytes: int = 1_000_000
    max_stderr_bytes: int = 1_000_000
    model_output_bytes: int = 24_000
    max_command_bytes: int = 65_536
    uid: int = field(default_factory=os.getuid)
    gid: int = field(default_factory=os.getgid)
    git_enabled: bool = True
    git_max_file_bytes: int = 2_000_000
    git_max_total_bytes: int = 32_000_000
    max_import_bytes: int = 512_000_000
    max_import_files: int = 20_000
    audit_enabled: bool = True
    wheel_cache_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("SANDBOX_WHEEL_CACHE", str(Path.home() / ".assistant/sandbox-wheels"))))
    package_install_timeout: int = 120

    def __post_init__(self) -> None:
        if self.uid <= 0 or self.gid <= 0:
            raise ValueError("Run the backend as an unprivileged Linux user (UID/GID > 0).")
        if self.uid != os.getuid() or self.gid != os.getgid():
            raise ValueError("Container UID/GID must match the backend user for bind ownership.")
        for name in ("memory_limit", "tmpfs_size"):
            if not re.fullmatch(r"[1-9][0-9]*[kmg]?", getattr(self, name)):
                raise ValueError(f"Invalid {name}")
        for name in ("pids_limit", "default_timeout", "max_timeout", "control_timeout",
                     "max_stdout_bytes", "max_stderr_bytes", "model_output_bytes",
                     "max_command_bytes", "git_max_file_bytes", "git_max_total_bytes",
                     "max_import_bytes", "max_import_files", "package_install_timeout"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.cpu_limit <= 256 or self.default_timeout > self.max_timeout:
            raise ValueError("Invalid CPU or timeout limits")
        if not self.image or self.image.startswith("-") or any(c.isspace() for c in self.image):
            raise ValueError("Invalid image reference")
        if not isinstance(self.wheel_cache_dir, Path):
            raise ValueError("wheel_cache_dir must be a pathlib.Path")

    def timeout(self, value: int | None) -> int:
        if value is None:
            return self.default_timeout
        if type(value) is not int or not 1 <= value <= self.max_timeout:
            raise ValueError(f"timeout must be an integer from 1 to {self.max_timeout}")
        return value
