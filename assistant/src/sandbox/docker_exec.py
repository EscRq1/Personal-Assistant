"""One named, disposable container per command. No docker exec or host shell."""
from __future__ import annotations
import csv
import hashlib
import io
import logging
import threading
import time
import uuid
from pathlib import Path
from .config import SandboxConfig
from .executor import run_bounded
from .result import ExecutionResult
from .session import SandboxSession

log = logging.getLogger(__name__)
CONTAINER_PREFIX = "assistant-sandbox-exec-"
LABEL = "io.assistant.sandbox.owner"


class DockerExecutor:
    def __init__(self, config: SandboxConfig, base_dir: Path) -> None:
        self.config = config
        self.owner = hashlib.sha256(str(base_dir).encode()).hexdigest()[:24]
        self._active: set[str] = set()
        self._lock = threading.Lock()

    def create_args(self, session: SandboxSession, name: str, command: str) -> list[str]:
        c = self.config
        stream = io.StringIO()
        csv.writer(stream, lineterminator="").writerow([
            "type=bind", f"src={session.workspace}", "dst=/workspace",
            "bind-propagation=rprivate", "bind-recursive=disabled",
        ])
        return [c.docker_binary, "create", "--pull=never", "--name", name,
                "--label", f"{LABEL}={self.owner}", "--label", f"io.assistant.sandbox.session={session.id}",
                "--hostname", "sandbox", "--network", "none", "--ipc", "private", "--read-only",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
                "--pids-limit", str(c.pids_limit), "--memory", c.memory_limit,
                "--memory-swap", c.memory_limit, "--cpus", str(c.cpu_limit),
                "--ulimit", "nofile=1024:1024", "--ulimit", "core=0:0",
                "--tmpfs", f"/tmp:rw,nosuid,nodev,size={c.tmpfs_size},mode=1777",
                "--shm-size", "64m", "--mount", stream.getvalue(),
                "--workdir", "/workspace", "--user", f"{c.uid}:{c.gid}",
                "--env", "HOME=/tmp", "--env", "TMPDIR=/tmp",
                "--env", "MPLCONFIGDIR=/tmp/matplotlib", "--env", "PYTHONUNBUFFERED=1",
                # Override Docker CLI proxy defaults, which can contain host credentials.
                "--env", "HTTP_PROXY=", "--env", "HTTPS_PROXY=", "--env", "ALL_PROXY=",
                "--env", "NO_PROXY=", "--env", "http_proxy=", "--env", "https_proxy=",
                "--env", "all_proxy=", "--env", "no_proxy=",
                "--env", "FTP_PROXY=", "--env", "ftp_proxy=",
                "--log-driver", "none", "--restart", "no", "--stop-timeout", "0",
                "--entrypoint", "/bin/bash", c.image, "--noprofile", "--norc", "-c", command]

    def _control(self, *args: str) -> ExecutionResult:
        return run_bounded([self.config.docker_binary, *args],
                           timeout=self.config.control_timeout,
                           stdout_limit=32_000, stderr_limit=16_000)

    def _remove(self, name: str) -> bool:
        # Verify absence; a successful client exit alone is insufficient if the
        # daemon disappeared during cleanup. Retry one failed removal.
        for _ in range(2):
            removed = self._control("rm", "--force", "--volumes", name)
            if removed.exit_code == 0:
                return True
            if "No such container" in removed.stderr:
                return True
        log.error("Could not remove sandbox execution %s", name)
        return False

    def exec(self, session: SandboxSession, command: str, timeout: int,
             cancel_event: threading.Event | None = None) -> ExecutionResult:
        c = self.config
        name = CONTAINER_PREFIX + uuid.uuid4().hex
        start = time.monotonic()
        result = ExecutionResult("", "", None, 0)
        if cancel_event and cancel_event.is_set():
            result.cancelled = True
            return result
        with self._lock:
            self._active.add(name)
        attempted = False
        uncertain_create = False
        try:
            attempted = True
            # Finish create before start, so cancellation cannot race a late
            # docker run create/start. create is bounded by control_timeout.
            created = run_bounded(self.create_args(session, name, command),
                                  timeout=c.control_timeout, stdout_limit=1024,
                                  stderr_limit=c.max_stderr_bytes)
            uncertain_create = created.timed_out
            if created.exit_code != 0:
                result.error = "container_create_failed"
                result.stderr = "Could not create sandbox container; check Docker and the configured image."
                log.error("Sandbox create failed: %s", created.stderr)
            elif cancel_event and cancel_event.is_set():
                result.cancelled = True
            else:
                result = run_bounded([c.docker_binary, "start", "--attach", name],
                                     timeout=timeout, stdout_limit=c.max_stdout_bytes,
                                     stderr_limit=c.max_stderr_bytes, cancel_event=cancel_event)
                if not result.timed_out and not result.cancelled:
                    status = self._control("inspect", "--format",
                                           "{{.State.Status}} {{.State.ExitCode}}", name)
                    fields = status.stdout.strip().split()
                    if status.exit_code == 0 and len(fields) == 2 and fields[0] == "exited":
                        result.exit_code = int(fields[1])
                    else:
                        result.error = "container_status_unavailable"
                        result.exit_code = None
        except (OSError, ValueError, TimeoutError):
            log.exception("Sandbox Docker execution failed")
            result.error = "docker_unavailable"
            result.stderr = "Docker execution failed; check backend logs and Docker availability."
        finally:
            cleaned = not attempted
            if attempted:
                try:
                    cleaned = self._remove(name)
                except (OSError, TimeoutError):
                    log.exception("Sandbox cleanup failed for %s", name)
            with self._lock:
                self._active.discard(name)
            if not cleaned or uncertain_create:
                result.error = "cleanup_failed"
                result.stderr += "\nContainer removal could not be confirmed; workspace is quarantined."
            result.duration_ms = int((time.monotonic() - start) * 1000)
        return result

    def reconcile(self) -> list[str]:
        """Backend-only, before serving requests, with exclusive ownership of base_dir."""
        with self._lock:
            if self._active:
                raise RuntimeError("Reconcile must run before accepting sandbox work")
            listed = self._control("ps", "--all", "--filter", f"label={LABEL}={self.owner}",
                                   "--format", "{{.Names}}")
            if listed.exit_code != 0 or listed.stdout_truncated:
                raise RuntimeError("Unable to list stale sandbox containers")
            removed: list[str] = []
            for name in listed.stdout.splitlines():
                if name.startswith(CONTAINER_PREFIX):
                    if not self._remove(name):
                        raise RuntimeError("Unable to remove a stale sandbox container")
                    removed.append(name)
            return removed
