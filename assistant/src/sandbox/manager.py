"""Small backend API: persistent copied workspaces, per-session execution locks."""
from __future__ import annotations
import fcntl
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import tempfile
import math
from pathlib import Path
from .config import SandboxConfig
from .docker_exec import DockerExecutor
from .git_baseline import init_command
from .result import ExecutionResult
from .session import SandboxSession, SessionState
from .workspace import WorkspaceManager, remove_tree

log = logging.getLogger(__name__)


class SandboxManager:
    def __init__(self, config: SandboxConfig | None = None, base_dir: Path | None = None,
                 executor: DockerExecutor | None = None) -> None:
        self.config = config or SandboxConfig()
        self.workspaces = WorkspaceManager(base_dir or Path.home() / ".assistant/sandbox", self.config)
        self.base_dir = self.workspaces.base_dir
        # One backend process owns a storage root. Per-session locks serialize
        # commands inside that process; independent sessions remain concurrent.
        self._owner_file = (self.base_dir / ".owner.lock").open("a")
        try:
            fcntl.flock(self._owner_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._owner_file.close()
            raise RuntimeError("Another backend owns this sandbox storage") from None
        self.executor = executor or DockerExecutor(self.config, self.base_dir)
        self._lock = threading.RLock()
        self._sessions: dict[str, SandboxSession] = {}
        self._session_locks: dict[str, threading.RLock] = {}
        self._active_cancel: dict[str, threading.Event] = {}
        self._closed = False
        mapping = self.base_dir / "contexts.json"
        self._contexts: dict[str, str] = json.loads(mapping.read_text()) if mapping.exists() else {}

    def _write_json(self, path: Path, value: object) -> None:
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)

    def _save(self, session: SandboxSession) -> None:
        self._write_json(self.workspaces.path(session.id) / "session.json",
                         {"id": session.id, "created_at": session.created_at, "state": session.state.value})

    @staticmethod
    def _context_key(context_id: str) -> str:
        if not isinstance(context_id, str) or not context_id:
            raise ValueError("A backend chat context is required")
        return hashlib.sha256(context_id.encode()).hexdigest()

    def create_session(self, approved_paths: list[Path] | None = None,
                       cancel_event: threading.Event | None = None) -> SandboxSession:
        """Trusted backend API; approved_paths must never originate in tool arguments."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Sandbox manager is closed")
            sid = SandboxSession.new_id()
            workspace = self.workspaces.create_workspace(sid)
            session = SandboxSession(sid, workspace)
            self._sessions[sid] = session
            self._session_locks[sid] = threading.RLock()
            self._save(session)
        try:
            if approved_paths:
                self.workspaces.import_files(session.id, approved_paths)
            if self.config.git_enabled:
                self._initialize_git(session, cancel_event)
            return session
        except BaseException:
            # Keep quarantined workspaces if cleanup could not be confirmed.
            if session.state != SessionState.BROKEN:
                self.destroy(session)
            raise

    def get_session(self, session_id: str) -> SandboxSession | None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Sandbox manager is closed")
            self.workspaces.path(session_id)  # Validate before opening any path.
            if session_id not in self._sessions:
                path = self.workspaces.path(session_id) / "session.json"
                if not path.exists():
                    return None
                data = json.loads(path.read_text())
                state = SessionState(data["state"])
                # A crash during execution requires reconciliation before reuse.
                if state == SessionState.RUNNING:
                    state = SessionState.BROKEN
                self._sessions[session_id] = SandboxSession(session_id,
                    self.workspaces.path(session_id) / "workspace", float(data["created_at"]), state)
                self._session_locks[session_id] = threading.RLock()
            session = self._sessions[session_id]
            return None if session.state == SessionState.DESTROYED else session

    def get_or_create_for_context(self, context_id: str,
                                  cancel_event: threading.Event | None = None) -> SandboxSession:
        key = self._context_key(context_id)
        with self._lock:
            sid = self._contexts.get(key)
            session = self.get_session(sid) if sid else None
            if session is None:
                session = self.create_session(cancel_event=cancel_event)
                self._contexts[key] = session.id
                self._write_json(self.base_dir / "contexts.json", self._contexts)
            return session

    def bind_context(self, previous_context: str, saved_chat_id: str) -> None:
        """Preserve a new conversation's workspace when chat persistence assigns an ID."""
        old, new = self._context_key(previous_context), self._context_key(saved_chat_id)
        with self._lock:
            if old in self._contexts:
                previous = self._contexts.get(new)
                if previous is not None and previous != self._contexts[old]:
                    raise ValueError("Saved chat already owns a different sandbox")
                self._contexts[new] = self._contexts[old]
                self._write_json(self.base_dir / "contexts.json", self._contexts)

    def _session_lock(self, session: SandboxSession) -> threading.RLock:
        with self._lock:
            if self.get_session(session.id) is not session:
                raise ValueError("Unknown or destroyed sandbox session")
            return self._session_locks[session.id]

    def exec(self, session: SandboxSession, command: str, timeout: int | None = None,
             cancel_event: threading.Event | None = None, caller: str | None = None) -> ExecutionResult:
        if not isinstance(command, str) or not command.strip() or "\0" in command:
            raise ValueError("command must be a nonempty string without NUL")
        if len(command.encode()) > self.config.max_command_bytes:
            raise ValueError("Command is too long")
        seconds = self.config.timeout(timeout)
        lock = self._session_lock(session)
        event = cancel_event if cancel_event is not None else threading.Event()
        while not lock.acquire(timeout=0.05):
            if event.is_set():
                return ExecutionResult("", "", None, 0, cancelled=True)
        try:
            if event.is_set():
                return ExecutionResult("", "", None, 0, cancelled=True)
            if session.state != SessionState.READY:
                raise RuntimeError("Sandbox is not ready; reconcile quarantined executions first")
            self.workspaces.validate(session.id, session.workspace)
            with self._lock:
                self._active_cancel[session.id] = event
            session.state = SessionState.RUNNING
            try:
                self._save(session)
                result = self.executor.exec(session, command, seconds, event)
            except BaseException:
                session.state = SessionState.BROKEN
                self._save(session)
                raise
            else:
                session.state = SessionState.BROKEN if result.error == "cleanup_failed" else SessionState.READY
                self._save(session)
                self._audit(session, command, result, caller)
                return result
            finally:
                with self._lock:
                    self._active_cancel.pop(session.id, None)
        finally:
            lock.release()

    def cancel(self, session: SandboxSession) -> None:
        """Cancel current command only. The executor observes this within 50 ms."""
        with self._lock:
            event = self._active_cancel.get(session.id)
            if event:
                event.set()

    def install(self, session: SandboxSession, packages: list[str], timeout: int | None = None,
                cancel_event: threading.Event | None = None,
                caller: str | None = None) -> ExecutionResult:
        """Install approved wheels offline under the SAME lock as exec/import/reset.

        No host pip, package imports or network operations. Staging is outside
        the mount and published by rename, never through model-created links.
        """
        from .package_cache import WheelCache, PackageCancelled, check_cancel, validate_requests
        from .packages import install_command

        requested = validate_requests(packages)
        seconds = self.config.timeout(timeout if timeout is not None else
                                      min(self.config.package_install_timeout, self.config.max_timeout))
        event = cancel_event if cancel_event is not None else threading.Event()
        lock = self._session_lock(session)
        while not lock.acquire(timeout=0.05):
            if event.is_set():
                return ExecutionResult("", "", None, 0, cancelled=True)
        start = time.monotonic()
        try:
            check_cancel(event)
            if session.state != SessionState.READY:
                raise RuntimeError("Sandbox is not ready")
            self.workspaces.validate(session.id, session.workspace)
            with self._lock:
                self._active_cancel[session.id] = event
            with tempfile.TemporaryDirectory(prefix="packages-", dir=self.workspaces.path(session.id)) as temporary:
                staging = Path(temporary)
                wheels = staging / "wheels"
                wheels.mkdir(mode=0o700)
                WheelCache(self.config.wheel_cache_dir, self.config.image).stage(wheels, requested, event)
                check_cancel(event)
                target = session.workspace / ".packages"
                previous = staging / "previous"
                if os.path.lexists(target):
                    target.rename(previous)  # Move a symlink itself, never follow it.
                try:
                    wheels.rename(target)
                except BaseException:
                    if os.path.lexists(previous):
                        previous.rename(target)
                    raise
                finally:
                    if os.path.lexists(previous):
                        if previous.is_dir() and not previous.is_symlink():
                            remove_tree(previous)
                        else:
                            previous.unlink()
            remaining = seconds - (time.monotonic() - start)
            if remaining <= 0:
                result = ExecutionResult("", "Package staging exceeded the timeout.", None, 0, timed_out=True)
            else:
                result = self.exec(session, install_command(requested), math.ceil(remaining), event,
                                   caller=caller or "backend:packages")
            result.duration_ms = int((time.monotonic() - start) * 1000)
            return result
        except PackageCancelled:
            return ExecutionResult("", "", None, int((time.monotonic() - start) * 1000), cancelled=True)
        finally:
            with self._lock:
                self._active_cancel.pop(session.id, None)
            lock.release()

    def import_files(self, session: SandboxSession, approved_paths: list[Path]) -> list[str]:
        with self._session_lock(session):
            if session.state != SessionState.READY:
                raise RuntimeError("Cannot import into a non-ready sandbox")
            paths = self.workspaces.import_files(session.id, approved_paths)
            if self.config.git_enabled and paths:
                self._initialize_git(session, imported_names=[p.removeprefix("/workspace/") for p in paths])
            return paths

    def _initialize_git(self, session: SandboxSession, cancel_event: threading.Event | None = None,
                        imported_names: list[str] | None = None) -> None:
        result = self.exec(session, init_command(self.config, imported_names), self.config.max_timeout,
                           cancel_event, caller="backend:git-baseline")
        if result.exit_code != 0 or result.error:
            raise RuntimeError("Could not initialize sandbox Git baseline; check Docker/image")

    def reset(self, session: SandboxSession) -> None:
        with self._session_lock(session):
            if session.state != SessionState.READY:
                raise RuntimeError("Cannot reset a non-ready sandbox")
            self.workspaces.reset(session.id)
            if self.config.git_enabled:
                self._initialize_git(session)

    def get_diff(self, session: SandboxSession) -> ExecutionResult:
        # Never stage arbitrary new data and never execute Git on the host.
        return self.exec(session, "git --no-pager diff --no-ext-diff --no-textconv HEAD --", caller="backend:diff")

    def destroy(self, session: SandboxSession) -> None:
        self.cancel(session)
        with self._session_lock(session):
            if session.state == SessionState.BROKEN:
                raise RuntimeError("Reconcile containers before deleting a quarantined workspace")
            self.workspaces.destroy(session.id)
            session.state = SessionState.DESTROYED
            with self._lock:
                self._contexts = {k: v for k, v in self._contexts.items() if v != session.id}
                self._write_json(self.base_dir / "contexts.json", self._contexts)

    def reconcile(self) -> list[str]:
        """Explicit startup operation, before accepting work; never removes workspaces."""
        with self._lock:
            if self._active_cancel:
                raise RuntimeError("Cannot reconcile during sandbox execution")
            removed = self.executor.reconcile()
            # Executor must confirm all matching containers are removed.
            for path in self.base_dir.glob("*/session.json"):
                session = self.get_session(path.parent.name)
                if session and session.state == SessionState.BROKEN:
                    session.state = SessionState.READY
                    self._save(session)
            return removed

    def _audit(self, session: SandboxSession, command: str, result: ExecutionResult, caller: str | None) -> None:
        if not self.config.audit_enabled:
            return
        try:
            with sqlite3.connect(self.base_dir / "audit.sqlite3", timeout=1) as db:
                db.execute("CREATE TABLE IF NOT EXISTS executions (session_id TEXT, timestamp REAL, command TEXT, "
                           "duration_ms INTEGER, exit_code INTEGER, timed_out INTEGER, stdout_truncated INTEGER, "
                           "stderr_truncated INTEGER, caller TEXT, cancelled INTEGER, error TEXT)")
                db.execute("INSERT INTO executions VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                    session.id, time.time(), command, result.duration_ms, result.exit_code,
                    result.timed_out, result.stdout_truncated, result.stderr_truncated,
                    caller, result.cancelled, result.error))
        except Exception:
            log.exception("Sandbox audit write failed")

    def close(self) -> None:
        """Release exclusive storage ownership after all executions have finished."""
        with self._lock:
            if self._active_cancel:
                raise RuntimeError("Cancel/wait for sandbox executions before closing")
            self._closed = True
            self._owner_file.close()
