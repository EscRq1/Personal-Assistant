"""Backend-only snapshots and result publication; no host execution of worker files."""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from agent_protocol import workspace_path

from sandbox.session import SessionState
from sandbox.workspace import SKIP_DIRS, _copy_node, _open_directory


def copy_snapshot(source: Path, target: Path, config, cancel):
    """Copy to an empty trusted directory, retaining the sandbox's no-link policy."""
    fd = _open_directory(source)
    budget = [0, 0]
    try:
        for name in sorted(os.listdir(fd)):
            if cancel.is_set():
                raise RuntimeError("Snapshot cancelled")
            if name in SKIP_DIRS or name.startswith("agent-results-"):
                continue
            _copy_node(fd, name, target / name, budget, config)
    finally:
        os.close(fd)


def manifest(trusted_copy: Path):
    """Only inspect a completed, link-free snapshot outside the container mount."""
    result = {}
    for path in sorted(trusted_copy.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(65536), b""):
                    digest.update(chunk)
            result[path.relative_to(trusted_copy).as_posix()] = {
                "sha256": digest.hexdigest(), "executable": bool(path.stat().st_mode & 0o111)}
    return result


def checkpoint(path, data):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class AgentWorkspaces:
    def __init__(self, manager, context_id, batch_id, cancel):
        self.manager, self.cancel = manager, cancel
        self.parent = manager.get_or_create_for_context(context_id, cancel)
        self.batch_id = batch_id
        self.children = []
        # Parent of workspace is never mounted in a model's container.
        self.root = manager.workspaces.path(self.parent.id) / "delegations" / batch_id
        self.root.mkdir(parents=True, mode=0o700)
        self.original = self.root / "original"
        self.original.mkdir(mode=0o700)
        with manager._session_lock(self.parent):
            if self.parent.state != SessionState.READY:
                raise RuntimeError("Parent sandbox is not ready")
            manager.workspaces.validate(self.parent.id, self.parent.workspace)
            copy_snapshot(self.parent.workspace, self.original, manager.config, cancel)
        self.before = manifest(self.original)

    def fork(self, name, resume_from=None):
        manager = self.manager
        source = self.original
        prior = None
        if resume_from:
            # References are validated by Assignment; resolve only under THIS chat.
            import re
            if not re.fullmatch("[0-9a-f]{32}", resume_from.get("batch_id", "")) or not re.fullmatch(
                    "agent[1-9][0-9]*", resume_from.get("agent_id", "")):
                raise ValueError("Invalid resume reference")
            previous = self.root.parent / resume_from["batch_id"] / resume_from["agent_id"]
            source = previous / "workspace"
            fd = _open_directory(source)
            os.close(fd)
            prior = json.loads((previous / "result.json").read_text())
            if prior.get("artifact_error"):
                raise ValueError("Cannot resume an incomplete artifact snapshot")
        session = manager.create_session(cancel_event=self.cancel)
        self.children.append(session)
        try:
            with manager._session_lock(session):
                copy_snapshot(source, session.workspace, manager.config, self.cancel)
                baseline = manager.workspaces.path(session.id) / "baseline"
                copy_snapshot(source, baseline, manager.config, self.cancel)
                if manager.config.git_enabled:
                    manager._initialize_git(session, self.cancel)
            journal = self.root / name
            journal.mkdir(mode=0o700)
            if prior is not None:
                checkpoint(journal / "continuation.json", {
                    "resume_from": resume_from, "prior_status": prior.get("status"),
                    "prior_result": prior.get("result", "")[:16000],
                    "blockers": prior.get("blockers", []), "next_step": prior.get("next_step", "")})
            return session, journal
        except BaseException:
            # Batch cleanup owns every child, including a failed fork.
            raise

    def collect(self, name, session, result, messages):
        folder = self.root / name
        checkpoint(folder / "transcript.json", messages)
        try:
            with self.manager._session_lock(session):
                if session.state != SessionState.READY:
                    raise RuntimeError("Worker sandbox is quarantined")
                output = folder / "workspace"
                output.mkdir(mode=0o700)
                # Tool execution has drained. Preserve progress even after /stop.
                copy_snapshot(session.workspace, output, self.manager.config, threading.Event())
            after = manifest(output)
            changes = {
                "added": sorted(after.keys() - self.before.keys()),
                "deleted": sorted(self.before.keys() - after.keys()),
                "modified": sorted(k for k in after.keys() & self.before.keys()
                                   if after[k] != self.before[k]),
            }
            checkpoint(folder / "changes.json", changes)
            result["changes"] = {key: len(value) for key, value in changes.items()}
            reported = result.get("reported_artifacts", [])
            missing = [path for path in reported if workspace_path(path) not in after]
            artifacts = sorted(set(changes["added"] + changes["modified"])
                               | {p for p in reported if p in after})
            result["artifacts"] = artifacts[:128]
            result["artifacts_total"] = len(artifacts)
            result["artifacts_truncated"] = len(artifacts) > 128
            if missing:
                result["blockers"].append(("Reported artifacts missing from final snapshot: " + ", ".join(missing))[:4000])
                result["stop_reason"] = "missing_artifacts"
                if result["status"] == "completed":
                    result["status"] = "partial"
            result["snapshot_complete"] = True
        except Exception as exc:
            result["artifact_error"] = str(exc)
            result["snapshot_complete"] = False
            if result.get("status") == "completed":
                result["status"] = "partial"
        checkpoint(folder / "result.json", result)
        return result

    def publish(self):
        """Publish a complete batch directory without overwriting original files."""
        import shutil
        import tempfile
        manager = self.manager
        name = "agent-results-" + self.batch_id
        with manager._session_lock(self.parent):
            if self.parent.state != SessionState.READY:
                raise RuntimeError("Parent sandbox is not ready")
            manager.workspaces.validate(self.parent.id, self.parent.workspace)
            target = self.parent.workspace / name
            if os.path.lexists(target):
                raise RuntimeError("Result directory already exists")
            with tempfile.TemporaryDirectory(dir=manager.workspaces.path(self.parent.id)) as tmp:
                stage = Path(tmp) / name
                stage.mkdir(mode=0o700)
                for folder in sorted(self.root.glob("agent*")):
                    # These directories are backend-owned, never container-mounted.
                    shutil.copytree(folder, stage / folder.name)
                stage.rename(target)
        return "/workspace/" + name

    def cleanup(self):
        errors = []
        for session in self.children:
            try:
                self.manager.destroy(session)
            except Exception as exc:
                errors.append({"session_id": session.id, "error": str(exc)})
        return errors
