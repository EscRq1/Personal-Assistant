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


def live_preview(manager, session, relative_path, cancel):
    """Read a trusted frozen copy at an idle tool boundary; never block a live tool."""
    import tempfile
    path = workspace_path(relative_path)
    lock = manager._session_lock(session)
    if not lock.acquire(blocking=False):
        return {"busy": True, "note": "Sandbox tool is active; use progress or interrupt_worker for a final checkpoint."}
    try:
        if session.state != SessionState.READY:
            return {"busy": True, "note": "Sandbox is not ready for a safe file view."}
        manager.workspaces.validate(session.id, session.workspace)
        with tempfile.TemporaryDirectory(prefix="preview-", dir=manager.workspaces.path(session.id)) as folder:
            copy_snapshot(session.workspace, Path(folder), manager.config, cancel)
            with (Path(folder) / path).open("rb") as stream:
                data = stream.read(12001)
            return {"busy": False, "path": path, "content": data[:12000].decode("utf-8", "replace"),
                    "truncated": len(data) > 12000, "view": "safe snapshot at an idle tool boundary"}
    finally:
        lock.release()


def merge_snapshot(manager, session, base, source, cancel):
    """Three-way file merge in a trusted staging tree; publish only without conflicts.

    base is the worker's dispatch-time manifest. source is its frozen, safe
    result snapshot. Neither path nor manifest comes from tool arguments.
    Coordinator commands and swaps share the sandbox's execution lock.
    """
    import shutil
    import tempfile
    from sandbox.workspace import remove_tree

    desired = manifest(source)
    changes = sorted(p for p in base.keys() | desired.keys() if base.get(p) != desired.get(p))
    with manager._session_lock(session):
        if session.state != SessionState.READY:
            raise RuntimeError("Integration sandbox is not ready")
        manager.workspaces.validate(session.id, session.workspace)
        parent = manager.workspaces.path(session.id)
        with tempfile.TemporaryDirectory(prefix="integrate-", dir=parent) as temporary:
            stage, backup = Path(temporary) / "stage", Path(temporary) / "old"
            stage.mkdir(mode=0o700)
            copy_snapshot(session.workspace, stage, manager.config, cancel)
            current = manifest(stage)
            conflicts = {p for p in changes if current.get(p) not in (base.get(p), desired.get(p))}
            removed = {p for p in changes if p not in desired}
            for p in changes:
                if p in desired:
                    conflicts.update(q for q in current if q.startswith(p + "/") and q not in removed)
                    conflicts.update(q for q in current if p.startswith(q + "/") and q not in removed)
            if conflicts:
                return {"integrated": False, "conflicts": sorted(conflicts), "changed_files": changes}
            for p in sorted(removed, key=lambda value: value.count("/"), reverse=True):
                target = stage / p
                if target.is_file():
                    target.unlink()
            for p in changes:
                if p not in desired:
                    continue
                target = stage / workspace_path(p)
                if target.is_dir():
                    # Only directories made empty by declared file removals can
                    # be replaced. No unrelated descendant may be discarded.
                    for directory in sorted(target.rglob("*"), reverse=True):
                        if directory.is_dir():
                            directory.rmdir()
                    target.rmdir()
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                shutil.copy2(source / p, target)
            # Apply the same total size/file budget to the combined candidate.
            with tempfile.TemporaryDirectory(prefix="validate-", dir=parent) as validation:
                copy_snapshot(stage, Path(validation), manager.config, cancel)
            if cancel.is_set():
                raise InterruptedError("Integration cancelled")
            moved = []
            session.workspace.rename(backup)
            try:
                stage.rename(session.workspace)
                # Preserve runtime environments and git metadata by renaming
                # directory entries; never traverse their model-created links.
                for entry in backup.iterdir():
                    if entry.name in SKIP_DIRS or entry.name.startswith("agent-results-"):
                        entry.rename(session.workspace / entry.name)
                        moved.append(entry.name)
            except BaseException:
                for name in reversed(moved):
                    (session.workspace / name).rename(backup / name)
                if session.workspace.exists():
                    session.workspace.rename(stage)
                backup.rename(session.workspace)
                raise
            remove_tree(backup)
    return {"integrated": True, "conflicts": [], "changed_files": changes}


class AgentWorkspaces:
    def __init__(self, manager, context_id, batch_id, cancel):
        self.manager, self.cancel = manager, cancel
        self.parent = manager.get_or_create_for_context(context_id, cancel)
        self.batch_id = batch_id
        self.children = []
        self.baselines = {}
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

    def fork(self, name, resume_from=None, *, source=None):
        manager = self.manager
        custom_source = source is not None
        source = source if custom_source else self.original
        prior = None
        if resume_from:
            if custom_source:
                raise ValueError("Choose a snapshot source or resume_from, not both")
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
            if custom_source:
                self.baselines[name] = manifest(source)
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
            before = self.baselines.get(name, self.before)
            changes = {
                "added": sorted(after.keys() - before.keys()),
                "deleted": sorted(before.keys() - after.keys()),
                "modified": sorted(k for k in after.keys() & before.keys()
                                   if after[k] != before[k]),
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
                folders = list(self.root.glob("agent*"))
                if (self.root / "orchestrator").is_dir():
                    folders.append(self.root / "orchestrator")
                for folder in sorted(folders):
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