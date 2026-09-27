"""Host-side copies. Never run host programs from an untrusted workspace."""
from __future__ import annotations
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path
from .config import SandboxConfig

SKIP_DIRS = {".git", ".venv", ".packages", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", "node_modules"}


def _copy_node(src_fd: int, name: str, destination: Path, budget: list[int],
               config: SandboxConfig, depth: int = 0) -> None:
    """Open every source component relative to a pinned directory; never follow links."""
    if depth > 128:
        raise ValueError("Import directory nesting exceeds 128 levels")
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=src_fd)
    try:
        info = os.fstat(fd)
        if stat.S_ISDIR(info.st_mode):
            if name in SKIP_DIRS:
                return
            budget[1] += 1
            if budget[1] > config.max_import_files:
                raise ValueError("Import file count limit exceeded")
            destination.mkdir(mode=0o700)
            for child in sorted(os.listdir(fd)):
                _copy_node(fd, child, destination / child, budget, config, depth + 1)
        elif stat.S_ISREG(info.st_mode):
            budget[1] += 1
            if budget[1] > config.max_import_files or budget[0] + info.st_size > config.max_import_bytes:
                raise ValueError("Import size/file count limit exceeded")
            with destination.open("xb") as out:
                while chunk := os.read(fd, 65_536):
                    budget[0] += len(chunk)
                    if budget[0] > config.max_import_bytes:
                        raise ValueError("Import size limit exceeded")
                    out.write(chunk)
            destination.chmod(0o700 if info.st_mode & 0o111 else 0o600)
        else:
            raise ValueError("Import accepts only regular files and directories")
    finally:
        os.close(fd)


def _open_directory(path: Path) -> int:
    """Reject symlinks in every component, including ancestors of a selected path."""
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def remove_tree(path: Path) -> None:
    """fd-safe rmtree; repair directory modes without following sandbox symlinks."""
    if path.is_symlink():
        path.unlink()
        return
    if not path.exists():
        return
    if not shutil.rmtree.avoids_symlink_attacks:
        raise RuntimeError("This platform lacks symlink-safe rmtree")
    # No execution is active while this runs. Directory ownership stays with the
    # backend UID (all capabilities dropped); only modes may need repairing.
    def repair(fd: int) -> None:
        os.fchmod(fd, 0o700)
        for name in os.listdir(fd):
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                os.chmod(name, 0o700, dir_fd=fd, follow_symlinks=False)
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                try:
                    repair(child)
                finally:
                    os.close(child)
    path.chmod(0o700, follow_symlinks=False)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        repair(fd)
    finally:
        os.close(fd)
    shutil.rmtree(path)


class WorkspaceManager:
    def __init__(self, base_dir: Path, config: SandboxConfig) -> None:
        self.config = config
        raw = Path(os.path.abspath(base_dir.expanduser()))
        # The base is trusted configuration, but never accept /, home, or a
        # workspace selected by the model. Each execution mounts a new child.
        if raw in {Path("/"), Path.home(), Path.cwd()}:
            raise ValueError("Use a dedicated sandbox storage directory")
        raw.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = _open_directory(raw)
        try:
            info = os.fstat(fd)
            if info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise ValueError("Sandbox storage must be owned by the backend user with mode 0700")
        finally:
            os.close(fd)
        self.base_dir = raw

    def path(self, session_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", session_id):
            raise ValueError("Invalid sandbox session ID")
        return self.base_dir / session_id

    def validate(self, session_id: str, workspace: Path) -> None:
        if workspace != self.path(session_id) / "workspace":
            raise ValueError("Workspace identity mismatch")
        fd = _open_directory(workspace)
        os.close(fd)

    def create_workspace(self, session_id: str) -> Path:
        parent = self.path(session_id)
        parent.mkdir(mode=0o700)
        (parent / "baseline").mkdir(mode=0o700)
        (parent / "workspace").mkdir(mode=0o700)
        return parent / "workspace"

    def import_files(self, session_id: str, approved_paths: list[Path]) -> list[str]:
        """Backend-only. Directories contribute their children; collisions get suffixes.

        Stage everything before publishing. Never merge through a model-created
        directory/symlink. Existing content, including previous edits, is retained.
        All symlinks are rejected, including links with an internal target.
        """
        parent = self.path(session_id)
        workspace, baseline = parent / "workspace", parent / "baseline"
        self.validate(session_id, workspace)
        budget = [0, 0]
        for root, dirs, files in os.walk(baseline):
            budget[1] += len(dirs) + len(files)
            budget[0] += sum((Path(root) / f).stat().st_size for f in files)
        imported: list[str] = []
        with tempfile.TemporaryDirectory(prefix="import-", dir=parent) as temp:
            stage = Path(temp)
            def copy_entry(fd: int, name: str) -> None:
                if name in SKIP_DIRS:
                    return
                chosen = name
                suffix = 1
                while any(os.path.lexists(p / chosen) for p in (workspace, baseline, stage)):
                    chosen = f"{name}.import-{suffix}"
                    suffix += 1
                _copy_node(fd, name, stage / chosen, budget, self.config)
                imported.append(chosen)
            for approved in approved_paths:
                source = Path(os.path.abspath(Path(approved).expanduser()))
                if source == Path("/") or source == self.base_dir or self.base_dir in source.parents or source in self.base_dir.parents:
                    raise ValueError("Cannot import sandbox storage or filesystem root")
                fd = _open_directory(source.parent)
                try:
                    info = os.stat(source.name, dir_fd=fd, follow_symlinks=False)
                    if stat.S_ISLNK(info.st_mode):
                        raise ValueError("Symlink imports are not allowed")
                    if stat.S_ISDIR(info.st_mode):
                        child_fd = os.open(source.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                        try:
                            for name in sorted(os.listdir(child_fd)):
                                copy_entry(child_fd, name)
                        finally:
                            os.close(child_fd)
                    else:
                        copy_entry(fd, source.name)
                finally:
                    os.close(fd)
            published: list[str] = []
            try:
                for name in imported:
                    src = stage / name
                    published.append(name)
                    if src.is_dir():
                        shutil.copytree(src, baseline / name)
                    else:
                        shutil.copyfile(src, baseline / name)
                        (baseline / name).chmod(src.stat().st_mode & 0o777)
                    src.rename(workspace / name)
            except BaseException:
                for name in published:
                    for directory in (workspace, baseline):
                        target = directory / name
                        if target.is_dir() and not target.is_symlink():
                            remove_tree(target)
                        elif os.path.lexists(target):
                            target.unlink()
                raise
        return ["/workspace/" + name for name in imported]

    def reset(self, session_id: str) -> Path:
        parent = self.path(session_id)
        workspace = parent / "workspace"
        # Parent is never mounted, so the model cannot replace it or baseline.
        fd = _open_directory(parent)
        os.close(fd)
        # Prepare the replacement first: disk/copy failure leaves the current
        # workspace intact. Neither directory is mounted while the lock is held.
        with tempfile.TemporaryDirectory(prefix="reset-", dir=parent) as temp:
            stage = Path(temp)
            replacement = stage / "workspace"
            old = stage / "old"
            shutil.copytree(parent / "baseline", replacement)
            replacement.chmod(0o700)
            moved_old = os.path.lexists(workspace)
            if moved_old:
                if not workspace.is_symlink():
                    workspace.chmod(0o700, follow_symlinks=False)
                workspace.rename(old)
            try:
                replacement.rename(workspace)
            except BaseException:
                if moved_old:
                    old.rename(workspace)
                raise
            if moved_old:
                remove_tree(old)
        return workspace

    def destroy(self, session_id: str) -> None:
        parent = self.path(session_id)
        fd = _open_directory(parent)
        os.close(fd)
        remove_tree(parent)
