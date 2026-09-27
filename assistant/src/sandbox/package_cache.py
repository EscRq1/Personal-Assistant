"""Operator-published wheels. No networking and no execution of package code here.

Publish is a trusted backend/CLI operation, never a model-facing tool. Published
caches are immutable snapshots; use a new directory to publish another version.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from email.parser import BytesParser
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import threading
from typing import BinaryIO, Iterator
import zipfile

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

from .workspace import _open_directory

MAX_REQUESTS = 20
MAX_WHEELS = 256
MAX_WHEEL_BYTES = 128 * 1024 * 1024
MAX_CACHE_BYTES = 512 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_METADATA_BYTES = 1024 * 1024
MAX_UNPACKED_BYTES = 2 * 1024 * 1024 * 1024
WHEEL_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.+!-]*\.whl\Z")
log = logging.getLogger(__name__)


class PackageError(ValueError):
    """Safe, model-facing package failure (never include host paths)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class PackageCancelled(Exception):
    pass


def check_cancel(event: threading.Event | None) -> None:
    if event is not None and event.is_set():
        raise PackageCancelled()


def validate_requests(packages: object) -> list[str]:
    if not isinstance(packages, list) or not 1 <= len(packages) <= MAX_REQUESTS:
        raise PackageError("invalid_arguments", "packages must contain 1 to 20 package requirements.")
    result: list[str] = []
    for value in packages:
        if not isinstance(value, str) or not 1 <= len(value) <= 200 or not value.isascii():
            raise PackageError("invalid_arguments", "Each package must be a short ASCII requirement.")
        try:
            requirement = Requirement(value)
        except InvalidRequirement:
            raise PackageError("invalid_arguments", "Use package names with optional extras and version constraints.") from None
        if requirement.url or requirement.marker:
            raise PackageError("invalid_arguments", "URLs, paths, indexes and environment markers are not accepted.")
        result.append(str(requirement))
    return result


@contextmanager
def regular_file(directory_fd: int, name: str, limit: int) -> Iterator[BinaryIO]:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > limit:
            raise PackageError("invalid_cache", "Cache entries must be bounded regular files without links.")
        yield stream


def _wheel_metadata(path: Path) -> tuple[str, str, int]:
    """Inspect bounded metadata; never extract or import a wheel on the host."""
    name, version, _, _ = parse_wheel_filename(path.name)
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > 50_000 or sum(x.file_size for x in members) > MAX_UNPACKED_BYTES:
            raise PackageError("invalid_cache", "Wheel unpacked size or file count exceeds the limit.")
        metadata = []
        seen: set[str] = set()
        for member in members:
            p = PurePosixPath(member.filename)
            mode = stat.S_IFMT(member.external_attr >> 16)
            if (p.is_absolute() or ".." in p.parts or "\\" in member.filename
                    or ":" in member.filename or "\0" in member.filename
                    or mode not in (0, stat.S_IFREG, stat.S_IFDIR)
                    or member.filename in seen):
                raise PackageError("invalid_cache", "Wheel contains unsafe or duplicate archive entries.")
            seen.add(member.filename)
            if len(p.parts) == 2 and p.parts[0].endswith(".dist-info") and p.name == "METADATA":
                metadata.append(member)
        if len(metadata) != 1 or metadata[0].file_size > MAX_METADATA_BYTES:
            raise PackageError("invalid_cache", "Wheel metadata is missing, ambiguous or oversized.")
        with archive.open(metadata[0]) as stream:
            raw = stream.read(MAX_METADATA_BYTES + 1)
        if len(raw) > MAX_METADATA_BYTES:
            raise PackageError("invalid_cache", "Wheel metadata is oversized.")
        message = BytesParser().parsebytes(raw)
        if canonicalize_name(message.get("Name", "")) != name or Version(message.get("Version", "")) != version:
            raise PackageError("invalid_cache", "Wheel filename and metadata disagree.")
        for value in message.get_all("Requires-Dist", []):
            if Requirement(value).url:
                raise PackageError("invalid_cache", "Wheel dependencies cannot use direct URLs or local paths.")
        return str(name), str(version), sum(x.file_size for x in members)


def publish(wheel_directory: Path, cache: Path, image: str) -> int:
    """Approve a LOCAL wheelhouse into a new immutable cache. Operator-only API."""
    source = Path(os.path.abspath(wheel_directory.expanduser()))
    cache = Path(os.path.abspath(cache.expanduser()))
    if not image or len(image) > 512 or any(c.isspace() for c in image):
        raise ValueError("A valid sandbox image reference is required")
    if os.path.lexists(cache):
        raise ValueError("Cache already exists; publish a new directory, then change SANDBOX_WHEEL_CACHE")
    cache.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    source_fd = _open_directory(source)
    try:
        names = sorted(os.listdir(source_fd))
        if not names or len(names) > MAX_WHEELS or any(not WHEEL_NAME.fullmatch(n) for n in names):
            raise PackageError("invalid_cache", "Wheelhouse must contain 1 to 256 .whl files only.")
        with tempfile.TemporaryDirectory(prefix="wheel-publish-", dir=cache.parent) as temporary:
            stage = Path(temporary) / "cache"
            stage.mkdir(mode=0o700)
            artifacts, versions = [], {}
            total = unpacked = 0
            for filename in names:
                destination = stage / filename
                digest, size = hashlib.sha256(), 0
                with regular_file(source_fd, filename, MAX_WHEEL_BYTES) as src, destination.open("xb") as dst:
                    while chunk := src.read(256 * 1024):
                        size += len(chunk)
                        total += len(chunk)
                        if size > MAX_WHEEL_BYTES or total > MAX_CACHE_BYTES:
                            raise PackageError("invalid_cache", "Wheelhouse exceeds its byte limit.")
                        digest.update(chunk)
                        dst.write(chunk)
                name, version, expanded = _wheel_metadata(destination)
                unpacked += expanded
                if unpacked > MAX_UNPACKED_BYTES:
                    raise PackageError("invalid_cache", "Combined unpacked wheel size exceeds the limit.")
                if name in versions and versions[name] != version:
                    raise PackageError("invalid_cache", "Publish one version per package; use separate caches for other versions.")
                versions[name] = version
                destination.chmod(0o400)
                artifacts.append(dict(filename=filename, name=name, version=version,
                                      size=size, sha256=digest.hexdigest()))
            (stage / "catalog.json").write_text(json.dumps({"format": 1, "image": image, "wheels": artifacts}))
            (stage / "catalog.json").chmod(0o400)
            # rename cannot replace a nonempty published cache. Concurrent
            # publishers to the same destination fail instead of merging.
            stage.rename(cache)
            return len(artifacts)
    finally:
        os.close(source_fd)

def stage_manual_wheelhouse(
    source: Path,
    destination: Path,
    packages: list[str],
    event: threading.Event,
) -> int:
    """Validate and copy a user-approved wheelhouse into backend staging.

    This never imports packages, executes package code, invokes pip, or
    accesses the network.
    """

    requested = validate_requests(packages)
    source = Path(os.path.abspath(source.expanduser()))

    check_cancel(event)

    try:
        fd = _open_directory(source)
    except OSError:
        raise PackageError(
            "invalid_manual_wheels",
            "Manual wheel upload is unavailable or invalid.",
        ) from None

    try:
        names = sorted(os.listdir(fd))

        if (
            not names
            or len(names) > MAX_WHEELS
            or any(
                not isinstance(name, str)
                or not WHEEL_NAME.fullmatch(name)
                for name in names
            )
        ):
            raise PackageError(
                "invalid_manual_wheels",
                "Manual package upload must contain 1 to 256 wheel files only.",
            )

        versions: dict[str, Version] = {}

        total = 0
        unpacked = 0

        for filename in names:
            check_cancel(event)

            target = destination / filename
            size = 0

            with (
                regular_file(
                    fd,
                    filename,
                    MAX_WHEEL_BYTES,
                ) as src,
                target.open("xb") as dst,
            ):
                while chunk := src.read(256 * 1024):
                    check_cancel(event)

                    size += len(chunk)
                    total += len(chunk)

                    if (
                        size > MAX_WHEEL_BYTES
                        or total > MAX_CACHE_BYTES
                    ):
                        raise PackageError(
                            "invalid_manual_wheels",
                            "Manual wheel upload exceeds package size limits.",
                        )

                    dst.write(chunk)

            name, version_text, expanded = _wheel_metadata(target)

            canonical_name = canonicalize_name(name)
            version = Version(version_text)

            unpacked += expanded

            if unpacked > MAX_UNPACKED_BYTES:
                raise PackageError(
                    "invalid_manual_wheels",
                    "Combined unpacked wheel size exceeds the limit.",
                )

            if canonical_name in versions:
                raise PackageError(
                    "invalid_manual_wheels",
                    "Manual upload contains multiple wheels for the same package.",
                )

            versions[canonical_name] = version

        for text in requested:
            requirement = Requirement(text)

            version = versions.get(
                canonicalize_name(requirement.name)
            )

            if (
                version is None
                or not requirement.specifier.contains(
                    version,
                    prereleases=True,
                )
            ):
                raise PackageError(
                    "manual_package_mismatch",
                    (
                        "Uploaded wheels do not contain the requested "
                        "package/version."
                    ),
                )

        return len(names)

    except PackageError:
        raise

    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        zipfile.BadZipFile,
    ):
        log.warning(
            "Manual wheel validation failed",
            exc_info=True,
        )

        raise PackageError(
            "invalid_manual_wheels",
            "Uploaded wheel files failed validation.",
        ) from None

    finally:
        os.close(fd)
        
@dataclass(frozen=True)
class WheelCache:
    path: Path
    image: str

    def stage(self, destination: Path, packages: list[str], event: threading.Event) -> None:
        """Copy verified bytes into a NEW backend-owned directory, never a workspace path."""
        check_cancel(event)
        try:
            fd = _open_directory(Path(os.path.abspath(self.path.expanduser())))
        except OSError:
            raise PackageError("package_cache_unavailable", "The backend has no readable approved wheel cache.") from None
        try:
            info = os.fstat(fd)
            if info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise PackageError("invalid_cache", "The approved cache must be private to the backend user.")
            with regular_file(fd, "catalog.json", MAX_MANIFEST_BYTES) as stream:
                raw = stream.read(MAX_MANIFEST_BYTES + 1)
            if len(raw) > MAX_MANIFEST_BYTES:
                raise PackageError("invalid_cache", "Cache catalog is oversized.")
            catalog = json.loads(raw)
            if catalog.get("format") != 1 or catalog.get("image") != self.image:
                raise PackageError("invalid_cache", "Cache image does not match the configured sandbox image.")
            wheels = catalog.get("wheels")
            if not isinstance(wheels, list) or not 1 <= len(wheels) <= MAX_WHEELS:
                raise PackageError("invalid_cache", "Invalid cache artifact count.")
            approved: dict[str, Version] = {}
            names: set[str] = set()
            for wheel in wheels:
                filename = wheel["filename"]
                if not isinstance(filename, str) or not WHEEL_NAME.fullmatch(filename) or filename in names:
                    raise PackageError("invalid_cache", "Invalid cache artifact name.")
                name, version, _, _ = parse_wheel_filename(filename)
                if wheel["name"] != name or wheel["version"] != str(version):
                    raise PackageError("invalid_cache", "Cache artifact metadata disagrees.")
                if (not re.fullmatch(r"[0-9a-f]{64}", wheel["sha256"])
                        or type(wheel["size"]) is not int or not 0 < wheel["size"] <= MAX_WHEEL_BYTES):
                    raise PackageError("invalid_cache", "Invalid cache artifact size or hash.")
                if name in approved and approved[name] != version:
                    raise PackageError("invalid_cache", "Cache contains multiple package versions.")
                approved[str(name)] = version
                names.add(filename)
            for text in packages:
                requirement = Requirement(text)
                version = approved.get(canonicalize_name(requirement.name))
                if version is None or not requirement.specifier.contains(version, prereleases=True):
                    raise PackageError("package_not_approved", "Requested package/version is not in the approved cache; ask the operator to publish it.")
            total = 0
            for wheel in wheels:
                check_cancel(event)
                digest, size = hashlib.sha256(), 0
                with regular_file(fd, wheel["filename"], MAX_WHEEL_BYTES) as src, (destination / wheel["filename"]).open("xb") as dst:
                    while chunk := src.read(256 * 1024):
                        check_cancel(event)
                        size += len(chunk)
                        total += len(chunk)
                        if size > wheel["size"] or total > MAX_CACHE_BYTES:
                            raise PackageError("invalid_cache", "Cache byte limit exceeded.")
                        digest.update(chunk)
                        dst.write(chunk)
                if size != wheel["size"] or digest.hexdigest() != wheel["sha256"]:
                    raise PackageError("invalid_cache", "Approved wheel integrity check failed.")
        except PackageError:
            raise
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            log.warning("Approved wheel cache validation failed", exc_info=True)
            raise PackageError("invalid_cache", "Approved wheel cache is invalid; consult the operator.") from None
        finally:
            os.close(fd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", required=True, type=Path, help="Trusted local wheelhouse; wheels only")
    parser.add_argument("--cache", required=True, type=Path, help="NEW approved cache directory")
    parser.add_argument("--image", required=True, help="Exact image reference used by SandboxConfig")
    args = parser.parse_args()
    count = publish(args.wheel_dir, args.cache, args.image)
    print(f"Published {count} approved wheel artifacts. No package code was executed.")


if __name__ == "__main__":
    main()
