"""Multipart attachment -> backend staging copy -> current sandbox.

Never accept host paths, extract archives, or execute uploads on the host.
"""
from __future__ import annotations
import asyncio
import json
import logging
import uuid
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from starlette.datastructures import FormData, UploadFile
from starlette.exceptions import HTTPException
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import Request, ClientDisconnect
from starlette.types import ASGIApp, Message, Receive, Scope, Send

log = logging.getLogger(__name__)
MAX_FILES = 20
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_BODY_BYTES = MAX_TOTAL_BYTES + 1024 * 1024
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_IMAGES_BYTES = 32 * 1024 * 1024
CHUNK_BYTES = 256 * 1024
VISION_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp"}
RESERVED_NAMES = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", "node_modules"}


class UploadError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class AskBodyLimitMiddleware:
    """Count actual /ask request bytes, including requests without Content-Length.

    MultiPartException makes Starlette's multipart parser close partially-created
    spool files. The send wrapper changes its normal 400 response to 413.
    """
    def __init__(self, app: ASGIApp, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] != 'http' or scope.get('path') != '/ask':
            await self.app(scope, receive, send)
            return
        from starlette.responses import JSONResponse
        headers = dict(scope.get('headers', []))
        try:
            length = int(headers.get(b'content-length', b'0'))
        except ValueError:
            await JSONResponse({'error': 'Invalid Content-Length'}, status_code=400)(scope, receive, send)
            return
        if length > self.max_bytes:
            await JSONResponse({'error': 'Attachment request exceeds the 128 MiB total limit.'}, status_code=413)(scope, receive, send)
            return
        received, exceeded = 0, False
        multipart = headers.get(b'content-type', b'').lower().startswith(b'multipart/form-data')

        async def bounded_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message['type'] == 'http.request':
                received += len(message.get('body', b''))
                if received > self.max_bytes:
                    exceeded = True
                    if multipart:
                        raise MultiPartException('Attachment request exceeds the 128 MiB total limit.')
                    raise HTTPException(413, 'Request body is too large.')
            return message

        async def bounded_send(message: Message) -> None:
            if exceeded and message['type'] == 'http.response.start':
                message = {**message, 'status': 413}
            await send(message)

        await self.app(scope, bounded_receive, bounded_send)


async def parse_upload_form(request: Request, cancel_event: threading.Event) -> FormData:
    """Ensure even older Starlette parsers close spools on stop/disconnect."""
    interrupted: BaseException | None = None

    async def guarded_stream():
        nonlocal interrupted
        try:
            async for chunk in request.stream():
                if cancel_event.is_set():
                    raise MultiPartException("Upload cancelled.")
                yield chunk
        except (asyncio.CancelledError, ClientDisconnect) as exc:
            interrupted = exc
            cancel_event.set()
            # Parser versions that catch only MultiPartException still close
            # their partial files; re-raise the original interruption below.
            raise MultiPartException("Upload interrupted.") from exc

    parser = MultiPartParser(request.headers, guarded_stream(), max_files=MAX_FILES,
                             max_fields=16, max_part_size=64 * 1024)
    try:
        return await parser.parse()
    except MultiPartException as exc:
        if interrupted is not None:
            raise interrupted
        raise UploadError(str(exc), 409 if cancel_event.is_set() else 400) from exc


def filename_for_upload(name: str | None) -> str:
    # Filename is a label, never a client-selected host path.
    if not isinstance(name, str) or not name or len(name.encode('utf-8')) > 200:
        raise UploadError('An attachment has an empty or excessively long filename.')
    if name in {'.', '..'} or any(ch in name for ch in ('/', '\\', ':')) or any(ord(ch) < 32 or ord(ch) == 127 for ch in name):
        raise UploadError('Attachment filenames must be plain names, without paths or control characters.')
    if name in RESERVED_NAMES:
        raise UploadError('This filename is reserved for sandbox metadata or caches.')
    return name


def collect_uploads(form: FormData) -> list[UploadFile]:
    uploads: list[UploadFile] = []
    total = 0
    image_total = 0
    for field, value in form.multi_items():
        if isinstance(value, UploadFile):
            if field not in {'file', 'image'}:
                raise UploadError('Unexpected file field; use file or image.')
            filename_for_upload(value.filename)
            size = value.size
            if size is not None:
                if size > MAX_FILE_BYTES:
                    raise UploadError('Each attachment must be at most 32 MiB.', 413)
                total += size
                if field == 'image' or value.content_type in VISION_TYPES:
                    if size > MAX_IMAGE_BYTES:
                        raise UploadError('Each image must be at most 16 MiB.', 413)
                    image_total += size
            if field == 'image' and value.content_type not in VISION_TYPES:
                raise UploadError('The image field requires a supported image MIME type.')
            uploads.append(value)
        elif field in {'file', 'image'}:
            raise UploadError('Attachments must contain uploaded bytes, not paths or strings.')
    if len(uploads) > MAX_FILES or total > MAX_TOTAL_BYTES or image_total > MAX_IMAGES_BYTES:
        raise UploadError('Attachment limits exceeded: 20 files, 128 MiB total, 32 MiB of images.', 413)
    return uploads


def form_text(form: FormData, key: str, default: str = '') -> str:
    value = form.get(key, default)
    if not isinstance(value, str):
        raise UploadError(f'{key} must be text.')
    return value


@dataclass
class ImportedBatch:
    files: list[dict[str, Any]]
    images: list[bytes]

@dataclass
class VisionUpload:
    data: bytes
    mime: str

def _stage_and_import(uploads: list[UploadFile], context_id: str,
                      cancel_event: threading.Event,
                      manager_factory: Callable[[], Any]) -> ImportedBatch:
    records: list[dict[str, Any]] = []
    paths: list[Path] = []
    images: list[bytes] = []
    total = image_total = 0
    # All blocking disk IO, copying, and Docker/Git setup happen in this worker.
    # This directory is backend-created and never exposed to the model.
    with tempfile.TemporaryDirectory(prefix='assistant-upload-') as temporary:
        stage = Path(temporary)
        for upload in uploads:
            if cancel_event.is_set():
                raise UploadError('Upload cancelled.', 409)
            original_name = filename_for_upload(upload.filename)
            name = original_name
            sequence = 2
            while (stage / name).exists():
                source = Path(original_name)
                name = f'{source.stem}.upload-{sequence}{source.suffix}'
                sequence += 1
            target = stage / name
            size = 0
            vision = upload.content_type in VISION_TYPES
            image_data = bytearray()
            upload.file.seek(0)
            with target.open('xb') as output:
                while chunk := upload.file.read(CHUNK_BYTES):
                    if cancel_event.is_set():
                        raise UploadError('Upload cancelled.', 409)
                    size += len(chunk)
                    total += len(chunk)
                    if size > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
                        raise UploadError('Attachment byte limits exceeded.', 413)
                    if vision:
                        image_total += len(chunk)
                        if size > MAX_IMAGE_BYTES or image_total > MAX_IMAGES_BYTES:
                            raise UploadError('Image byte limits exceeded.', 413)
                        image_data.extend(chunk)
                    output.write(chunk)
            paths.append(target)
            records.append({'name': original_name, 'size': size})
            if vision:
                images.append(
                    VisionUpload(
                        data=bytes(image_data),
                        mime=upload.content_type,
                    )
                )
        if cancel_event.is_set():
            raise UploadError('Upload cancelled.', 409)
        manager = manager_factory()
        session = manager.get_or_create_for_context(context_id)
        if cancel_event.is_set():
            raise UploadError('Upload cancelled.', 409)
        try:
            imported = manager.import_files(session, paths)
        except ValueError as exc:
            log.warning('Sandbox rejected attachment batch: %s', exc)
            raise UploadError('Sandbox import limits were exceeded or a filename was rejected.', 413) from exc
        if len(imported) != len(records):
            raise RuntimeError('Sandbox importer returned an incomplete attachment manifest')
        for record, path in zip(records, imported):
            if not isinstance(path, str) or not path.startswith('/workspace/'):
                raise RuntimeError('Sandbox importer returned an invalid container path')
            record['path'] = path
        return ImportedBatch(records, images)


async def import_uploads(uploads: list[UploadFile], context_id: str,
                         cancel_event: threading.Event,
                         manager_factory: Callable[[], Any]) -> ImportedBatch:
    if not uploads:
        return ImportedBatch([], [])
    worker = asyncio.create_task(asyncio.to_thread(_stage_and_import, uploads, context_id, cancel_event, manager_factory))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        # Unlike inference-result waiting, request cleanup must not close spool
        # files or release the chat admission slot while an import is still using
        # them. The worker owns and cleans its staging directory.
        cancel_event.set()
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not worker.cancelled():
            worker.exception()
        raise


def attachment_query(query: str, files: list[dict[str, Any]]) -> str:
    if not files:
        return query
    # Stored in the user turn, never promoted to a system instruction. Names and
    # file contents are untrusted data; the model can inspect paths using tools.
    return query.rstrip() + '\n\n[SANDBOX_FILES: ' + json.dumps(files, ensure_ascii=False) + ']'


def validate_draft_id(value: str | None) -> str | None:
    if not value:
        return None
    try:
        if not isinstance(value, str):
            raise ValueError
        return str(uuid.UUID(value))
    except (ValueError, AttributeError):
        raise UploadError('Invalid draft identity.') from None