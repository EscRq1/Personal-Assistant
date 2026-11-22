"""User-approved wheel downloads followed by existing isolated installation."""
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from sandbox.package_cache import PackageError, validate_requests, MAX_CACHE_BYTES
from sandbox.result import cap_text
from user_approvals import approvals

log = logging.getLogger(__name__)
SCRIPT = Path(__file__).resolve().parent.parent / 'scripts' / 'download_sandbox_wheels.sh'

REQUEST_PACKAGE_SCHEMA = {
    'type': 'function', 'function': {
        'name': 'request_package',
        'description': 'Install Python requirements into the current sandbox. First tries the approved offline wheel cache without asking. Only requirements unavailable in that cache require user approval to download wheels and dependencies from PyPI. Wait for the result, then use sandbox_exec. No manual commands are needed.',
        'parameters': {'type': 'object', 'properties': {
            'packages': {'type': 'array', 'items': {'type': 'string'},
                         'description': '1–20 Python requirements, e.g. pandas or requests>=2. URLs, paths, indexes and markers are forbidden.'},
            'reason': {'type': 'string', 'description': 'Why these packages are needed.'},
        }, 'required': ['packages', 'reason'], 'additionalProperties': False},
    },
}


def download_and_install(manager, session, packages, ctx):
    if ctx.cancel_event.is_set():
        return {'error': 'cancelled'}
    with tempfile.TemporaryDirectory(prefix='assistant-approved-wheels-') as temporary:
        root = Path(temporary)
        wheels = root / 'wheels'
        wheels.mkdir()
        # Isolated pip and a clean environment prevent user pip configuration,
        # extra indexes or injected Python paths from changing the approved source.
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(('PIP_', 'PYTHON'))}
        env['PIP_CONFIG_FILE'] = os.devnull
        with (root / 'download.log').open('w+b') as output:
            process = subprocess.Popen(
                ['bash', str(SCRIPT), sys.executable, str(wheels), *packages],
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                env=env, start_new_session=True,
            )
            deadline = time.monotonic() + 300
            failure = None
            try:
                while process.poll() is None:
                    if ctx.cancel_event.wait(0.1):
                        failure = 'cancelled'
                        break
                    if time.monotonic() >= deadline:
                        failure = 'download_timeout'
                        break
                    if sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) > MAX_CACHE_BYTES:
                        failure = 'download_size_limit'
                        break
            finally:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            if failure or ctx.cancel_event.is_set():
                return {'error': failure or 'cancelled', 'requested': packages}
            if process.returncode:
                log.warning('Approved wheel download failed (exit %s)', process.returncode)
                return {'error': 'download_failed', 'message': 'Could not download compatible wheels from PyPI.',
                        'requested': packages}
        if ctx.cancel_event.is_set():
            return {'error': 'cancelled'}
        # Existing code validates wheel contents, dependencies, hashes and limits;
        # pip installation executes only inside the session sandbox container.
        result = manager.install_manual(session, wheels, packages,
                                        cancel_event=ctx.cancel_event, caller='user:approved-package-request')
        return package_result(result, manager, session, packages, 'user_approved_download')


def package_result(result, manager, session, packages, source):
    payload = result.to_dict()
    for name in ('stdout', 'stderr'):
        payload[name], truncated = cap_text(payload.get(name, ''), manager.config.model_output_bytes // 2)
        payload[name + '_truncated'] = payload.get(name + '_truncated', False) or truncated
    return {**payload, 'requested': packages, 'session_id': session.id,
            'environment': '/workspace/.venv', 'source': source}


def request_package(args, ctx, *, manager=None, session=None):
    """Backend-only session override keeps delegated installs in their own workspace."""
    if ctx is None or not (ctx.sandbox_context_id or ctx.chat_id):
        return {'error': 'missing_context'}
    try:
        if not isinstance(args, dict) or set(args) != {'packages', 'reason'}:
            raise ValueError('Provide packages and reason only.')
        packages = validate_requests(args['packages'])
        reason = args['reason']
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
            raise ValueError('Provide a reason of 1–1000 characters.')
        if ctx.cancel_event.is_set():
            return {'status': 'cancelled'}
        if manager is None:
            from sandbox_tools import get_manager
            manager = get_manager()
        if session is None:
            session = manager.get_or_create_for_context(ctx.sandbox_context_id or ctx.chat_id, ctx.cancel_event)
        # Missing requirements or an unavailable cache warrant a network approval.
        # Invalid caches, timeouts and execution failures must not become downloads.
        try:
            cached = manager.install(session, packages, cancel_event=ctx.cancel_event,
                                     caller='assistant/agents:approved-cache')
        except PackageError as exc:
            if exc.code not in {'package_not_approved', 'package_cache_unavailable'}:
                raise
        else:
            return package_result(cached, manager, session, packages, 'approved_cache')
        if ctx.cancel_event.is_set():
            return {'status': 'cancelled', 'requested': packages}
        on_wait = getattr(ctx, 'approval_wait', None)
        try:
            if on_wait:
                on_wait(True)
            status = approvals.request(ctx, 'package', {
                'reason': reason, 'packages': packages, 'session_id': session.id,
                'scope': 'current_sandbox', 'source': 'PyPI',
                'actor': getattr(ctx, 'approval_actor', 'Assistant / coder'),
            })
        finally:
            if on_wait:
                on_wait(False)
        if status != 'approved' or ctx.cancel_event.is_set():
            return {'status': status if not ctx.cancel_event.is_set() else 'cancelled', 'requested': packages}
        # Keep the exact session and validated requirements that the user approved.
        return download_and_install(manager, session, packages, ctx)
    except (ValueError, PackageError) as exc:
        return {'error': getattr(exc, 'code', 'invalid_arguments'), 'message': str(exc)}
    except Exception:
        log.exception('Package request failed')
        return {'error': 'package_install_unavailable', 'message': 'Installation failed; consult backend logs.'}
