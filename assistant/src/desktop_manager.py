"""Own the project's desktop worker, its IPC connection and its lifecycle."""
import atexit
import json
import logging
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

from desktop_control.protocol import receive_message, send_message

log = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent


class DesktopManager:
    def __init__(self, config_path=None):
        path = Path(config_path) if config_path else PROJECT_DIR / 'desktop_config.json'
        self.config = {'enabled': True, 'mode': 'current', 'width': 1440, 'height': 900}
        if path.exists():
            self.config.update(json.loads(path.read_text()))
        if type(self.config['enabled']) is not bool or self.config['mode'] not in ('current', 'virtual'):
            raise ValueError('Invalid desktop_config.json enabled/mode')
        for dimension in ('width', 'height'):
            if type(self.config[dimension]) is not int or not 320 <= self.config[dimension] <= 7680:
                raise ValueError(f'Invalid desktop {dimension}')
        self.lock = threading.RLock()
        self.send_lock = threading.Lock()
        self.connection = None
        self.process = None
        self.closing = threading.Event()
        self.display = ''
        self.authority = ''
        self.last_error = ''
        atexit.register(self.close)

    @property
    def enabled(self):
        return self.config['enabled']

    def _worker_command(self, fd):
        return [sys.executable, '-m', 'desktop_control.runtime', '--fd', str(fd),
                '--mode', self.config['mode'], '--width', str(self.config['width']),
                '--height', str(self.config['height'])]

    def _send(self, message):
        with self.send_lock:
            send_message(self.connection, message)

    def start(self):
        with self.lock:
            if not self.enabled or self.closing.is_set():
                return False
            if self.process is not None and self.process.poll() is None:
                return True
            self._dispose()
            child = None
            try:
                self.connection, child = socket.socketpair()
                self.connection.settimeout(5)
                self.process = subprocess.Popen(self._worker_command(child.fileno()),
                    cwd=PROJECT_DIR, pass_fds=(child.fileno(),), start_new_session=True)
                child.close()
                ready = receive_message(self.connection, timeout=15)
                if not ready.get('ready'):
                    raise RuntimeError(ready.get('error', 'Desktop worker startup failed'))
                self.display = ready.get('display', '')
                self.authority = ready.get('authority', '')
                self.last_error = ''
                log.info('Desktop worker ready on display %s (%s mode)', self.display, self.config['mode'])
                return True
            except (OSError, EOFError, ValueError, RuntimeError) as error:
                self.last_error = str(error)
                self._dispose()
                log.warning('Desktop unavailable: %s', self.last_error)
                return False
            finally:
                if child is not None:
                    child.close()

    def _dispose(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        process, self.process = self.process, None
        if process is not None:
            # The worker and its Xvfb/Openbox children share this owned process group.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            # Clean up descendants if the worker exited before them.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.display = ''
        self.authority = ''

    def close(self):
        # Signal active input without waiting for the transaction lock.
        self.closing.set()
        with self.lock:
            self._dispose()

    def call(self, command, ctx, body=None):
        if not self.enabled:
            return {'error': 'Desktop control is disabled in desktop_config.json.'}
        if ctx is None:
            return {'error': 'Desktop tools require an active request context.'}
        while not self.lock.acquire(timeout=0.05):
            if ctx.cancel_event.is_set() or self.closing.is_set():
                return {'error': 'Request cancelled.'}
        stop_sent = False
        cancel_started = None

        def cancel_if_needed():
            nonlocal stop_sent, cancel_started
            if (ctx.cancel_event.is_set() or self.closing.is_set()) and not stop_sent:
                self._send({'command': 'stop'})
                stop_sent = True
                cancel_started = time.monotonic()
            if cancel_started is not None and time.monotonic() - cancel_started > 3:
                raise TimeoutError('Desktop cancellation did not drain')

        try:
            if ctx.cancel_event.is_set() or self.closing.is_set():
                return {'error': 'Request cancelled.'}
            if not self.start():
                return {'error': self.last_error or 'Desktop control is unavailable.'}
            self._send({'command': command, 'body': body})
            result = receive_message(self.connection, timeout=130, on_wait=cancel_if_needed)
            cancel_if_needed()
            if stop_sent:
                # Socket ordering ensures stop is processed before resume. The
                # action response has drained before the next command starts.
                self._send({'command': 'resume'})
                receive_message(self.connection, timeout=5)
                return {'error': 'Request cancelled.'}
            return result
        except (OSError, EOFError, ValueError, RuntimeError) as error:
            self.last_error = str(error)
            self._dispose()
            return {'error': 'Desktop worker failed. Input may have partially executed. Observe again before acting.',
                    'detail': self.last_error}
        finally:
            self.lock.release()
