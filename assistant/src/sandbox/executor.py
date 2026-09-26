"""Bounded subprocess transport for Docker CLI only; never a host shell."""
from __future__ import annotations
import os
import selectors
import subprocess
import threading
import time
from collections.abc import Sequence
from .result import BoundedOutput, ExecutionResult


def run_bounded(
    argv: Sequence[str], *, timeout: float, stdout_limit: int, stderr_limit: int,
    cancel_event: threading.Event | None = None,
) -> ExecutionResult:
    start = time.monotonic()
    out, err = BoundedOutput(stdout_limit), BoundedOutput(stderr_limit)
    timed_out = cancelled = False
    # No shell, inherited stdin, temporary output files, or unbounded communicate().
    with subprocess.Popen(list(argv), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, bufsize=0, close_fds=True) as process:
        assert process.stdout is not None and process.stderr is not None
        try:
            with selectors.DefaultSelector() as selector:
                for stream, buffer in ((process.stdout, out), (process.stderr, err)):
                    os.set_blocking(stream.fileno(), False)
                    selector.register(stream, selectors.EVENT_READ, buffer)
                while selector.get_map() or process.poll() is None:
                    cancelled = bool(cancel_event and cancel_event.is_set())
                    timed_out = time.monotonic() - start >= timeout and not cancelled
                    if timed_out or cancelled:
                        break
                    for key, _ in selector.select(timeout=0.05):
                        chunk = os.read(key.fd, 65_536)
                        if chunk:
                            key.data.feed(chunk)
                        else:
                            selector.unregister(key.fileobj)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
    return ExecutionResult(
        out.text(), err.text(), None if timed_out or cancelled else process.returncode,
        int((time.monotonic() - start) * 1000), timed_out, out.truncated,
        err.truncated, cancelled,
    )
