"""Length-prefixed JSON over an inherited Unix socket."""
import json
import select
import struct
import time

MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def send_message(connection, message):
    encoded = json.dumps(message).encode('utf-8')
    if len(encoded) > MAX_MESSAGE_BYTES:
        raise ValueError('Desktop message exceeds 64 MiB')
    connection.sendall(struct.pack('!I', len(encoded)) + encoded)


def receive_message(connection, timeout=None, on_wait=None):
    deadline = None if timeout is None else time.monotonic() + timeout

    def read_exact(count):
        parts = []
        remaining = count
        while remaining:
            if on_wait is not None:
                on_wait()
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('Desktop worker response timed out')
            if not select.select([connection], [], [], 0.02)[0]:
                continue
            block = connection.recv(remaining)
            if not block:
                raise EOFError('Desktop worker disconnected')
            parts.append(block)
            remaining -= len(block)
        return b''.join(parts)

    size = struct.unpack('!I', read_exact(4))[0]
    if size > MAX_MESSAGE_BYTES:
        raise ValueError('Desktop message exceeds 64 MiB')
    message = json.loads(read_exact(size))
    if not isinstance(message, dict):
        raise ValueError('Desktop message must be an object')
    return message
