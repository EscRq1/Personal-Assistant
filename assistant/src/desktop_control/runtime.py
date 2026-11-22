"""Private child entry point, launched by desktop_manager.py."""
import argparse
import os
from pathlib import Path
import queue
import secrets
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time

from .protocol import receive_message, send_message


def virtual_display(width, height):
    """Create an authenticated Xvfb display and its window manager."""
    for executable in ('Xvfb', 'xauth', 'openbox'):
        if shutil.which(executable) is None:
            raise RuntimeError(f'Missing {executable}. Install xvfb, xauth and openbox for virtual mode.')
    temporary = tempfile.TemporaryDirectory(prefix='assistant-desktop-')
    processes = []
    try:
        authority = str(Path(temporary.name) / 'Xauthority')
        Path(authority).touch(mode=0o600)
        for number in range(90, 200):
            if Path(f'/tmp/.X{number}-lock').exists() or Path(f'/tmp/.X11-unix/X{number}').exists():
                continue
            display = f':{number}'
            subprocess.run(['xauth', '-f', authority, 'add', display, '.', secrets.token_hex(16)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            process = subprocess.Popen([
                'Xvfb', display, '-screen', '0', f'{width}x{height}x24',
                '-auth', authority, '-nolisten', 'tcp', '-noreset',
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            processes.append(process)
            for _ in range(100):
                if process.poll() is not None:
                    break
                lock = Path(f'/tmp/.X{number}-lock')
                if lock.exists() and lock.read_text().strip() == str(process.pid) and Path(f'/tmp/.X11-unix/X{number}').exists():
                    os.environ['DISPLAY'] = display
                    os.environ['XAUTHORITY'] = authority
                    processes.append(subprocess.Popen(['openbox', '--sm-disable'],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
                    return temporary, processes
                time.sleep(0.02)
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=2)
        raise RuntimeError('Could not allocate a virtual X11 display')
    except BaseException:
        cleanup_display(temporary, processes)
        raise


def cleanup_display(temporary, processes):
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    if temporary is not None:
        temporary.cleanup()


def serve(connection, desktop):
    """A reader handles stop immediately while the main thread executes input."""
    pending = queue.Queue()

    def read_commands():
        try:
            while True:
                command = receive_message(connection)
                if command.get('command') == 'stop':
                    desktop.stop()
                else:
                    pending.put(command)
                    if command.get('command') == 'close':
                        desktop.stop()
                        return
        except (OSError, EOFError, ValueError):
            desktop.stop()
            pending.put({'command': 'close'})

    reader = threading.Thread(target=read_commands, daemon=True)
    reader.start()
    send_message(connection, {'ready': True, 'display': os.getenv('DISPLAY', ''),
                             'authority': os.getenv('XAUTHORITY', '')})
    while True:
        command = pending.get()
        name = command.get('command')
        if name == 'close':
            return
        try:
            from .models import ActRequest
            if name == 'observe':
                result = desktop.observe().model_dump()
            elif name == 'act':
                result = desktop.act(ActRequest.model_validate(command['body']))
                result['observation'] = result['observation'].model_dump()
            elif name == 'resume':
                result = desktop.resume()
            else:
                raise ValueError('Unknown desktop command')
        except Exception as error:
            result = {'error': str(error), 'error_type': type(error).__name__}
        send_message(connection, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fd', type=int, required=True)
    parser.add_argument('--mode', choices=('current', 'virtual'), required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    args = parser.parse_args()
    connection = socket.socket(fileno=args.fd)
    temporary, processes = None, []
    desktop = None

    def terminate(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    try:
        if args.mode == 'virtual':
            temporary, processes = virtual_display(args.width, args.height)
        elif not os.getenv('DISPLAY'):
            raise RuntimeError('No X11 DISPLAY. Launch from your desktop session or set mode to virtual in desktop_config.json.')
        from .desktop import Desktop, X11Backend
        desktop = Desktop(X11Backend())
        serve(connection, desktop)
    except Exception as error:
        try:
            send_message(connection, {'error': str(error), 'error_type': type(error).__name__})
        except OSError:
            pass
    finally:
        if desktop is not None:
            desktop.stop()
        connection.close()
        cleanup_display(temporary, processes)


if __name__ == '__main__':
    main()
