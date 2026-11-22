"""X11 backend. Imports requiring DISPLAY are delayed until startup."""
import base64
import io
import time
from datetime import datetime, timezone
from threading import Event, Lock
from uuid import uuid4

from .models import ActRequest, Observation


class DesktopError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(message)


class X11Backend:
    def __init__(self):
        import mss
        import pyautogui
        from PIL import Image

        self.mss = mss
        self.image = Image
        self.input = pyautogui
        self.input.PAUSE = 0
        self.input.FAILSAFE = True
        # One X screen is used for both capture and input.
        self.width, self.height = self.input.size()

    def capture(self):
        # MSS handles belong to their calling thread.
        with self.mss.mss() as capture:
            shot = capture.grab({"left": 0, "top": 0, "width": self.width, "height": self.height})
            image = self.image.frombytes("RGB", shot.size, shot.rgb)
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue(), self.width, self.height

    def valid_keys(self, keys):
        return all(key in self.input.KEYBOARD_KEYS for key in keys)

    def move(self, point):
        self.input.moveTo(point.x, point.y)

    def down(self, button):
        self.input.mouseDown(button=button)

    def up(self, button):
        # A corner fail-safe must never prevent releasing a held input.
        self.input.FAILSAFE = False
        try:
            self.input.mouseUp(button=button)
        finally:
            self.input.FAILSAFE = True

    def key_down(self, key):
        self.input.keyDown(key)

    def key_up(self, key):
        self.input.FAILSAFE = False
        try:
            self.input.keyUp(key)
        finally:
            self.input.FAILSAFE = True

    def scroll(self, steps):
        self.input.scroll(steps)

    def type_character(self, character):
        key = {"\n": "enter", "\t": "tab"}.get(character, character)
        self.input.press(key)


class Desktop:
    def __init__(self, backend, frame_ttl=60.0):
        self.backend = backend
        self.frame_ttl = frame_ttl
        self.lock = Lock()
        self.stopped = Event()
        self.frame = None
        self.frame_time = 0.0

    def _observe(self):
        png, width, height = self.backend.capture()
        observation = Observation(
            frame_id=uuid4().hex,
            captured_at=datetime.now(timezone.utc).isoformat(),
            width=width,
            height=height,
            image_data_url="data:image/png;base64," + base64.b64encode(png).decode("ascii"),
        )
        self.frame = observation
        self.frame_time = time.monotonic()
        return observation

    def observe(self):
        with self.lock:
            return self._observe()

    def stop(self):
        # Independent of the input lock so an active hold/drag can be interrupted.
        self.stopped.set()
        return {"status": "stopping"}

    def resume(self):
        with self.lock:
            self.stopped.clear()
            self.frame = None
        return {"status": "ready"}

    def _check_stop(self):
        if self.stopped.is_set():
            raise DesktopError(409, "Desktop stopped. Resume and observe before acting.")

    def _wait(self, seconds):
        if self.stopped.wait(seconds):
            self._check_stop()

    def act(self, request: ActRequest):
        with self.lock:
            self._check_stop()
            if self.frame is None or request.frame_id != self.frame.frame_id:
                raise DesktopError(409, "Frame consumed or replaced. Observe again.")
            if time.monotonic() - self.frame_time > self.frame_ttl:
                raise DesktopError(409, "Frame expired. Observe again.")
            action = request.action
            points = action.path if action.kind == "drag" else []
            if action.kind in ("click", "move"):
                points = [action]
            for point in points:
                if point.x >= self.frame.width or point.y >= self.frame.height:
                    raise DesktopError(422, "Coordinates must lie inside the observed image.")
            if action.kind == "keypress" and not self.backend.valid_keys(action.keys):
                raise DesktopError(422, "Unknown key. Use PyAutoGUI key names such as ctrl, enter, left.")
            # Invalidate before execution. Failures may have partially changed the desktop.
            self.frame = None
            try:
                self._execute(action)
                self._wait(0.15)
                self._check_stop()
                return {"status": "ok", "observation": self._observe()}
            except DesktopError:
                raise
            except Exception as error:
                raise DesktopError(503, f"Desktop operation failed ({type(error).__name__}). Observe before retrying.") from error

    def _execute(self, action):
        self._check_stop()
        kind = action.kind
        if kind in ("click", "move"):
            self.backend.move(action)
            if kind == "click":
                for index in range(action.clicks):
                    self._check_stop()
                    try:
                        self.backend.down(action.button)
                    finally:
                        self.backend.up(action.button)
                    if index + 1 < action.clicks:
                        self._wait(0.1)
        elif kind == "drag":
            self.backend.move(action.path[0])
            try:
                self.backend.down(action.button)
                for point in action.path[1:]:
                    self._wait(action.duration / (len(action.path) - 1))
                    self.backend.move(point)
            finally:
                self.backend.up(action.button)
        elif kind == "keypress":
            held = []
            try:
                for key in action.keys:
                    self._check_stop()
                    held.append(key)
                    self.backend.key_down(key)
                self._wait(action.hold)
            finally:
                for key in reversed(held):
                    self.backend.key_up(key)
        elif kind == "type":
            for character in action.text:
                self._check_stop()
                self.backend.type_character(character)
                self._wait(action.interval)
        elif kind == "scroll":
            self.backend.scroll(action.steps)
        elif kind == "wait":
            self._wait(action.seconds)
