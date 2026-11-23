"""X11 backend. Imports requiring DISPLAY are delayed until startup."""
import base64
import io
import time
from datetime import datetime, timezone
from threading import Event, Lock
from uuid import uuid4

from .models import ActRequest, Observation, ObserveRequest


class DesktopError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(message)


class X11Backend:
    def __init__(self):
        import mss
        import pyautogui
        from PIL import Image
        from Xlib.display import Display

        self.mss = mss
        self.image = Image
        self.input = pyautogui
        self.input.PAUSE = 0
        self.input.FAILSAFE = True
        # One X screen is used for both capture and input.
        self.display = Display()
        self.width, self.height = self.screen_size()

    def screen_size(self):
        # Query current root geometry: PyAutoGUI's X11 screen dimensions can
        # be cached, and coordinate mappings must match the captured screen.
        geometry = self.display.screen().root.get_geometry()
        return geometry.width, geometry.height

    def capture(self):
        self.width, self.height = self.screen_size()
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

    def _require_frame(self, frame_id):
        if self.frame is None or frame_id != self.frame.frame_id:
            raise DesktopError(409, "Frame consumed or replaced. Observe again.")
        if time.monotonic() - self.frame_time > self.frame_ttl:
            raise DesktopError(409, "Frame expired. Observe again.")
        return self.frame

    def _check_geometry(self, reference, width, height):
        if (width, height) != (reference.screen_width, reference.screen_height):
            self.frame = None
            raise DesktopError(409, "Screen dimensions changed. Request a full-screen observation.")

    def _observe(self, region=None, reference=None):
        png, screen_width, screen_height = self.backend.capture()
        origin_x = origin_y = 0
        width, height = screen_width, screen_height
        if region is not None:
            self._check_geometry(reference, screen_width, screen_height)
            origin_x = reference.origin_x + region.x
            origin_y = reference.origin_y + region.y
            width, height = region.width, region.height
            # Crop native pixels on CPU. Never upscale, resize or encode the
            # full-screen image for the model when it requested a region.
            from PIL import Image
            with Image.open(io.BytesIO(png)) as image:
                if image.size != (screen_width, screen_height):
                    raise DesktopError(503, "Capture dimensions do not match the image.")
                crop = image.crop((origin_x, origin_y, origin_x + width, origin_y + height))
                output = io.BytesIO()
                crop.save(output, format="PNG")
                png = output.getvalue()
        observation = Observation(
            frame_id=uuid4().hex,
            captured_at=datetime.now(timezone.utc).isoformat(),
            width=width, height=height,
            origin_x=origin_x, origin_y=origin_y,
            screen_width=screen_width, screen_height=screen_height,
            view="region" if region is not None else "full",
            image_data_url="data:image/png;base64," + base64.b64encode(png).decode("ascii"),
        )
        self.frame = observation
        self.frame_time = time.monotonic()
        return observation

    def observe(self, request=None):
        request = request or ObserveRequest()
        if isinstance(request, dict):
            request = ObserveRequest.model_validate(request)
        with self.lock:
            reference = None
            if request.region is not None:
                reference = self._require_frame(request.frame_id)
                region = request.region
                if region.x + region.width > reference.width or region.y + region.height > reference.height:
                    raise DesktopError(422, "Region must lie inside the referenced image.")
            return self._observe(request.region, reference)

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
            reference = self._require_frame(request.frame_id)
            if hasattr(self.backend, "screen_size"):
                self._check_geometry(reference, *self.backend.screen_size())
            action = request.action
            points = action.path if action.kind == "drag" else []
            if action.kind in ("click", "move"):
                points = [action]
            for point in points:
                if point.x >= self.frame.width or point.y >= self.frame.height:
                    raise DesktopError(422, "Coordinates must lie inside the observed image.")
            if action.kind == "keypress" and not self.backend.valid_keys(action.keys):
                raise DesktopError(422, "Unknown key. Use PyAutoGUI key names such as ctrl, enter, left.")
            # Only the backend-held frame determines the origin. All model
            # coordinates are image-local, including every drag path point.
            if action.kind in ("click", "move"):
                action = action.model_copy(update={
                    "x": reference.origin_x + action.x,
                    "y": reference.origin_y + action.y,
                })
            elif action.kind == "drag":
                action = action.model_copy(update={"path": [
                    point.model_copy(update={"x": reference.origin_x + point.x,
                                             "y": reference.origin_y + point.y})
                    for point in action.path
                ]})
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
