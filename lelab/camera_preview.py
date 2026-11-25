# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Live camera pictures for the web UI, read by the server from the OpenCV device LeRobot records from.

The browser cannot tell two identical USB cameras apart, so it never opens a camera. Each attached camera
has at most one reader thread here, shared by every page that shows it. Frames get the camera's saved lens
and colour gain in the order the recorder applies them (``camera_frames.STAGES``), and are not rotated: the
page rotates the picture, as it does the saved reference.
"""

from __future__ import annotations

import logging
import platform
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from . import camera_identity, color_gain, lens
from .color_calibration import CalibrationCamera

logger = logging.getLogger(__name__)

STREAM_FPS = 15.0
WARMUP_FRAMES = 5
FRAME_TIMEOUT_S = 3.0


class PreviewBusyError(RuntimeError):
    """A recording or inference session owns the cameras."""


class PreviewUnavailableError(RuntimeError):
    """The camera could not be opened or did not deliver the asked picture."""


@dataclass(frozen=True)
class PreviewSettings:
    width: int = 640
    height: int = 480
    fps: int = 30
    fourcc: str | None = None
    lens: bool = True
    gain: bool = True

    @classmethod
    def from_camera(cls, camera: Mapping[str, Any]) -> PreviewSettings:
        """The settings a robot profile camera records with."""
        return cls(
            width=int(camera.get("width", 640)),
            height=int(camera.get("height", 480)),
            fps=int(camera.get("fps", 30)),
            fourcc=camera.get("fourcc") or None,
            lens=camera.get("lens_correction", True) is not False,
            gain=camera.get("color_correction", True) is not False,
        )


class Capture(Protocol):
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def release(self) -> None: ...


def open_opencv(index: int, settings: PreviewSettings) -> Capture:
    """Open the camera with the backend the recorder pins (``record._platform_backend``)."""
    import cv2

    backend = {"Darwin": cv2.CAP_AVFOUNDATION, "Linux": cv2.CAP_V4L2, "Windows": cv2.CAP_DSHOW}.get(
        platform.system(), cv2.CAP_ANY
    )
    capture = cv2.VideoCapture(index, backend)
    if not capture.isOpened():
        raise PreviewUnavailableError(f"Could not open the camera at OpenCV index {index}")
    if settings.fourcc:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*settings.fourcc))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    capture.set(cv2.CAP_PROP_FPS, settings.fps)
    return capture


def corrections_for(
    camera: CalibrationCamera, settings: PreviewSettings
) -> Callable[[np.ndarray], np.ndarray]:
    """Lens, then colour gain, as saved under the camera's key by the lens and colour tools."""
    lens_model = lens.load_saved().get(camera.key) if settings.lens else None
    gains = color_gain.load_saved().get(camera.key) if settings.gain else None

    def correct(frame: np.ndarray) -> np.ndarray:
        if lens_model:
            frame = lens.apply_lens(frame, lens_model)
        if gains:
            frame = color_gain.apply_gains(frame, gains)
        return frame

    return correct


class _Reader:
    def __init__(
        self,
        camera_id: str,
        settings: PreviewSettings,
        capture: Capture,
        correct: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self.camera_id, self.settings = camera_id, settings
        self._capture, self._correct = capture, correct
        self._changed = threading.Condition()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._stopped = False
        self.clients = 0
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"preview-{camera_id}")
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stopped:
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue
                corrected = self._correct(frame)
                with self._changed:
                    self._frame, self._seq = corrected, self._seq + 1
                    self._changed.notify_all()
        except Exception as e:
            logger.warning("Preview of %s stopped: %s", self.camera_id, e)
        finally:
            self._capture.release()
            with self._changed:
                self._stopped = True
                self._changed.notify_all()

    def next_frame(self, after: int) -> tuple[int, np.ndarray]:
        with self._changed:
            if not self._changed.wait_for(lambda: self._seq > after or self._stopped, FRAME_TIMEOUT_S):
                raise PreviewUnavailableError(f"Camera {self.camera_id} sent no picture")
            if self._frame is None or self._seq <= after:
                raise PreviewUnavailableError(f"Camera {self.camera_id} closed")
            return self._seq, self._frame

    def stop(self) -> None:
        self._stopped = True
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)


class PreviewHub:
    def __init__(
        self,
        open_capture: Callable[[int, PreviewSettings], Capture] = open_opencv,
        attached: Callable[[], list[CalibrationCamera]] = camera_identity.attached,
        busy: Callable[[], str | None] = lambda: None,
    ) -> None:
        self._open, self._attached, self._busy = open_capture, attached, busy
        self._readers: dict[str, _Reader] = {}
        self._lock = threading.Lock()

    def open_cameras(self) -> list[str]:
        with self._lock:
            return sorted(self._readers)

    def check(self, camera_id: str) -> CalibrationCamera:
        """Raise before a stream starts, so the endpoint can answer with a status code."""
        reason = self._busy()
        if reason:
            raise PreviewBusyError(reason)
        return camera_identity.find(camera_id, self._attached())

    def _acquire(self, camera_id: str, settings: PreviewSettings) -> _Reader:
        camera = self.check(camera_id)
        with self._lock:
            reader = self._readers.get(camera_id)
            if reader is not None and reader.settings != settings:
                # The last caller's settings win; streams at the old ones end and the page reloads them.
                del self._readers[camera_id]
                reader.stop()
                reader = None
            if reader is None:
                capture = self._open(camera.index, settings)
                reader = _Reader(camera_id, settings, capture, corrections_for(camera, settings))
                self._readers[camera_id] = reader
            reader.clients += 1
            return reader

    def _release(self, reader: _Reader) -> None:
        with self._lock:
            reader.clients -= 1
            if reader.clients > 0:
                return
            if self._readers.get(reader.camera_id) is reader:
                del self._readers[reader.camera_id]
        reader.stop()

    def stream(self, camera_id: str, settings: PreviewSettings) -> Iterator[bytes]:
        """Multipart JPEG chunks. The reader is taken on the first chunk and given back when the stream closes."""
        import cv2

        reader = self._acquire(camera_id, settings)
        try:
            seq = 0
            while True:
                started = time.monotonic()
                seq, frame = reader.next_frame(seq)
                ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                time.sleep(max(0.0, 1.0 / STREAM_FPS - (time.monotonic() - started)))
        except PreviewUnavailableError as e:
            logger.info("Preview of %s ended: %s", camera_id, e)
        finally:
            self._release(reader)

    def snapshot(self, camera_id: str, settings: PreviewSettings) -> np.ndarray:
        """A corrected, unrotated frame, taken after a few frames so exposure has settled."""
        reader = self._acquire(camera_id, settings)
        try:
            seq = 0
            for _ in range(WARMUP_FRAMES):
                seq, frame = reader.next_frame(seq)
        finally:
            self._release(reader)
        height, width = frame.shape[:2]
        if (width, height) != (settings.width, settings.height):
            raise PreviewUnavailableError(
                f"The camera delivered {width}×{height} but the profile asks for {settings.width}×{settings.height}"
            )
        return frame

    def close_all(self) -> None:
        with self._lock:
            readers, self._readers = list(self._readers.values()), {}
        for reader in readers:
            reader.stop()


def encode_png(frame: np.ndarray) -> bytes:
    import cv2

    ok, png = cv2.imencode(".png", frame)
    if not ok:
        raise PreviewUnavailableError("Could not encode the picture")
    return png.tobytes()


def _busy() -> str | None:
    from . import record, rollout

    if record.recording_active:
        return "Recording is using the cameras; its own view is on the Record page"
    if rollout.inference_active:
        return "Inference is using the cameras"
    return None


hub = PreviewHub(attached=lambda: camera_identity.attached(), busy=_busy)
