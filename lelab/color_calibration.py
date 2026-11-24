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

"""Automatic colour calibration of a camera against a white sheet.

The camera's white balance is swept across its range and left on the temperature where the sheet comes out
closest to neutral; whatever cast remains is measured and turned into the per-channel gain of
``lelab.color_gain``. ``lelab-camera calibrate`` and the ``lelab-camera-calibrate`` window both drive
``Calibrator``; nothing is saved until ``Calibrator.save``.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from . import camera_controls, color_gain
from .utils import uvc
from .utils.camera_listing import list_cameras_in_cv2_order, usb_id_from_unique_id

Roi = tuple[int, int, int, int]
"""``(x, y, width, height)`` in frame pixels."""

MIN_LEVEL = 50.0
MAX_CLIPPED = 0.10
WARN_CLIPPED = 0.01
WARN_SPREAD = 0.25

_WB_AUTO = "white_balance_temperature_auto"
_WB_TEMPERATURE = "white_balance_temperature"


class CalibrationError(ValueError):
    """The measurement cannot produce a usable gain (too dark, clipped, camera not readable, ...)."""


class CameraNotAttachedError(CalibrationError):
    """No attached camera matches the requested name or index."""


# -- cameras ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationCamera:
    index: int
    name: str
    usb_id: tuple[int, int] | None = None

    @property
    def device(self) -> str | None:
        """The ``vid:pid`` id ``lelab-camera`` uses, for a USB camera."""
        return camera_controls.device_id(*self.usb_id) if self.usb_id else None

    @property
    def label(self) -> str:
        suffix = f"USB {self.device}" if self.device else "no USB controls"
        return f"{self.name} ({suffix}, index {self.index})"


def list_calibration_cameras(listing: Sequence[Mapping[str, Any]] | None = None) -> list[CalibrationCamera]:
    """Attached cameras in OpenCV index order, USB cameras first."""
    entries = list_cameras_in_cv2_order() if listing is None else listing
    cameras = [
        CalibrationCamera(
            index=int(entry["index"]),
            name=color_gain.normalize_name(str(entry.get("name", ""))),
            usb_id=usb_id_from_unique_id(str(entry.get("unique_id", ""))),
        )
        for entry in entries
    ]
    return sorted(cameras, key=lambda camera: (camera.usb_id is None, camera.index))


def find_camera(cameras: Sequence[CalibrationCamera], wanted: str | None) -> CalibrationCamera:
    """Pick a camera by OpenCV index or name; with nothing given, the first USB camera."""
    if not cameras:
        raise CameraNotAttachedError("No cameras found")
    if wanted is None:
        return next((camera for camera in cameras if camera.usb_id), cameras[0])
    text = wanted.strip()
    if text.isdigit():
        match = [camera for camera in cameras if camera.index == int(text)]
    else:
        match = [camera for camera in cameras if camera.name == color_gain.normalize_name(text)]
    if not match:
        known = ", ".join(f"'{camera.name}' ({camera.index})" for camera in cameras)
        raise CameraNotAttachedError(f"No camera '{wanted}'. Attached: {known}")
    return match[0]


# -- measuring --------------------------------------------------------------------------------------


def default_roi(width: int, height: int, fraction: float = 0.3) -> Roi:
    """A centred box covering ``fraction`` of each side, where the white sheet goes."""
    w, h = max(1, round(width * fraction)), max(1, round(height * fraction))
    return (width - w) // 2, (height - h) // 2, w, h


def clamp_roi(roi: Roi, width: int, height: int) -> Roi:
    x, y, w, h = roi
    x = min(max(0, x), width - 1)
    y = min(max(0, y), height - 1)
    return x, y, max(1, min(w, width - x)), max(1, min(h, height - y))


@dataclass(frozen=True)
class PatchStats:
    """Median level of each channel inside the box, with the share of clipped pixels and how uneven it is."""

    blue: float
    green: float
    red: float
    clipped: float
    spread: float

    @property
    def red_ratio(self) -> float:
        return self.red / max(self.green, 1.0)

    @property
    def blue_ratio(self) -> float:
        return self.blue / max(self.green, 1.0)


def measure_patch(frames: Sequence[np.ndarray], roi: Roi) -> PatchStats:
    """Measure the box over BGR frames. Medians ignore specks, glare and the odd noisy frame."""
    if not frames:
        raise CalibrationError("No frames to measure")
    height, width = frames[0].shape[:2]
    x, y, w, h = clamp_roi(roi, width, height)
    pixels = np.concatenate([frame[y : y + h, x : x + w].reshape(-1, 3) for frame in frames])
    blue, green, red = (float(v) for v in np.median(pixels, axis=0))
    clipped = float(np.mean(np.any(pixels >= 255, axis=1)))
    p10, p90 = np.percentile(pixels[:, 1], [10, 90])
    spread = float((p90 - p10) / max(green, 1.0))
    return PatchStats(blue=blue, green=green, red=red, clipped=clipped, spread=spread)


def neutrality(stats: PatchStats) -> float:
    """0 for a perfectly neutral white; grows with the colour cast either way."""
    green = max(stats.green, 1.0)
    return abs(math.log(max(stats.red, 1.0) / green)) + abs(math.log(max(stats.blue, 1.0) / green))


def check_exposure(stats: PatchStats) -> None:
    """Refuse a box that is too dark or too bright to say anything about its colour."""
    if max(stats.blue, stats.green, stats.red) < MIN_LEVEL:
        raise CalibrationError(
            "The box is too dark to measure. Put the white sheet in the box and light it, or open up the exposure."
        )
    if min(stats.blue, stats.green, stats.red) < MIN_LEVEL:
        raise CalibrationError(
            "The box is strongly coloured and does not look like a white sheet. Move the box onto the sheet."
        )
    if stats.clipped > MAX_CLIPPED:
        raise CalibrationError(
            f"{stats.clipped:.0%} of the box is overexposed. Dim the light or lower the exposure, then try again."
        )


def gains_from_patch(stats: PatchStats) -> tuple[dict[str, float], list[str]]:
    """The gain that makes the box neutral (green stays 1.0), with warnings about a doubtful measurement."""
    check_exposure(stats)
    raw = {"blue": stats.green / stats.blue, "green": 1.0, "red": stats.green / stats.red}
    try:
        gains = color_gain.validate_gains({name: round(value, 4) for name, value in raw.items()})
    except color_gain.ColorGainError:
        raise CalibrationError(
            f"The cast is too strong to correct (needs blue ×{raw['blue']:.2f}, red ×{raw['red']:.2f}; "
            f"the limit is ×{color_gain.MAX_GAIN:g}). Check the box only covers the white sheet."
        ) from None
    warnings = []
    if stats.clipped > WARN_CLIPPED:
        warnings.append(f"{stats.clipped:.0%} of the box is overexposed; the gain may be slightly off.")
    if stats.spread > WARN_SPREAD:
        warnings.append("The box is uneven. Make sure it only covers the white sheet and the light is even.")
    return gains, warnings


# -- the camera -------------------------------------------------------------------------------------


class FrameSource(Protocol):
    def read(self) -> np.ndarray: ...


class WhiteBalance(Protocol):
    def range(self) -> tuple[int, int, int]: ...
    def set(self, kelvin: int) -> int: ...
    def save(self, kelvin: int) -> int: ...


class OpenCVFrameSource:
    """A camera opened the way LeRobot opens it (AVFoundation on macOS), giving BGR frames."""

    def __init__(self, index: int, size: tuple[int, int] | None = None) -> None:
        import cv2

        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        self._capture = cv2.VideoCapture(index, backend)
        if not self._capture.isOpened():
            raise CalibrationError(f"Could not open camera {index}. Is another app holding it?")
        if size is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])

    def read(self) -> np.ndarray:
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CalibrationError("The camera stopped sending frames")
        return frame

    def close(self) -> None:
        self._capture.release()


class UvcWhiteBalance:
    """The camera's own white balance over USB, through ``camera_controls`` so the saved file is shared."""

    def __init__(self, device: str) -> None:
        self.device = device

    def range(self) -> tuple[int, int, int]:
        vid, pid = camera_controls.parse_device_id(self.device)
        with uvc.open_camera(vid, pid) as camera:
            info = camera.describe(_WB_TEMPERATURE)
        return int(info["min"]), int(info["max"]), int(info.get("step", 1))

    def set(self, kelvin: int) -> int:
        return self._write(kelvin, save=False)

    def save(self, kelvin: int) -> int:
        return self._write(kelvin, save=True)

    def _write(self, kelvin: int, save: bool) -> int:
        applied = camera_controls.set_controls(self.device, {_WB_AUTO: 0, _WB_TEMPERATURE: kelvin}, save=save)
        return applied.get(_WB_TEMPERATURE, kelvin)


def white_balance_for(camera: CalibrationCamera) -> UvcWhiteBalance | None:
    return UvcWhiteBalance(camera.device) if camera.device else None


# -- calibration ------------------------------------------------------------------------------------


@dataclass
class CalibrationResult:
    temperature: int | None
    gains: dict[str, float]
    before: PatchStats
    after: PatchStats
    warnings: list[str] = field(default_factory=list)
    sweep: list[tuple[int, float]] = field(default_factory=list)
    """``(kelvin, neutrality)`` for each temperature tried."""


Progress = Callable[[float, str], None]


def temperature_steps(low: int, high: int, step: int, count: int) -> list[int]:
    """``count`` temperatures from ``low`` to ``high``, on the camera's step."""
    step = max(step, 1)
    if high <= low or count < 2:
        return [low]
    values = [low + round((high - low) * i / (count - 1) / step) * step for i in range(count)]
    return sorted({min(value, high) for value in values})


class Calibrator:
    def __init__(
        self,
        source: FrameSource,
        white_balance: WhiteBalance | None = None,
        *,
        settle_frames: int = 15,
        sample_frames: int = 6,
        sweep_steps: int = 7,
        on_frame: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        self.source = source
        self.white_balance = white_balance
        self.settle_frames = settle_frames
        self.sample_frames = sample_frames
        self.sweep_steps = sweep_steps
        self.on_frame = on_frame

    def _read(self) -> np.ndarray:
        frame = self.source.read()
        if self.on_frame is not None:
            self.on_frame(frame)
        return frame

    def _settle(self) -> None:
        for _ in range(self.settle_frames):
            self._read()

    def _sample(self) -> list[np.ndarray]:
        return [self._read() for _ in range(self.sample_frames)]

    def _sweep(self, roi: Roi, progress: Progress) -> tuple[int, list[tuple[int, float]]]:
        assert self.white_balance is not None
        steps = temperature_steps(*self.white_balance.range(), self.sweep_steps)
        tried: list[tuple[int, float]] = []
        for i, kelvin in enumerate(steps):
            progress(i / (len(steps) + 1), f"Trying white balance {kelvin} K")
            applied = self.white_balance.set(kelvin)
            self._settle()
            tried.append((applied, neutrality(measure_patch(self._sample(), roi))))
        best = min(tried, key=lambda entry: entry[1])[0]
        progress(len(steps) / (len(steps) + 1), f"Measuring the gain at {best} K")
        self.white_balance.set(best)
        self._settle()
        return best, tried

    def run(self, roi: Roi, progress: Progress | None = None) -> CalibrationResult:
        """Sweep the white balance (when the camera has one), then measure the gain. Saves nothing."""
        report = progress or (lambda fraction, message: None)
        temperature, tried = None, []
        report(0.0, "Checking the white sheet")
        check_exposure(measure_patch(self._sample(), roi))
        if self.white_balance is not None:
            temperature, tried = self._sweep(roi, report)
        else:
            self._settle()
        frames = self._sample()
        before = measure_patch(frames, roi)
        gains, warnings = gains_from_patch(before)
        after = measure_patch([color_gain.apply_gains(frame, gains) for frame in frames], roi)
        if self.white_balance is None:
            warnings.append("This camera has no USB white-balance control, so only the gain was measured.")
        report(1.0, "Done")
        return CalibrationResult(temperature, gains, before, after, warnings, tried)

    def save(self, result: CalibrationResult, name: str) -> dict[str, float]:
        """Keep the calibration: the temperature in the camera-controls file, the gain under the camera name."""
        if self.white_balance is not None and result.temperature is not None:
            self.white_balance.save(result.temperature)
        return color_gain.save_gains(name, result.gains)


def describe_result(result: CalibrationResult) -> list[str]:
    """The result as short lines for a terminal or a status panel."""
    lines = []
    if result.temperature is not None:
        tried = ", ".join(f"{kelvin} K ({score:.2f})" for kelvin, score in result.sweep)
        lines.append(f"White balance {result.temperature} K (tried {tried}; 0 is neutral)")
    before, after, gains = result.before, result.after, result.gains
    lines.append(f"White sheet: red/green {before.red_ratio:.2f}, blue/green {before.blue_ratio:.2f}")
    lines.append(f"Gain: blue ×{gains['blue']:.3f}, green ×{gains['green']:.3f}, red ×{gains['red']:.3f}")
    lines.append(f"Corrected: red/green {after.red_ratio:.2f}, blue/green {after.blue_ratio:.2f}")
    return lines + result.warnings
