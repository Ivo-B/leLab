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

import logging
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from . import camera_controls, color_gain
from .utils import uvc
from .utils.camera_listing import (
    CameraKeyError,
    camera_key,
    list_cameras_in_cv2_order,
    usb_id_from_unique_id,
    usb_location_from_unique_id,
)

logger = logging.getLogger(__name__)

Roi = tuple[int, int, int, int]
"""``(x, y, width, height)`` in frame pixels."""

MIN_LEVEL = 50.0
# A channel below MIN_LEVEL and below this share of the brightest one marks a box that is not a white sheet.
COLOUR_RATIO = 0.25
MAX_CLIPPED = 0.10
WARN_CLIPPED = 0.01
WARN_SPREAD = 0.25
# The camera's brightness is fitted so the sheet's green median sits in GREEN_LOW..GREEN_HIGH with (almost)
# nothing clipped. A raise aims at GREEN_TARGET, which is below where a typical sheet starts to clip; a clipped
# sheet is lowered by at least CLIP_STEP, so the two never chase each other.
GREEN_LOW = 120.0
GREEN_HIGH = 215.0
GREEN_TARGET = 140.0
CLIP_STEP = 8
FIT_STEPS = 8

_WB_AUTO = "white_balance_temperature_auto"
_WB_TEMPERATURE = "white_balance_temperature"
_BRIGHTNESS = "brightness"


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
    unique_id: str = ""
    location: int | None = None
    """USB bus and port path. With ``usb_id`` it picks this camera among identical ones."""

    @property
    def key(self) -> str:
        """What the colour gain is saved under: the name and the id, so two cameras with one name stay apart."""
        return camera_key(self.name, self.unique_id)

    @property
    def device(self) -> str | None:
        """The ``vid:pid@location`` id ``lelab-camera`` uses, for a USB camera."""
        return camera_controls.device_id(*self.usb_id, self.location) if self.usb_id else None

    @property
    def label(self) -> str:
        """The camera as `lelab-camera list` names it (`0abd:8050@130000`), plus its OpenCV index."""
        if not self.device:
            return f"{self.name} (no USB controls, index {self.index})"
        return f"{self.name} (USB {self.device}, index {self.index})"


def list_calibration_cameras(listing: Sequence[Mapping[str, Any]] | None = None) -> list[CalibrationCamera]:
    """Attached cameras in OpenCV index order, USB cameras first."""
    entries = list_cameras_in_cv2_order() if listing is None else listing
    cameras = [
        CalibrationCamera(
            index=int(entry["index"]),
            name=color_gain.normalize_name(str(entry.get("name", ""))),
            usb_id=usb_id_from_unique_id(str(entry.get("unique_id", ""))),
            unique_id=str(entry.get("unique_id", "")),
            location=usb_location_from_unique_id(str(entry.get("unique_id", ""))),
        )
        for entry in entries
    ]
    return sorted(cameras, key=lambda camera: (camera.usb_id is None, camera.index))


def find_camera(cameras: Sequence[CalibrationCamera], wanted: str | None) -> CalibrationCamera:
    """Pick a camera by OpenCV index, key or name; with nothing given, the first USB camera.

    A name shared by several cameras is an error that lists how to pick one, never a guess.
    """
    if not cameras:
        raise CameraNotAttachedError("No cameras found")
    if wanted is None:
        return next((camera for camera in cameras if camera.usb_id), cameras[0])
    text = wanted.strip()
    if text.isdigit():
        match = [camera for camera in cameras if camera.index == int(text)]
    else:
        match = [camera for camera in cameras if camera.key == text]
        if not match:
            match = [camera for camera in cameras if camera.name == color_gain.normalize_name(text)]
    if not match:
        known = ", ".join(f"'{camera.name}' ({camera.index})" for camera in cameras)
        raise CameraNotAttachedError(f"No camera '{wanted}'. Attached: {known}")
    if len(match) > 1:
        options = ", ".join(f"index {camera.index} ({camera.label})" for camera in match)
        raise CameraKeyError(f"{len(match)} cameras are named '{text}'. Pick one by its index: {options}")
    return match[0]


def _listing(cameras: Sequence[CalibrationCamera]) -> list[dict[str, Any]]:
    return [{"index": c.index, "name": c.name, "unique_id": c.unique_id} for c in cameras]


def saved_gains_for(
    camera: CalibrationCamera, cameras: Sequence[CalibrationCamera]
) -> dict[str, float] | None:
    """The gain saved for this camera: under its own key, or under its name when no other camera has it."""
    return color_gain.index_gains(_listing(cameras), color_gain.load_saved()).get(camera.index)


def forget_gain_for(camera: CalibrationCamera, cameras: Sequence[CalibrationCamera]) -> None:
    """Drop the gain saved for this camera, including a name-only entry that applied only to it."""
    color_gain.forget_gains(camera.key)
    legacy = [other for other in cameras if other.name == camera.name]
    if len(legacy) == 1:
        color_gain.forget_gains(camera.name)


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


def check_exposure(stats: PatchStats, adjusted: bool = False, expect_white: bool = True) -> None:
    """Refuse a box that is too dark or too bright to say anything about its colour.

    ``adjusted`` says the camera's brightness was already fitted, so asking for a different one would not help.
    ``expect_white`` is False before the white balance is swept: a camera's automatic white balance can leave a
    real white sheet nearly blue-less, which the sweep then corrects, so only the sheet's brightness is judged.
    """
    if max(stats.blue, stats.green, stats.red) < MIN_LEVEL:
        raise CalibrationError(
            "The box is too dark to measure. Put the white sheet in the box and light it, or add light."
        )
    dimmest = min(stats.blue, stats.green, stats.red)
    if (
        expect_white
        and dimmest < MIN_LEVEL
        and dimmest < COLOUR_RATIO * max(stats.blue, stats.green, stats.red)
    ):
        raise CalibrationError(
            "The box is strongly coloured and does not look like a white sheet. Move the box onto the sheet."
        )
    if stats.clipped > MAX_CLIPPED:
        if adjusted:
            raise CalibrationError(
                f"{stats.clipped:.0%} of the box is overexposed even with the camera's brightness turned all the way down. "
                "Dim the light or move the sheet away from it, then try again."
            )
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
    def remember(self) -> None: ...
    def restore(self) -> None: ...
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
        self._before: dict[str, int] = {}

    def range(self) -> tuple[int, int, int]:
        vid, pid, location = camera_controls.parse_device(self.device)
        with uvc.open_camera(vid, pid, location) as camera:
            info = camera.describe(_WB_TEMPERATURE)
        return int(info["min"]), int(info["max"]), int(info.get("step", 1))

    def remember(self) -> None:
        self._before = _read_controls(self.device, (_WB_AUTO, _WB_TEMPERATURE))

    def restore(self) -> None:
        _restore_controls(self.device, self._before, _WB_AUTO)

    def set(self, kelvin: int) -> int:
        return self._write(kelvin, save=False)

    def save(self, kelvin: int) -> int:
        return self._write(kelvin, save=True)

    def _write(self, kelvin: int, save: bool) -> int:
        applied = camera_controls.set_controls(self.device, {_WB_AUTO: 0, _WB_TEMPERATURE: kelvin}, save=save)
        return applied.get(_WB_TEMPERATURE, kelvin)


def _read_controls(device: str, names: Sequence[str]) -> dict[str, int]:
    vid, pid, location = camera_controls.parse_device(device)
    with uvc.open_camera(vid, pid, location) as camera:
        return {name: camera.read(name) for name in names}


def _restore_controls(device: str, before: Mapping[str, int], auto: str) -> None:
    """Put a camera back to the values read before: just the auto switch when it was automatic.

    A manual value cannot be written while the camera is in automatic mode, and does not matter there.
    """
    if not before:
        return
    wanted = dict(before) if before[auto] == uvc.GATES[auto] else {auto: before[auto]}
    camera_controls.set_controls(device, wanted, save=False)


def white_balance_for(camera: CalibrationCamera) -> UvcWhiteBalance | None:
    return UvcWhiteBalance(camera.device) if camera.device else None


class Brightness(Protocol):
    def range(self) -> tuple[int, int, int]: ...
    def start(self) -> int: ...
    def set(self, value: int) -> int: ...
    def save(self, value: int) -> int: ...
    def restore(self) -> None: ...


class UvcBrightness:
    """The camera's own brightness offset over USB.

    The camera's automatic exposure stays on: it also raises the sensor gain, which a fixed exposure time cannot
    (a locked 33 ms exposure gave a sheet at 19 of 255 where automatic gave 150 at the same 30 fps). The
    brightness offset then moves the picture up or down by about one level per unit.
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self._before: int | None = None

    def range(self) -> tuple[int, int, int]:
        vid, pid, location = camera_controls.parse_device(self.device)
        with uvc.open_camera(vid, pid, location) as camera:
            info = camera.describe(_BRIGHTNESS)
        return int(info["min"]), int(info["max"]), int(info.get("step", 1))

    def start(self) -> int:
        """The brightness the camera has now, remembered so ``restore`` can put it back."""
        self._before = _read_controls(self.device, (_BRIGHTNESS,))[_BRIGHTNESS]
        return self._before

    def restore(self) -> None:
        if self._before is not None:
            camera_controls.set_controls(self.device, {_BRIGHTNESS: self._before}, save=False)

    def set(self, value: int) -> int:
        return self._write(value, save=False)

    def save(self, value: int) -> int:
        return self._write(value, save=True)

    def _write(self, value: int, save: bool) -> int:
        applied = camera_controls.set_controls(self.device, {_BRIGHTNESS: value}, save=save)
        return applied.get(_BRIGHTNESS, value)


def brightness_for(camera: CalibrationCamera) -> UvcBrightness | None:
    """The camera's brightness control, or ``None`` when it is not a USB camera or has none."""
    if not camera.device:
        return None
    vid, pid, location = camera_controls.parse_device(camera.device)
    try:
        with uvc.open_camera(vid, pid, location) as usb_camera:
            supported = set(usb_camera.supported())
    except (uvc.UvcUnavailableError, uvc.UvcTransferError, uvc.CameraNotFoundError):
        return None
    return UvcBrightness(camera.device) if _BRIGHTNESS in supported else None


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
    brightness: int | None = None
    """The brightness the camera was left at, when the camera has the control."""


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
        brightness: Brightness | None = None,
        settle_frames: int = 15,
        sample_frames: int = 6,
        sweep_steps: int = 7,
        on_frame: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        self.source = source
        self.white_balance = white_balance
        self.brightness = brightness
        self._level: int | None = None
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

    def _fit_brightness(self, roi: Roi, progress: Progress) -> None:
        """Move the camera's brightness until the sheet's green median is in range and (almost) nothing clips.

        One brightness unit moves the picture by about one level, so a dim sheet is raised by the distance to
        ``GREEN_TARGET``; a clipped one is lowered by at least ``CLIP_STEP``. Stops at the camera's limits, where
        ``check_exposure`` then decides whether the sheet is usable.
        """
        assert self.brightness is not None and self._level is not None
        low, high, step = self.brightness.range()
        step = max(step, 1)
        progress(0.0, "Setting the brightness")
        for _ in range(FIT_STEPS):
            stats = measure_patch(self._sample(), roi)
            clipped = stats.clipped > WARN_CLIPPED
            if GREEN_LOW <= stats.green <= GREEN_HIGH and not clipped:
                return
            shift = round(GREEN_TARGET - stats.green)
            if clipped:
                shift = min(shift, -CLIP_STEP)
            wanted = low + round((self._level + shift - low) / step) * step
            wanted = min(max(wanted, low), high)
            if wanted == self._level:
                return
            self._level = self.brightness.set(wanted)
            self._settle()

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
        if self.brightness is not None:
            self._fit_brightness(roi, progress)  # the white balance changes the brightness
        return best, tried

    def run(self, roi: Roi, progress: Progress | None = None) -> CalibrationResult:
        """Fit the brightness and sweep the white balance (when the camera has them), then measure the gain.

        Saves nothing. The camera is left at the calibrated settings so the result can be checked; when the
        measurement fails it is put back as it was, and ``restore`` does the same for a result that is not kept.
        """
        if self.white_balance is not None:
            self.white_balance.remember()
        try:
            return self._run(roi, progress)
        except Exception:
            self.restore()
            raise

    def restore(self) -> None:
        """Put the camera's white balance and brightness back as they were before ``run``."""
        for control in (self.brightness, self.white_balance):
            if control is None:
                continue
            try:
                control.restore()
            except (
                uvc.UvcUnavailableError,
                uvc.UvcTransferError,
                uvc.UnsupportedControlError,
                uvc.CameraNotFoundError,
                camera_controls.CameraControlsError,
            ) as exc:
                logger.warning("Could not restore the camera's settings: %s", exc)

    def _run(self, roi: Roi, progress: Progress | None) -> CalibrationResult:
        report = progress or (lambda fraction, message: None)
        temperature, tried = None, []
        report(0.0, "Checking the white sheet")
        if self.brightness is not None:
            self._level = self.brightness.start()
            self._fit_brightness(roi, report)
        check_exposure(
            measure_patch(self._sample(), roi),
            adjusted=self.brightness is not None,
            expect_white=self.white_balance is None,
        )
        if self.white_balance is not None:
            temperature, tried = self._sweep(roi, report)
        else:
            self._settle()
        frames = self._sample()
        before = measure_patch(frames, roi)
        gains, warnings = gains_from_patch(before)
        after = measure_patch([color_gain.apply_gains(frame, gains) for frame in frames], roi)
        if self.brightness is not None and before.green < GREEN_LOW:
            warnings.append(
                f"The sheet is dim (green {before.green:.0f} of 255) even with the camera's brightness turned up. "
                "Add light for a cleaner measurement."
            )
        if self.white_balance is None:
            warnings.append("This camera has no USB white-balance control, so only the gain was measured.")
        report(1.0, "Done")
        return CalibrationResult(temperature, gains, before, after, warnings, tried, self._level)

    def save(self, result: CalibrationResult, name: str) -> dict[str, float]:
        """Keep the calibration: temperature and brightness in the camera-controls file, the gain under the camera key."""
        if self.white_balance is not None and result.temperature is not None:
            self.white_balance.save(result.temperature)
        if self.brightness is not None and result.brightness is not None:
            self.brightness.save(result.brightness)
        return color_gain.save_gains(name, result.gains)


def describe_result(result: CalibrationResult) -> list[str]:
    """The result as short lines for a terminal or a status panel."""
    lines = []
    if result.temperature is not None:
        tried = ", ".join(f"{kelvin} K ({score:.2f})" for kelvin, score in result.sweep)
        lines.append(f"White balance {result.temperature} K (tried {tried}; 0 is neutral)")
    if result.brightness is not None:
        lines.append(f"Brightness {result.brightness:+d} (the camera's automatic exposure stays on)")
    before, after, gains = result.before, result.after, result.gains
    lines.append(f"White sheet: red/green {before.red_ratio:.2f}, blue/green {before.blue_ratio:.2f}")
    lines.append(f"Gain: blue ×{gains['blue']:.3f}, green ×{gains['green']:.3f}, red ×{gains['red']:.3f}")
    lines.append(f"Corrected: red/green {after.red_ratio:.2f}, blue/green {after.blue_ratio:.2f}")
    return lines + result.warnings
