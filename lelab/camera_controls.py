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

"""Saved UVC camera settings (white balance and friends), applied to the camera itself.

OpenCV on macOS cannot change a camera's white balance, so ``lelab.utils.uvc`` sends the USB control
requests directly. The values live in the camera and survive a stream open and close, but not a power cycle,
so the ones chosen with ``save`` are written to disk and re-applied when the server starts and whenever the
camera list is read.

Cameras are identified by USB ``vid:pid`` as lowercase hex (``0abd:8050``). Two identical cameras share an
id and the first one found is used.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from typing import Any

from .utils import config, uvc

logger = logging.getLogger(__name__)

_DEVICE_ID = re.compile(r"^([0-9a-fA-F]{1,4}):([0-9a-fA-F]{1,4})$")


class CameraControlsError(ValueError):
    """The camera id or the requested controls are not valid."""


def device_id(vid: int, pid: int) -> str:
    return f"{vid:04x}:{pid:04x}"


def parse_device_id(text: str) -> tuple[int, int]:
    match = _DEVICE_ID.match(text.strip())
    if not match:
        raise CameraControlsError(f"'{text}' is not a USB id like 0abd:8050")
    return int(match.group(1), 16), int(match.group(2), 16)


def validate_controls(controls: Mapping[str, Any]) -> dict[str, int]:
    """Check names and value types without touching a camera. Whether a camera supports a name is checked later."""
    if not controls:
        raise CameraControlsError("No controls given")
    checked: dict[str, int] = {}
    for name, value in controls.items():
        if name not in uvc.CONTROLS:
            raise CameraControlsError(f"Unknown camera control '{name}'. Known: {', '.join(uvc.CONTROLS)}")
        if isinstance(value, bool) or not isinstance(value, int):
            raise CameraControlsError(f"'{name}' must be a whole number, got {value!r}")
        checked[name] = value
    return checked


# -- saved settings ---------------------------------------------------------------------------------


def load_saved() -> dict[str, dict[str, int]]:
    path = config.CAMERA_CONTROLS_FILE
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        logger.warning("Ignoring unreadable camera controls file %s", path)
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): dict(v) for k, v in data.items() if isinstance(v, dict)}


def _write_saved(saved: Mapping[str, Mapping[str, int]]) -> None:
    path = config.CAMERA_CONTROLS_FILE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(saved, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


# -- operations -------------------------------------------------------------------------------------


def list_cameras() -> list[dict[str, Any]]:
    """Every UVC camera with each supported control (range, default, current value) and its saved values."""
    saved = load_saved()
    cameras: list[dict[str, Any]] = []
    for info in uvc.list_cameras():
        ident = device_id(info.vid, info.pid)
        entry: dict[str, Any] = {
            "id": ident,
            "vid": info.vid,
            "pid": info.pid,
            "name": info.name,
            "controls": [],
            "saved": saved.get(ident, {}),
        }
        try:
            with uvc.open_camera(info.vid, info.pid) as camera:
                entry["controls"] = [camera.describe(name) for name in camera.supported()]
        except (uvc.UvcTransferError, uvc.CameraNotFoundError) as exc:
            entry["error"] = str(exc)
        cameras.append(entry)
    return cameras


def set_controls(device: str, controls: Mapping[str, Any], save: bool = False) -> dict[str, int]:
    """Apply controls to one camera and return the values the camera reports back.

    With ``save`` the applied values (after the camera's own clamping) are merged into the saved file, so they
    come back after a replug. Nothing is saved if the camera rejected the request.
    """
    vid, pid = parse_device_id(device)
    checked = validate_controls(controls)
    with uvc.open_camera(vid, pid) as camera:
        applied = camera.apply(checked)
    if save:
        saved = load_saved()
        saved[device_id(vid, pid)] = {**saved.get(device_id(vid, pid), {}), **applied}
        _write_saved(saved)
    return applied


def reset_controls(device: str, forget: bool = True) -> dict[str, int]:
    """Restore the camera's own defaults. With ``forget`` the saved values for it are dropped too."""
    vid, pid = parse_device_id(device)
    with uvc.open_camera(vid, pid) as camera:
        restored = camera.reset()
    if forget:
        saved = load_saved()
        if saved.pop(device_id(vid, pid), None) is not None:
            _write_saved(saved)
    return restored


def apply_saved() -> dict[str, dict[str, int] | str]:
    """Write every saved setting to its camera.

    Returns the applied values per camera id, or a message for a camera that could not be updated (not plugged
    in, rejected a value). One failing camera does not stop the next. With nothing saved USB is not touched.
    """
    results: dict[str, dict[str, int] | str] = {}
    for ident, controls in load_saved().items():
        try:
            vid, pid = parse_device_id(ident)
            with uvc.open_camera(vid, pid) as camera:
                results[ident] = camera.apply(validate_controls(controls))
        except (
            CameraControlsError,
            uvc.UnsupportedControlError,
            uvc.UvcTransferError,
            uvc.CameraNotFoundError,
        ) as exc:
            results[ident] = str(exc)
    return results
