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

"""One id per physical camera, and the OpenCV index it has right now.

On macOS the id is AVFoundation's uniqueID (``0x1300000abd8050``). It holds the USB port, so two identical
cameras get different ids and keep them across restarts. Elsewhere the OS lists no such id and a camera is
known by its OpenCV index (``index:2``). The web page, ``lelab-camera-calibrate`` and ``lelab-camera-lens``
all describe a camera with the same ``CalibrationCamera``: one label, one key for its saved lens and gain.
Everything that opens a camera (recording, inference, previews, snapshots) turns the id into an index then.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from .color_calibration import CalibrationCamera, list_calibration_cameras
from .utils.camera_platforms import list_cameras_for_platform

LISTING_TTL_SECONDS = 3.0
_cache: tuple[float, list[CalibrationCamera]] | None = None


class CameraNotAttachedError(LookupError):
    """No attached camera has this id."""


def attached(fresh: bool = False) -> list[CalibrationCamera]:
    """The attached cameras, USB first. Without ``fresh`` a listing up to a few seconds old is reused."""
    global _cache
    now = time.monotonic()
    if fresh or _cache is None or now - _cache[0] > LISTING_TTL_SECONDS:
        _cache = (now, list_calibration_cameras(list_cameras_for_platform()))
    return _cache[1]


def find(camera_id: str, cameras: list[CalibrationCamera]) -> CalibrationCamera:
    for camera in cameras:
        if camera.camera_id == camera_id:
            return camera
    raise CameraNotAttachedError(f"Camera {camera_id} is not attached. Plug it into the same USB port again.")


def resolve_session_cameras(
    cameras: Mapping[str, Mapping[str, Any]], listing: list[CalibrationCamera]
) -> dict[str, dict[str, Any]]:
    """The session cameras with the OpenCV index their id has now. Raises if one is unset or unplugged."""
    resolved: dict[str, dict[str, Any]] = {}
    for name, camera in cameras.items():
        camera_id = camera.get("camera_id")
        if not camera_id:
            raise ValueError(f"Pick the camera for '{name}' again on the Calibration page.")
        resolved[name] = {**camera, "camera_index": find(str(camera_id), listing).index}
    return resolved


def describe(camera: CalibrationCamera) -> dict[str, Any]:
    """What the web page shows for a camera, with the lens and gain saved under its key."""
    from . import color_gain, lens

    return {
        "camera_id": camera.camera_id,
        "index": camera.index,
        "name": camera.name,
        "usb_id": camera.device,
        "label": camera.label,
        "key": camera.key,
        "lens": lens.load_saved().get(camera.key),
        "gain": color_gain.load_saved().get(camera.key),
    }
