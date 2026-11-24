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

"""Fixed per-channel colour gain for one camera, applied to every frame LeRobot reads from it.

The camera's lowest white-balance setting (2800 K on the Waveshare top camera) still leaves a green-yellow
cast. A gain measured once from a white patch (``lelab-camera gain``) removes it. Gains are saved per camera
*name* (the label shown by the OS and the browser), because OpenCV indices move when cameras are replugged.

Recording, the live view and inference all read frames through ``OpenCVCamera._postprocess_image``;
``install`` registers the gains with ``lelab.camera_frames``, which wraps that method. The inference
subprocess receives the same index map via the environment.
"""

from __future__ import annotations

import json
import logging
import math
import platform
from collections.abc import Mapping
from functools import partial
from typing import Any

import numpy as np

from . import camera_frames
from .utils import config
from .utils.camera_listing import index_by_name, list_cameras_in_cv2_order
from .utils.camera_store import CameraStore, clean_name

logger = logging.getLogger(__name__)

# OpenCV frames are BGR, so the gain vector is ordered blue, green, red.
CHANNELS = ("blue", "green", "red")
MAX_GAIN = 4.0
ENV_VAR = "LELAB_CAMERA_GAINS"

Gains = dict[str, float]


class ColorGainError(ValueError):
    """The camera name or the gain values are not valid."""


def normalize_name(name: str) -> str:
    """Camera names come from the OS with stray spaces (``"Innomaker-U20CAM-720P  "``); compare without them."""
    return clean_name(name, ColorGainError)


def validate_gains(values: Mapping[str, Any]) -> Gains:
    """Return the three channel gains as floats, or raise if a channel is missing or not a sensible multiplier."""
    if set(values) != set(CHANNELS):
        raise ColorGainError(f"Give exactly the gains {', '.join(CHANNELS)}")
    checked: Gains = {}
    for channel in CHANNELS:
        value = values[channel]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ColorGainError(f"'{channel}' must be a number, got {value!r}")
        number = float(value)
        if not math.isfinite(number) or not 0 < number <= MAX_GAIN:
            raise ColorGainError(f"'{channel}' must be above 0 and at most {MAX_GAIN}, got {value}")
        checked[channel] = number
    return checked


# -- saved gains ------------------------------------------------------------------------------------


_store: CameraStore[Gains] = CameraStore(
    lambda: config.CAMERA_GAINS_FILE, validate_gains, normalize_name, "colour gain"
)
load_saved = _store.load
save_gains = _store.save
forget_gains = _store.forget


# -- applying gains ---------------------------------------------------------------------------------


def apply_gains(image: np.ndarray, gains: Mapping[str, float]) -> np.ndarray:
    """Multiply each channel of a BGR uint8 frame by its gain, rounding and clipping to 0..255."""
    scale = np.array([gains[c] for c in CHANNELS], dtype=np.float32)
    scaled = np.rint(image.astype(np.float32) * scale)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def index_gains(cameras: list[dict[str, Any]], saved: Mapping[str, Gains]) -> dict[int, Gains]:
    """Map each OpenCV index to the saved gain of the camera listed at that index."""
    return index_by_name(cameras, saved, "Colour gain")


def gains_for_attached_cameras() -> dict[int, Gains]:
    """The gains for the cameras plugged in right now. Enumeration is only implemented on macOS."""
    saved = load_saved()
    if not saved:
        return {}
    if platform.system() != "Darwin":
        logger.warning("Colour gains are only applied on macOS; ignoring %d saved gain(s)", len(saved))
        return {}
    return index_gains(list_cameras_in_cv2_order(), saved)


def gains_for_session(cameras: Mapping[str, Mapping[str, Any]]) -> dict[int, Gains]:
    """The gains for this session's cameras. A camera with ``color_correction: false`` is left alone.

    Cameras that omit the flag keep the correction, so a profile saved before the switch existed stays corrected.
    """
    enabled = {
        camera.get("camera_index")
        for camera in cameras.values()
        if camera.get("color_correction", True) is not False
    }
    return {index: gains for index, gains in gains_for_attached_cameras().items() if index in enabled}


# -- hooking LeRobot --------------------------------------------------------------------------------


def install(index_map: Mapping[int, Gains]) -> None:
    """Apply these gains to the frames of the OpenCV cameras at these indices. Replaces any earlier map.

    Registers the ``colour`` stage of ``camera_frames``, which owns the one wrapper around LeRobot's frame
    postprocessing. An empty map switches the correction off.
    """
    camera_frames.register(
        "colour", {int(index): partial(apply_gains, gains=dict(gains)) for index, gains in index_map.items()}
    )


def encode_for_env(index_map: Mapping[int, Gains]) -> str:
    return json.dumps({str(index): gains for index, gains in index_map.items()}, sort_keys=True)


def decode_from_env(text: str) -> dict[int, Gains]:
    data = json.loads(text)
    return {int(index): validate_gains(gains) for index, gains in data.items()}
