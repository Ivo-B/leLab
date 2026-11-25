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
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

import numpy as np

from . import camera_frames
from .utils import config
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


def gains_for_session(cameras: Mapping[str, Mapping[str, Any]], listing: Sequence[Any]) -> dict[int, Gains]:
    """The gain saved for each resolved session camera, by OpenCV index. ``color_correction: false`` skips one.

    ``listing`` is ``camera_identity.attached()``: each camera's ``camera_id`` and the ``key`` its gain is saved under.
    """
    saved = load_saved()
    keys = {camera.camera_id: camera.key for camera in listing}
    gains: dict[int, Gains] = {}
    for camera in cameras.values():
        key = keys.get(str(camera.get("camera_id")))
        if camera.get("color_correction", True) is not False and key in saved:
            gains[int(camera["camera_index"])] = saved[key]
    return gains


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
