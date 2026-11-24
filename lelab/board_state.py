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

"""Read the chess position from one `top` frame using the saved calibration and
the `top` camera reference as the empty board."""

from __future__ import annotations

import cv2
import numpy as np

from . import camera_reference
from .chess.calibration import CalibrationError, load_calibration
from .chess.game import EXTRA_LOCATIONS
from .chess.vision import classify_locations

TOP_CAMERA = "top"


class BoardStateError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def decode_png(data: bytes) -> np.ndarray:
    """PNG/JPEG bytes -> RGB uint8 array."""
    array = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(array, cv2.IMREAD_COLOR) if array.size else None
    if bgr is None:
        raise BoardStateError("Body must be a decodable PNG image", 400)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_board_state(png: bytes) -> dict:
    """Occupancy of the 64 squares, the bin and the parking squares in `png`.

    The `top` reference is saved lens- and colour-corrected, so the caller must send
    an equally corrected frame; nothing is corrected here.
    """
    if len(png) > camera_reference.MAX_REFERENCE_BYTES:
        raise BoardStateError("Frame image is too large", 400)
    try:
        calibration = load_calibration()
    except CalibrationError as e:
        raise BoardStateError(str(e), 500) from e
    if calibration is None:
        raise BoardStateError("Calibrate the board on the Setup page first", 409)
    reference_png = camera_reference.load_reference(calibration.robot, TOP_CAMERA)
    if reference_png is None:
        raise BoardStateError(
            f"Capture the '{TOP_CAMERA}' camera reference for robot '{calibration.robot}' first", 409
        )
    frame = decode_png(png)
    reference = decode_png(reference_png)
    occupancy = classify_locations(frame, reference, calibration)
    return {
        "robot": calibration.robot,
        "squares": {k: v for k, v in occupancy.items() if k not in EXTRA_LOCATIONS},
        "extras": {k: occupancy[k] for k in EXTRA_LOCATIONS},
    }
