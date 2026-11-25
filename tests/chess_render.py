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

"""Synthetic top-camera frames for the chess vision tests.

A 400x400 image with the board filling 40..360 in both axes, a1 at the bottom
left as a camera above the board sees it, a bin and two parking squares to the
right. Pieces are discs: light for white, dark for black.
"""

from __future__ import annotations

import chess
import cv2
import numpy as np

from lelab.chess.calibration import BoardCalibration

SIZE = 400
LIGHT = 170
DARK = 90
BACKGROUND = 120


def calibration(robot: str = "arm") -> BoardCalibration:
    return BoardCalibration(
        robot=robot,
        image_size=(SIZE, SIZE),
        board_corners=[(40, 360), (360, 360), (360, 40), (40, 40)],
        bin=[(365, 300), (395, 300), (395, 360), (365, 360)],
        park_white=[(365, 40), (395, 40), (395, 70), (365, 70)],
        park_black=[(365, 80), (395, 80), (395, 110), (365, 110)],
    )


def _poly(cal: BoardCalibration, name: str) -> np.ndarray:
    return np.round(cal.square_polygon(name)).astype(np.int32)


def empty_board(cal: BoardCalibration) -> np.ndarray:
    frame = np.full((SIZE, SIZE, 3), BACKGROUND, dtype=np.uint8)
    for name in chess.SQUARE_NAMES:
        square = chess.parse_square(name)
        shade = LIGHT if (chess.square_file(square) + chess.square_rank(square)) % 2 else DARK
        cv2.fillPoly(frame, [_poly(cal, name)], (shade, shade, shade))
    for name in ("bin", "park_white", "park_black"):
        cv2.fillPoly(frame, [_poly(cal, name)], (LIGHT, LIGHT, LIGHT))
    return frame


def render(cal: BoardCalibration, occupancy: dict[str, str]) -> np.ndarray:
    """The empty board with a disc on every non-empty location."""
    frame = empty_board(cal)
    for name, value in occupancy.items():
        if value == "empty":
            continue
        colour = (235, 235, 235) if value == "white" else (15, 15, 15)
        cx, cy = cal.square_center(name)
        cv2.circle(frame, (int(round(cx)), int(round(cy))), cal.marker_radius(name) + 6, colour, -1)
    return frame


def contract_cameras(**overrides: dict) -> dict[str, dict]:
    """The three recording cameras at the dataset contract; `overrides` patch one camera by name."""
    cameras = {
        name: {"type": "opencv", "camera_index": i, "width": 640, "height": 480, "fps": 30, "rotation": 0}
        for i, name in enumerate(("top", "wrist", "side"))
    }
    for name, patch in overrides.items():
        cameras[name] = {**cameras[name], **patch}
    return cameras
