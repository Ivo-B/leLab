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

"""Where the board, the bin and the parking squares are in the `top` frame.

The operator clicks the four board corners (a1, h1, h8, a8) and the four corners
of the bin and of each parking square on the saved `top` camera reference. A
homography from board units to pixels yields the 64 square polygons; the other
three polygons are stored as clicked. Pixel coordinates refer to `image_size`;
`scaled_to()` adapts them to a frame of another size.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import chess
import cv2
import numpy as np

from lelab.utils import config

from .game import BIN, EXTRA_LOCATIONS, PARK_BLACK, PARK_WHITE

Point = tuple[float, float]

BOARD_CORNER_ORDER = ("a1", "h1", "h8", "a8")
LOCATION_NAMES: tuple[str, ...] = tuple(chess.SQUARE_NAMES) + EXTRA_LOCATIONS
MARKER_DIAMETER_FRACTION = 0.4
MIN_QUAD_AREA = 1.0  # pixel^2

_UNIT_CORNERS = np.array([(0, 0), (8, 0), (8, 8), (0, 8)], dtype=np.float32)


class CalibrationError(ValueError):
    """The calibration payload is unusable."""


def _quad(points: object, name: str) -> list[Point]:
    """Validate four corners: finite, strictly convex, non-degenerate (either winding)."""
    if not isinstance(points, list | tuple) or len(points) != 4:
        raise CalibrationError(f"{name} needs exactly four corner points")
    out: list[Point] = []
    for point in points:
        if not isinstance(point, list | tuple) or len(point) != 2:
            raise CalibrationError(f"{name} corners must be [x, y] pairs")
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError) as e:
            raise CalibrationError(f"{name} corners must be numeric") from e
        if not (math.isfinite(x) and math.isfinite(y)):
            raise CalibrationError(f"{name} corners must be finite numbers")
        out.append((x, y))
    crosses = []
    for i in range(4):
        (x0, y0), (x1, y1), (x2, y2) = out[i], out[(i + 1) % 4], out[(i + 2) % 4]
        crosses.append((x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1))
    if not (all(c > 0 for c in crosses) or all(c < 0 for c in crosses)):
        raise CalibrationError(f"{name} corners must form a convex, non-degenerate quadrilateral")
    area = abs(sum(out[i][0] * out[(i + 1) % 4][1] - out[(i + 1) % 4][0] * out[i][1] for i in range(4))) / 2
    if area < MIN_QUAD_AREA:
        raise CalibrationError(f"{name} quadrilateral is too small")
    return out


@dataclass
class BoardCalibration:
    robot: str
    image_size: tuple[int, int]
    board_corners: list[Point]
    bin: list[Point]
    park_white: list[Point]
    park_black: list[Point]
    pixel_diff_threshold: int = 40
    occupied_fraction: float = 0.15
    colour_threshold: float = 128.0

    def __post_init__(self) -> None:
        if not config.is_valid_robot_name(self.robot):
            raise CalibrationError("Invalid robot name")
        try:
            width, height = (int(v) for v in self.image_size)
        except (TypeError, ValueError) as e:
            raise CalibrationError("image_size must be [width, height]") from e
        if width <= 0 or height <= 0:
            raise CalibrationError("image_size must be positive")
        self.image_size = (width, height)
        try:
            self.pixel_diff_threshold = int(self.pixel_diff_threshold)
            self.occupied_fraction = float(self.occupied_fraction)
            self.colour_threshold = float(self.colour_threshold)
        except (TypeError, ValueError) as e:
            raise CalibrationError("thresholds must be numeric") from e
        if self.pixel_diff_threshold < 0:
            raise CalibrationError("pixel_diff_threshold must be >= 0")
        if not 0 < self.occupied_fraction <= 1:
            raise CalibrationError("occupied_fraction must be in (0, 1]")
        if not 0 <= self.colour_threshold <= 255:
            raise CalibrationError("colour_threshold must be in [0, 255]")
        self.board_corners = _quad(self.board_corners, "board_corners")
        self.bin = _quad(self.bin, BIN)
        self.park_white = _quad(self.park_white, PARK_WHITE)
        self.park_black = _quad(self.park_black, PARK_BLACK)
        self._homography = cv2.getPerspectiveTransform(
            _UNIT_CORNERS, np.array(self.board_corners, dtype=np.float32)
        )

    def square_polygon(self, name: str) -> np.ndarray:
        """Four pixel corners of a square or extra location, shape (4, 2), float32."""
        if name == BIN:
            return np.array(self.bin, dtype=np.float32)
        if name == PARK_WHITE:
            return np.array(self.park_white, dtype=np.float32)
        if name == PARK_BLACK:
            return np.array(self.park_black, dtype=np.float32)
        square = chess.parse_square(name)  # ValueError for anything else
        f, r = chess.square_file(square), chess.square_rank(square)
        unit = np.array([(f, r), (f + 1, r), (f + 1, r + 1), (f, r + 1)], dtype=np.float32)
        return cv2.perspectiveTransform(unit.reshape(-1, 1, 2), self._homography).reshape(4, 2)

    def polygons(self) -> dict[str, np.ndarray]:
        return {name: self.square_polygon(name) for name in LOCATION_NAMES}

    def square_center(self, name: str) -> Point:
        centre = self.square_polygon(name).mean(axis=0)
        return (float(centre[0]), float(centre[1]))

    def marker_radius(self, name: str) -> int:
        """Radius of a goal disc: diameter is 40 % of the polygon's shortest side."""
        poly = self.square_polygon(name)
        sides = [float(np.linalg.norm(poly[i] - poly[(i + 1) % 4])) for i in range(4)]
        return max(1, int(round(MARKER_DIAMETER_FRACTION * min(sides) / 2)))

    def scaled_to(self, width: int, height: int) -> BoardCalibration:
        if (width, height) == self.image_size:
            return self
        sx, sy = width / self.image_size[0], height / self.image_size[1]

        def scale(points: list[Point]) -> list[Point]:
            return [(x * sx, y * sy) for x, y in points]

        return BoardCalibration(
            robot=self.robot,
            image_size=(width, height),
            board_corners=scale(self.board_corners),
            bin=scale(self.bin),
            park_white=scale(self.park_white),
            park_black=scale(self.park_black),
            pixel_diff_threshold=self.pixel_diff_threshold,
            occupied_fraction=self.occupied_fraction,
            colour_threshold=self.colour_threshold,
        )

    def to_dict(self) -> dict:
        return {
            "robot": self.robot,
            "image_size": list(self.image_size),
            "board_corners": [list(p) for p in self.board_corners],
            "bin": [list(p) for p in self.bin],
            "park_white": [list(p) for p in self.park_white],
            "park_black": [list(p) for p in self.park_black],
            "pixel_diff_threshold": self.pixel_diff_threshold,
            "occupied_fraction": self.occupied_fraction,
            "colour_threshold": self.colour_threshold,
        }

    @classmethod
    def from_dict(cls, data: object) -> BoardCalibration:
        if not isinstance(data, dict):
            raise CalibrationError("Calibration must be a JSON object")
        try:
            return cls(
                robot=data["robot"],
                image_size=data["image_size"],
                board_corners=data["board_corners"],
                bin=data["bin"],
                park_white=data["park_white"],
                park_black=data["park_black"],
                pixel_diff_threshold=int(data.get("pixel_diff_threshold", 40)),
                occupied_fraction=float(data.get("occupied_fraction", 0.15)),
                colour_threshold=float(data.get("colour_threshold", 128.0)),
            )
        except CalibrationError:
            raise
        except (KeyError, TypeError, ValueError) as e:
            raise CalibrationError(f"Invalid calibration: {e}") from e


def load_calibration() -> BoardCalibration | None:
    path = config.BOARD_CALIBRATION_FILE
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise CalibrationError(f"Calibration file is not valid JSON: {e}") from e
    return BoardCalibration.from_dict(data)


def save_calibration(cal: BoardCalibration) -> None:
    config._atomic_write_text(config.BOARD_CALIBRATION_FILE, json.dumps(cal.to_dict(), indent=2))
