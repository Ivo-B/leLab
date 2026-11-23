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

"""Where everything sits in the simulated cell, in metres, in the arm base frame.

x points away from the arm, y to its left, z up; the table surface is z = 0.
Rank 1 is nearest the arm and file a is on its left, as White sees the board.
The real cell is built to match this file; M3 writes measured camera poses back.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

import chess

from lelab.chess.game import BIN, PARK_BLACK, PARK_WHITE
from lelab.utils import config


def _number(name: str, value: object, *, positive: bool = False) -> float:
    """`value` as a finite float; bools and non-numbers raise ValueError naming `name`."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got {value!r}")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be a number, got {value!r}") from e
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return number


def _numbers(name: str, value: object, length: int, *, positive: bool = False) -> tuple[float, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Iterable):
        raise ValueError(f"{name} must be a sequence of {length} numbers, got {value!r}")
    items = list(value)
    if len(items) != length:
        raise ValueError(f"{name} must have {length} values, got {len(items)}")
    return tuple(_number(name, item, positive=positive) for item in items)


@dataclass(frozen=True)
class PieceSize:
    radius: float  # collision cylinder radius
    height: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius", _number("radius", self.radius, positive=True))
        object.__setattr__(self, "height", _number("height", self.height, positive=True))


@dataclass(frozen=True)
class CameraPose:
    pos: tuple[float, float, float]
    xyaxes: tuple[float, float, float, float, float, float]
    fovy: float

    def __post_init__(self) -> None:
        pos = _numbers("pos", self.pos, 3)
        xyaxes = _numbers("xyaxes", self.xyaxes, 6)
        x_axis, y_axis = xyaxes[:3], xyaxes[3:]
        if not any(x_axis) or not any(y_axis):
            raise ValueError(f"xyaxes must hold two non-zero 3-vectors, got {xyaxes!r}")
        cross = (
            x_axis[1] * y_axis[2] - x_axis[2] * y_axis[1],
            x_axis[2] * y_axis[0] - x_axis[0] * y_axis[2],
            x_axis[0] * y_axis[1] - x_axis[1] * y_axis[0],
        )
        if not any(cross):
            raise ValueError(f"xyaxes must not be parallel, got {xyaxes!r}")
        fovy = _number("fovy", self.fovy)
        if not 0 < fovy < 180:
            raise ValueError(f"fovy must be in (0, 180) degrees, got {self.fovy!r}")
        object.__setattr__(self, "pos", pos)
        object.__setattr__(self, "xyaxes", xyaxes)
        object.__setattr__(self, "fovy", fovy)


def _default_pieces() -> dict[str, PieceSize]:
    return {
        "p": PieceSize(0.0085, 0.030),
        "n": PieceSize(0.0095, 0.036),
        "b": PieceSize(0.0095, 0.040),
        "r": PieceSize(0.0095, 0.034),
        "q": PieceSize(0.0100, 0.046),
        "k": PieceSize(0.0100, 0.050),
    }


@dataclass(frozen=True)
class SimCell:
    square_size: float = 0.028
    board_x0: float = 0.11  # x of the rank-1 edge
    board_thickness: float = 0.004
    bin_center: tuple[float, float] = (0.30, -0.17)
    bin_half: tuple[float, float] = (0.04, 0.035)
    bin_wall_height: float = 0.03
    park_white_center: tuple[float, float] = (0.17, 0.15)
    park_black_center: tuple[float, float] = (0.23, 0.15)
    park_half: float = 0.016
    pieces: dict[str, PieceSize] = field(default_factory=_default_pieces)
    top_camera: CameraPose = CameraPose((0.22, 0.0, 0.55), (0, -1, 0, 1, 0, 0), 45.0)
    side_camera: CameraPose = CameraPose((0.22, -0.45, 0.30), (1, 0, 0, 0, 0.55, 0.835), 50.0)

    def __post_init__(self) -> None:
        for name in ("square_size", "board_x0", "board_thickness", "bin_wall_height", "park_half"):
            object.__setattr__(self, name, _number(name, getattr(self, name), positive=True))
        object.__setattr__(self, "bin_half", _numbers("bin_half", self.bin_half, 2, positive=True))
        for name in ("bin_center", "park_white_center", "park_black_center"):
            object.__setattr__(self, name, _numbers(name, getattr(self, name), 2))
        if not isinstance(self.pieces, dict) or not all(
            isinstance(p, PieceSize) for p in self.pieces.values()
        ):
            raise ValueError("pieces must map piece letters to PieceSize")

    @property
    def half(self) -> float:
        return 4 * self.square_size

    @property
    def board_center(self) -> tuple[float, float]:
        return (self.board_x0 + self.half, 0.0)

    def location_xy(self, name: str) -> tuple[float, float]:
        if name == BIN:
            return self.bin_center
        if name == PARK_WHITE:
            return self.park_white_center
        if name == PARK_BLACK:
            return self.park_black_center
        square = chess.parse_square(name)  # ValueError for anything else
        f, r = chess.square_file(square), chess.square_rank(square)
        return (self.board_x0 + (r + 0.5) * self.square_size, self.half - (f + 0.5) * self.square_size)

    def location_rect(self, name: str) -> tuple[tuple[float, float], tuple[float, float]]:
        """Axis-aligned (min, max) corners of a location on the table."""
        x, y = self.location_xy(name)
        if name == BIN:
            hx, hy = self.bin_half
        elif name in (PARK_WHITE, PARK_BLACK):
            hx = hy = self.park_half
        else:
            hx = hy = self.square_size / 2
        return ((x - hx, y - hy), (x + hx, y + hy))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SimCell:
        try:
            data = dict(data)
            if "pieces" in data:
                data["pieces"] = {k: PieceSize(**v) for k, v in data["pieces"].items()}
            for key in ("top_camera", "side_camera"):
                if key in data:
                    data[key] = CameraPose(**data[key])
            return cls(**data)
        except (TypeError, KeyError, AttributeError) as e:
            raise ValueError(f"Invalid sim cell: {e}") from e


def load_cell() -> SimCell:
    path = config.SIM_CELL_FILE
    if not os.path.isfile(path):
        return SimCell()
    with open(path) as f:
        return SimCell.from_dict(json.load(f))


def save_cell(cell: SimCell) -> None:
    config._atomic_write_text(config.SIM_CELL_FILE, json.dumps(cell.to_dict(), indent=2))
