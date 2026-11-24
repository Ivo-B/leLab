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

"""Domain randomization for the simulated cell, drawn from one seed.

`randomize_cell` perturbs the two fixed cameras (what a re-seated or re-aimed camera does);
`randomize_variation` draws everything else the scene and arm can vary: piece mass and friction,
lighting, colours, actuator gain and the arm's starting pose. One `random.Random` feeds both, in
that order, so a seed reproduces the whole scene. The expert gate never calls them.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, replace
from typing import Any

from .cell import CameraPose, SimCell
from .scene import (
    BLACK_PIECE_RGB,
    BOARD_RGB,
    LIGHT_DIR,
    LIGHT_INTENSITY,
    PIECE_FRICTION,
    PIECE_MASS,
    TABLE_RGB,
    WHITE_PIECE_RGB,
    Rgb,
)

CAMERA_POS_M = 0.01  # camera position, each axis
CAMERA_YAW_DEG = 2.0  # camera angle, about the world's vertical axis
CAMERA_FOVY_DEG = 2.0
MASS_FRACTION = 0.30
FRICTION_FRACTION = 0.30
GAIN_FRACTION = 0.15  # actuator position gain
JOINT_DEG = 2.0  # initial joint angles
LIGHT_DIR_SPREAD = 0.15  # each component of the key light's direction
LIGHT_INTENSITY_FRACTION = 0.30
BOARD_COLOUR_SPREAD = 0.15
TABLE_COLOUR_SPREAD = 0.15
WHITE_PIECE_COLOUR_SPREAD = 0.08
BLACK_PIECE_COLOUR_SPREAD = 0.40  # of a near-black, so still dark


def _around(rng: random.Random, value: float, amount: float) -> float:
    return value + rng.uniform(-amount, amount)


def _scaled(rng: random.Random, value: float, fraction: float) -> float:
    return value * rng.uniform(1 - fraction, 1 + fraction)


def _yaw(vector: tuple[float, float, float], angle: float) -> tuple[float, float, float]:
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = vector
    return (c * x - s * y, s * x + c * y, z)


def _camera(pose: CameraPose, rng: random.Random) -> CameraPose:
    pos = tuple(_around(rng, v, CAMERA_POS_M) for v in pose.pos)
    angle = math.radians(rng.uniform(-CAMERA_YAW_DEG, CAMERA_YAW_DEG))
    x_axis, y_axis = _yaw(pose.xyaxes[:3], angle), _yaw(pose.xyaxes[3:], angle)  # type: ignore[arg-type]
    fovy = _around(rng, pose.fovy, CAMERA_FOVY_DEG)
    return CameraPose(pos, (*x_axis, *y_axis), fovy)  # type: ignore[arg-type]


def randomize_cell(cell: SimCell, rng: random.Random) -> SimCell:
    """`cell` with both cameras moved by up to 1 cm and 2 degrees of yaw, and their field of view
    by up to 2 degrees. `cell` itself is frozen and left as it was."""
    return replace(cell, top_camera=_camera(cell.top_camera, rng), side_camera=_camera(cell.side_camera, rng))


def randomize_joints(joints, rng: random.Random) -> tuple[float, ...]:
    """`joints` with each value off by up to 2 degrees (the gripper's 0-100 scale too)."""
    return tuple(float(_around(rng, float(j), JOINT_DEG)) for j in joints)


def _tint(rgb: Rgb, rng: random.Random, spread: float) -> Rgb:
    """`rgb` under one brightness factor and a small per-channel one, clipped to [0, 1]."""
    brightness = rng.uniform(1 - spread, 1 + spread)
    r, g, b = (min(1.0, max(0.0, c * brightness * rng.uniform(1 - spread / 3, 1 + spread / 3))) for c in rgb)
    return (r, g, b)


@dataclass(frozen=True)
class SimVariation:
    """What `ChessWorld` varies besides the cameras. The defaults are the nominal scene."""

    piece_mass: float = PIECE_MASS
    piece_friction: float = PIECE_FRICTION
    light_dir: tuple[float, float, float] = LIGHT_DIR
    light_intensity: float = LIGHT_INTENSITY
    board_rgb: tuple[Rgb, Rgb] = BOARD_RGB
    white_piece_rgb: Rgb = WHITE_PIECE_RGB
    black_piece_rgb: Rgb = BLACK_PIECE_RGB
    table_rgb: Rgb = TABLE_RGB
    actuator_gain: float = 1.0  # multiplies every arm actuator's position gain
    initial_joints: tuple[float, ...] | None = None  # None: the world's own start pose

    def scene_kwargs(self) -> dict[str, Any]:
        """The arguments `scene_xml` takes for this variation."""
        return {
            "piece_mass": self.piece_mass,
            "piece_friction": self.piece_friction,
            "light_dir": self.light_dir,
            "light_intensity": self.light_intensity,
            "board_rgb": self.board_rgb,
            "white_piece_rgb": self.white_piece_rgb,
            "black_piece_rgb": self.black_piece_rgb,
            "table_rgb": self.table_rgb,
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SimVariation:
        def rgb(v) -> Rgb:
            return tuple(float(c) for c in v)  # type: ignore[return-value]

        data = dict(data)
        for key in ("white_piece_rgb", "black_piece_rgb", "table_rgb", "light_dir"):
            if key in data:
                data[key] = rgb(data[key])
        if "board_rgb" in data:
            data["board_rgb"] = tuple(rgb(v) for v in data["board_rgb"])
        if data.get("initial_joints") is not None:
            data["initial_joints"] = tuple(float(j) for j in data["initial_joints"])
        return cls(**data)


def randomize_variation(rng: random.Random, joints) -> SimVariation:
    """A variation drawn from `rng`: `joints` is the nominal starting pose to perturb."""
    return SimVariation(
        piece_mass=_scaled(rng, PIECE_MASS, MASS_FRACTION),
        piece_friction=_scaled(rng, PIECE_FRICTION, FRICTION_FRACTION),
        light_dir=tuple(_around(rng, v, LIGHT_DIR_SPREAD) for v in LIGHT_DIR),  # type: ignore[arg-type]
        light_intensity=_scaled(rng, LIGHT_INTENSITY, LIGHT_INTENSITY_FRACTION),
        board_rgb=tuple(_tint(c, rng, BOARD_COLOUR_SPREAD) for c in BOARD_RGB),  # type: ignore[arg-type]
        white_piece_rgb=_tint(WHITE_PIECE_RGB, rng, WHITE_PIECE_COLOUR_SPREAD),
        black_piece_rgb=_tint(BLACK_PIECE_RGB, rng, BLACK_PIECE_COLOUR_SPREAD),
        table_rgb=_tint(TABLE_RGB, rng, TABLE_COLOUR_SPREAD),
        actuator_gain=_scaled(rng, 1.0, GAIN_FRACTION),
        initial_joints=randomize_joints(joints, rng),
    )
