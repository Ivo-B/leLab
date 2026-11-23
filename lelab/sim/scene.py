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

"""MJCF for one chess position in the simulated cell.

The arm is not in this file: `world.py` attaches MuJoCo Menagerie's SO-101 at the
`so101_mount` site. Each piece is a free body with one collision cylinder and
visual-only base and head geoms; the collision shape stays a primitive so contact
is fast and stable, whatever the visuals look like.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess

from lelab.chess.game import BIN, PARK_BLACK, PARK_WHITE, Occupancy, piece_colours

from .cell import CameraPose, SimCell

PIECE_MASS = 0.012  # kg; light wooden pieces


@dataclass
class SimPosition:
    board: chess.Board
    park_white: bool = True
    park_black: bool = True
    bin_pieces: tuple[str, ...] = ()

    def piece_at(self, location: str) -> str | None:
        if location == PARK_WHITE:
            return "Q" if self.park_white else None
        if location == PARK_BLACK:
            return "q" if self.park_black else None
        if location == BIN:
            return self.bin_pieces[-1] if self.bin_pieces else None
        piece = self.board.piece_at(chess.parse_square(location))
        return piece.symbol() if piece else None

    def occupancy(self) -> dict[str, Occupancy]:
        occ: dict[str, Occupancy] = dict(piece_colours(self.board))
        occ[PARK_WHITE] = "white" if self.park_white else "empty"
        occ[PARK_BLACK] = "black" if self.park_black else "empty"
        top = self.piece_at(BIN)
        occ[BIN] = "empty" if top is None else ("white" if top.isupper() else "black")
        return occ


def piece_body_name(location: str, symbol: str, index: int = 0) -> str:
    return f"p_{location}_{symbol}" + (f"_{index}" if location == BIN else "")


def _piece(cell: SimCell, name: str, symbol: str, x: float, y: float, z0: float) -> str:
    size = cell.pieces[symbol.lower()]
    r, h = size.radius, size.height
    mat = "white_piece" if symbol.isupper() else "black_piece"
    z = z0 + h / 2 + 0.0005
    return (
        f'<body name="{name}" pos="{x:.5f} {y:.5f} {z:.5f}"><freejoint/>'
        f'<geom class="piece" type="cylinder" size="{r:.5f} {h / 2:.5f}" material="{mat}"/>'
        f'<geom class="piece_visual" type="cylinder" size="{1.45 * r:.5f} {0.06 * h:.5f}" '
        f'pos="0 0 {-0.44 * h:.5f}" material="{mat}"/>'
        f'<geom class="piece_visual" type="sphere" size="{1.1 * r:.5f}" pos="0 0 {h / 2 - r:.5f}" '
        f'material="{mat}"/>'
        "</body>"
    )


def _camera(name: str, pose: CameraPose) -> str:
    pos = " ".join(f"{v:.4f}" for v in pose.pos)
    axes = " ".join(f"{v:.4f}" for v in pose.xyaxes)
    return f'<camera name="{name}" pos="{pos}" xyaxes="{axes}" fovy="{pose.fovy:.2f}"/>'


def scene_xml(cell: SimCell, position: SimPosition) -> str:
    top_z = cell.board_thickness
    bodies: list[str] = []
    for square, piece in position.board.piece_map().items():
        location = chess.square_name(square)
        x, y = cell.location_xy(location)
        bodies.append(_piece(cell, piece_body_name(location, piece.symbol()), piece.symbol(), x, y, top_z))
    for location, symbol, present in (
        (PARK_WHITE, "Q", position.park_white),
        (PARK_BLACK, "q", position.park_black),
    ):
        if present:
            x, y = cell.location_xy(location)
            bodies.append(_piece(cell, piece_body_name(location, symbol), symbol, x, y, 0.002))
    bx, by = cell.bin_center
    for i, symbol in enumerate(position.bin_pieces):
        # Captured pieces lie in a row on the bin floor; they only need to be there, not tidy.
        x = bx - cell.bin_half[0] + 0.012 + (i % 5) * 0.016
        y = by - cell.bin_half[1] + 0.012 + (i // 5) * 0.016
        bodies.append(_piece(cell, piece_body_name(BIN, symbol, i), symbol, x, y, 0.003))

    cx, _ = cell.board_center
    hx, hy = cell.bin_half
    wall = cell.bin_wall_height / 2
    return f"""<mujoco model="garage_chess">
  <statistic center="{cx:.3f} 0 0.1" extent="0.6"/>
  <visual>
    <headlight diffuse="0.4 0.4 0.4" ambient="0.25 0.25 0.25" specular="0 0 0"/>
    <global offwidth="1280" offheight="960"/>
  </visual>
  <asset>
    <texture name="checker" type="2d" builtin="checker" rgb1="0.93 0.86 0.71" rgb2="0.46 0.30 0.18" width="512" height="512"/>
    <material name="board" texture="checker" texrepeat="4 4"/>
    <material name="table" rgba="0.80 0.72 0.60 1"/>
    <material name="white_piece" rgba="0.95 0.93 0.88 1"/>
    <material name="black_piece" rgba="0.08 0.08 0.08 1"/>
    <material name="bin" rgba="0.30 0.30 0.35 1"/>
    <material name="park" rgba="0.55 0.65 0.80 1"/>
  </asset>
  <default>
    <default class="piece">
      <geom mass="{PIECE_MASS}" condim="4" friction="1 0.02 0.002" solref="0.01 1" priority="1"/>
    </default>
    <default class="piece_visual">
      <geom contype="0" conaffinity="0" mass="0" group="1"/>
    </default>
    <default class="bin_wall"><geom type="box" material="bin"/></default>
  </default>
  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <light name="key" pos="0.3 -0.4 0.8" dir="-0.3 0.5 -1" diffuse="0.25 0.25 0.25" castshadow="true"/>
    <geom name="table" type="box" size="0.40 0.40 0.015" pos="0.20 0 -0.015" material="table"/>
    <geom name="board" type="box" size="{cell.half:.5f} {cell.half:.5f} {top_z / 2:.5f}" pos="{cx:.5f} 0 {top_z / 2:.5f}" material="board"/>
    <geom name="park_white_pad" type="box" size="{cell.park_half:.4f} {cell.park_half:.4f} 0.001" pos="{cell.park_white_center[0]:.4f} {cell.park_white_center[1]:.4f} 0.001" material="park"/>
    <geom name="park_black_pad" type="box" size="{cell.park_half:.4f} {cell.park_half:.4f} 0.001" pos="{cell.park_black_center[0]:.4f} {cell.park_black_center[1]:.4f} 0.001" material="park"/>
    <body name="bin" pos="{bx:.4f} {by:.4f} 0">
      <geom class="bin_wall" size="{hx:.4f} {hy:.4f} 0.0015" pos="0 0 0.0015"/>
      <geom class="bin_wall" size="0.002 {hy:.4f} {wall:.4f}" pos="{hx:.4f} 0 {wall:.4f}"/>
      <geom class="bin_wall" size="0.002 {hy:.4f} {wall:.4f}" pos="{-hx:.4f} 0 {wall:.4f}"/>
      <geom class="bin_wall" size="{hx:.4f} 0.002 {wall:.4f}" pos="0 {hy:.4f} {wall:.4f}"/>
      <geom class="bin_wall" size="{hx:.4f} 0.002 {wall:.4f}" pos="0 {-hy:.4f} {wall:.4f}"/>
    </body>
    <site name="so101_mount" pos="0 0 0"/>
    {"".join(bodies)}
    {_camera("top", cell.top_camera)}
    {_camera("side", cell.side_camera)}
  </worldbody>
</mujoco>"""
