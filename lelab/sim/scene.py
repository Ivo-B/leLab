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
`so101_mount` site. Each piece is a free body with one collision cylinder and, when
meshes are on, a visual-only Staunton mesh (`pieces.py`); the collision shape stays a
primitive so contact is fast and stable, whatever the visuals look like.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess

from lelab.chess.game import BIN, PARK_BLACK, PARK_WHITE, Occupancy, piece_colours

from .cell import CameraPose, PieceSize, SimCell
from .pieces import PIECE_MESHES, piece_mesh_assets_xml, piece_visual_xml

PIECE_MASS = 0.012  # kg; light wooden pieces
PIECE_FRICTION = 1.0  # sliding friction coefficient of a piece geom
LIGHT_DIR = (-0.3, 0.5, -1.0)  # direction of the shadow-casting key light
LIGHT_INTENSITY = 1.0  # scale of both lights' diffuse colour
Rgb = tuple[float, float, float]
BOARD_RGB: tuple[Rgb, Rgb] = ((0.46, 0.30, 0.18), (0.78, 0.68, 0.52))  # dark and light squares
WHITE_PIECE_RGB: Rgb = (0.95, 0.93, 0.88)
BLACK_PIECE_RGB: Rgb = (0.08, 0.08, 0.08)
TABLE_RGB: Rgb = (0.80, 0.72, 0.60)
BIN_WALL_HALF_THICKNESS = 0.002
BIN_FLOOR_TOP = 0.003  # the bin floor is a 1.5 mm half-height box resting on the table
BIN_GAP = 0.002  # clear space between captured pieces, and between layers
BIN_MARGIN = 0.001  # clear space between a captured piece and a bin wall
_EPS = 1e-9  # so an exact fit is not pushed to the next row by float noise


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


def _size(cell: SimCell, symbol: str) -> PieceSize:
    try:
        return cell.pieces[symbol.lower()]
    except (KeyError, AttributeError):
        raise ValueError(f"unknown piece symbol {symbol!r}; expected one of {sorted(cell.pieces)}") from None


def _bin_slots(cell: SimCell, symbols: tuple[str, ...]) -> list[tuple[float, float, float]]:
    """Where each captured piece spawns in the bin, as (x, y, z of its base), in `symbols` order.

    Pieces are packed in rows by their own collision diameters on the bin floor, inside the walls'
    inner faces. When the floor is full the next piece starts a layer raised by the tallest
    captured piece plus a gap; it falls onto the pieces below.
    """
    sizes = [_size(cell, s) for s in symbols]
    bx, by = cell.bin_center
    width = 2 * (cell.bin_half[0] - BIN_WALL_HALF_THICKNESS - BIN_MARGIN)
    depth = 2 * (cell.bin_half[1] - BIN_WALL_HALF_THICKNESS - BIN_MARGIN)
    tallest = max((size.height for size in sizes), default=0.0)
    slots: list[tuple[float, float, float]] = []
    layer, cursor, row_y, row_depth = 0, 0.0, 0.0, 0.0
    for symbol, size in zip(symbols, sizes, strict=True):
        d = 2 * size.radius
        if d > width + _EPS or d > depth + _EPS:
            raise ValueError(f"piece {symbol!r} does not fit in the bin")
        if cursor + d > width + _EPS:  # row full: start the next row
            cursor, row_y, row_depth = 0.0, row_y + row_depth + BIN_GAP, 0.0
        if row_y + d > depth + _EPS:  # floor full: start the next layer
            layer, cursor, row_y, row_depth = layer + 1, 0.0, 0.0, 0.0
        x = bx - width / 2 + cursor + d / 2
        y = by - depth / 2 + row_y + d / 2
        slots.append((x, y, BIN_FLOOR_TOP + layer * (tallest + BIN_GAP)))
        cursor += d + BIN_GAP
        row_depth = max(row_depth, d)
    return slots


def _piece(cell: SimCell, name: str, symbol: str, x: float, y: float, z0: float, meshes: bool) -> str:
    size = _size(cell, symbol)
    h = size.height
    z = z0 + h / 2 + 0.0005
    mesh_file = PIECE_MESHES[symbol.lower()].file if meshes else None
    geoms = piece_visual_xml(name, symbol, mesh_file, radius=size.radius, height=h)
    return f'<body name="{name}" pos="{x:.5f} {y:.5f} {z:.5f}"><freejoint/>{geoms}</body>'


def _camera(name: str, pose: CameraPose) -> str:
    pos = " ".join(f"{v:.4f}" for v in pose.pos)
    axes = " ".join(f"{v:.4f}" for v in pose.xyaxes)
    return f'<camera name="{name}" pos="{pos}" xyaxes="{axes}" fovy="{pose.fovy:.2f}"/>'


def _rgb(rgb: Rgb) -> str:
    return " ".join(f"{v:.4f}" for v in rgb)


def _vec(values: tuple[float, ...]) -> str:
    return " ".join(f"{v:.5f}" for v in values)


def scene_xml(
    cell: SimCell,
    position: SimPosition,
    *,
    meshes: bool = False,
    piece_mass: float = PIECE_MASS,
    piece_friction: float = PIECE_FRICTION,
    light_dir: tuple[float, float, float] = LIGHT_DIR,
    light_intensity: float = LIGHT_INTENSITY,
    board_rgb: tuple[Rgb, Rgb] = BOARD_RGB,
    white_piece_rgb: Rgb = WHITE_PIECE_RGB,
    black_piece_rgb: Rgb = BLACK_PIECE_RGB,
    table_rgb: Rgb = TABLE_RGB,
) -> str:
    """The scene's MJCF. With `meshes`, pieces render as Staunton meshes: `compile_scene` must then
    be given the directory from `pieces.ensure_piece_meshes()`. Without, they render as their
    collision cylinders and the XML compiles with no files at all.

    The remaining arguments are what `randomize.py` varies; their defaults are the nominal scene.
    `light_intensity` scales the diffuse colour of both lights."""
    top_z = cell.board_thickness
    bodies: list[str] = []
    for square, piece in position.board.piece_map().items():
        location = chess.square_name(square)
        x, y = cell.location_xy(location)
        bodies.append(
            _piece(cell, piece_body_name(location, piece.symbol()), piece.symbol(), x, y, top_z, meshes)
        )
    for location, symbol, present in (
        (PARK_WHITE, "Q", position.park_white),
        (PARK_BLACK, "q", position.park_black),
    ):
        if present:
            x, y = cell.location_xy(location)
            bodies.append(_piece(cell, piece_body_name(location, symbol), symbol, x, y, 0.002, meshes))
    bx, by = cell.bin_center
    slots = _bin_slots(cell, position.bin_pieces)
    for i, (symbol, (x, y, z0)) in enumerate(zip(position.bin_pieces, slots, strict=True)):
        bodies.append(_piece(cell, piece_body_name(BIN, symbol, i), symbol, x, y, z0, meshes))

    mesh_assets = piece_mesh_assets_xml({k: size.radius for k, size in cell.pieces.items()}) if meshes else ""
    cx, _ = cell.board_center
    hx, hy = cell.bin_half
    wall = cell.bin_wall_height / 2
    sun = 0.45 * light_intensity
    key = 0.25 * light_intensity
    return f"""<mujoco model="garage_chess">
  <statistic center="{cx:.3f} 0 0.1" extent="0.6"/>
  <visual>
    <headlight diffuse="0.1 0.1 0.1" ambient="0.2 0.2 0.2" specular="0 0 0"/>
    <global offwidth="1280" offheight="960"/>
  </visual>
  <asset>
    {mesh_assets}
    <texture name="checker" type="2d" builtin="checker" rgb1="{_rgb(board_rgb[0])}" rgb2="{_rgb(board_rgb[1])}" width="512" height="512"/>
    <material name="board" texture="checker" texrepeat="4 4"/>
    <material name="table" rgba="{_rgb(table_rgb)} 1"/>
    <material name="white_piece" rgba="{_rgb(white_piece_rgb)} 1"/>
    <material name="black_piece" rgba="{_rgb(black_piece_rgb)} 1"/>
    <material name="bin" rgba="0.30 0.30 0.35 1"/>
    <material name="park" rgba="0.55 0.65 0.80 1"/>
  </asset>
  <default>
    <default class="piece">
      <geom mass="{piece_mass:.6f}" condim="4" friction="{piece_friction:.5f} 0.02 0.002" solref="0.01 1" priority="1"/>
    </default>
    <default class="piece_visual">
      <geom contype="0" conaffinity="0" mass="0" group="1"/>
    </default>
    <default class="bin_wall"><geom type="box" material="bin"/></default>
  </default>
  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true" diffuse="{sun:.5f} {sun:.5f} {sun:.5f}"/>
    <light name="key" pos="0.3 -0.4 0.8" dir="{_vec(light_dir)}" diffuse="{key:.5f} {key:.5f} {key:.5f}" castshadow="true"/>
    <geom name="table" type="box" size="0.40 0.40 0.015" pos="0.20 0 -0.015" material="table"/>
    <geom name="board" type="box" size="{cell.half:.5f} {cell.half:.5f} {top_z / 2:.5f}" pos="{cx:.5f} 0 {top_z / 2:.5f}" material="board"/>
    <geom name="park_white_pad" type="box" size="{cell.park_half:.4f} {cell.park_half:.4f} 0.001" pos="{cell.park_white_center[0]:.4f} {cell.park_white_center[1]:.4f} 0.001" material="park"/>
    <geom name="park_black_pad" type="box" size="{cell.park_half:.4f} {cell.park_half:.4f} 0.001" pos="{cell.park_black_center[0]:.4f} {cell.park_black_center[1]:.4f} 0.001" material="park"/>
    <body name="bin" pos="{bx:.4f} {by:.4f} 0">
      <geom class="bin_wall" size="{hx:.4f} {hy:.4f} 0.0015" pos="0 0 0.0015"/>
      <geom class="bin_wall" size="{BIN_WALL_HALF_THICKNESS} {hy:.4f} {wall:.4f}" pos="{hx:.4f} 0 {wall:.4f}"/>
      <geom class="bin_wall" size="{BIN_WALL_HALF_THICKNESS} {hy:.4f} {wall:.4f}" pos="{-hx:.4f} 0 {wall:.4f}"/>
      <geom class="bin_wall" size="{hx:.4f} {BIN_WALL_HALF_THICKNESS} {wall:.4f}" pos="0 {hy:.4f} {wall:.4f}"/>
      <geom class="bin_wall" size="{hx:.4f} {BIN_WALL_HALF_THICKNESS} {wall:.4f}" pos="0 {-hy:.4f} {wall:.4f}"/>
    </body>
    <site name="so101_mount" pos="0 0 0"/>
    {"".join(bodies)}
    {_camera("top", cell.top_camera)}
    {_camera("side", cell.side_camera)}
  </worldbody>
</mujoco>"""
