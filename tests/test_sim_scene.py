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

import xml.etree.ElementTree as ET  # nosec B405 - parsing our own generated XML

import chess
import pytest

from lelab.sim.cell import SimCell
from lelab.sim.scene import BIN_WALL_HALF_THICKNESS, SimPosition, piece_body_name, scene_xml

from .sim_helpers import requires_mujoco


def bodies(xml: str) -> dict[str, ET.Element]:
    root = ET.fromstring(xml)  # nosec B314
    return {b.get("name"): b for b in root.iter("body")}


def test_starting_position_has_32_free_pieces_and_two_parked_queens() -> None:
    xml = scene_xml(SimCell(), SimPosition(chess.Board()))
    pieces = {name: b for name, b in bodies(xml).items() if name.startswith("p_")}
    assert len(pieces) == 34
    assert piece_body_name("e2", "P") in pieces
    assert piece_body_name("park_white", "Q") in pieces
    assert piece_body_name("park_black", "q") in pieces
    assert all(b.find("freejoint") is not None for b in pieces.values())


def test_piece_positions_follow_the_cell() -> None:
    cell = SimCell()
    xml = scene_xml(cell, SimPosition(chess.Board(), park_white=False, park_black=False))
    e2 = bodies(xml)[piece_body_name("e2", "P")]
    x, y, z = (float(v) for v in e2.get("pos").split())
    ex, ey = cell.location_xy("e2")
    assert abs(x - ex) < 1e-4 and abs(y - ey) < 1e-4
    assert z > cell.board_thickness
    assert piece_body_name("park_white", "Q") not in bodies(xml)


def test_bin_contents_cameras_and_mount() -> None:
    position = SimPosition(chess.Board(chess.STARTING_FEN), bin_pieces=("p", "N"))
    xml = scene_xml(SimCell(), position)
    root = ET.fromstring(xml)  # nosec B314
    names = bodies(xml)
    assert piece_body_name("bin", "p", 0) in names and piece_body_name("bin", "N", 1) in names
    assert {c.get("name") for c in root.iter("camera")} == {"top", "side"}
    assert any(s.get("name") == "so101_mount" for s in root.iter("site"))
    assert root.find(".//material[@name='board']").get("texrepeat") == "4 4"


def test_occupancy_includes_extras() -> None:
    position = SimPosition(chess.Board(), park_white=True, park_black=False, bin_pieces=("p",))
    occ = position.occupancy()
    assert occ["e2"] == "white" and occ["e7"] == "black" and occ["e4"] == "empty"
    assert occ["park_white"] == "white" and occ["park_black"] == "empty" and occ["bin"] == "black"
    assert position.piece_at("park_white") == "Q"
    assert position.piece_at("e4") is None


def test_unknown_bin_symbol_raises_value_error_naming_it() -> None:
    with pytest.raises(ValueError, match="x"):
        scene_xml(SimCell(), SimPosition(chess.Board(), bin_pieces=("p", "x")))


MIXED = "pnbrqPNBRQ"


def _mixed(count: int) -> tuple[str, ...]:
    return tuple(MIXED[i % len(MIXED)] for i in range(count))


@requires_mujoco
@pytest.mark.parametrize(
    "bin_pieces", [(), _mixed(10), _mixed(30), ("p",) * 10], ids=["none", "10", "30", "10-pawns"]
)
def test_scene_compiles_without_interpenetration_and_bin_pieces_stay_inside(
    bin_pieces: tuple[str, ...],
) -> None:
    import mujoco

    cell = SimCell()
    xml = scene_xml(cell, SimPosition(chess.Board(), bin_pieces=bin_pieces))
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert all(data.contact.dist >= -0.0005)

    bx, by = cell.bin_center
    ix = cell.bin_half[0] - BIN_WALL_HALF_THICKNESS
    iy = cell.bin_half[1] - BIN_WALL_HALF_THICKNESS
    for i, symbol in enumerate(bin_pieces):
        body = model.body(piece_body_name("bin", symbol, i))
        x, y, _ = data.xpos[body.id]
        r = cell.pieces[symbol.lower()].radius
        assert abs(x - bx) + r < ix and abs(y - by) + r < iy


@requires_mujoco
def test_ten_typical_captures_share_the_bin_floor() -> None:
    import mujoco

    cell = SimCell()
    symbols = ("p", "p", "p", "p", "P", "P", "P", "n", "B", "r")
    model = mujoco.MjModel.from_xml_string(scene_xml(cell, SimPosition(chess.Board(), bin_pieces=symbols)))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    floors = {
        round(
            data.xpos[model.body(piece_body_name("bin", s, i)).id][2] - cell.pieces[s.lower()].height / 2, 4
        )
        for i, s in enumerate(symbols)
    }
    assert len(floors) == 1
