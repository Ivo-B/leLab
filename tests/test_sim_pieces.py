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
from pathlib import Path

import chess
import pytest

from lelab.sim import pieces
from lelab.sim.assets import SimUnavailable
from lelab.sim.cell import SimCell
from lelab.sim.scene import SimPosition, scene_xml


def test_visual_uses_a_mesh_and_collision_stays_a_cylinder() -> None:
    from lelab.sim.pieces import piece_visual_xml

    xml = piece_visual_xml("p_e2_P", "P", "pawn.obj", radius=0.0085, height=0.030)
    assert 'mesh="' in xml or "mesh=" in xml
    assert 'type="cylinder"' in xml
    assert "pawn.obj" in xml


def test_collision_cylinder_is_unchanged_and_the_mesh_geom_is_visual_only() -> None:
    xml = pieces.piece_visual_xml("p_e2_P", "P", "pawn.stl", radius=0.0085, height=0.030)
    collision, visual = ET.fromstring(f"<body>{xml}</body>")  # nosec B314
    assert collision.get("class") == "piece" and collision.get("type") == "cylinder"
    assert collision.get("size") == "0.00850 0.01500"
    assert collision.get("pos") is None  # centred on the body, as before
    assert collision.get("group") == "3"  # hidden: the mesh renders instead
    assert visual.get("class") == "piece_visual" and visual.get("type") == "mesh"
    assert visual.get("mesh") == "pawn.stl"


def test_without_a_mesh_the_cylinder_stays_visible() -> None:
    xml = pieces.piece_visual_xml("p_e2_P", "P", None, radius=0.0085, height=0.030)
    (collision,) = ET.fromstring(f"<body>{xml}</body>")  # nosec B314
    assert collision.get("type") == "cylinder" and collision.get("group") is None


@pytest.mark.parametrize("symbol", list("pnbrqkPNBRQK"))
def test_mesh_base_diameter_equals_the_collision_diameter(symbol: str) -> None:
    mesh = pieces.PIECE_MESHES[symbol.lower()]
    radius = 0.0095
    assert pieces.mesh_scale(symbol, radius) * mesh.base_radius == pytest.approx(radius)
    xml = pieces.piece_visual_xml("x", symbol, mesh.file, radius=radius, height=0.04)
    visual = ET.fromstring(f"<body>{xml}</body>")[1]  # nosec B314
    x, y, z = (float(v) for v in visual.get("pos").split())
    s = pieces.mesh_scale(symbol, radius)
    # the base centre (mesh.cx, mesh.cz) lands on the body axis, the base on the cylinder's bottom
    assert (x + s * mesh.cx, y - s * mesh.cz) == pytest.approx((0.0, 0.0), abs=1e-6)
    assert z == pytest.approx(-0.02)


def test_mesh_path_is_case_insensitive_and_rejects_unknown_symbols(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("lelab.utils.config.SIM_ASSETS_PATH", str(tmp_path))
    assert pieces.piece_mesh_path("N") == pieces.piece_mesh_path("n") == tmp_path / "staunton" / "knight.stl"
    assert {pieces.piece_mesh_path(s).name for s in "pnbrqk"} == {
        m.file for m in pieces.PIECE_MESHES.values()
    }
    with pytest.raises(ValueError, match="unknown piece symbol"):
        pieces.piece_mesh_path("x")


def test_scene_declares_meshes_only_when_asked() -> None:
    cell, position = SimCell(), SimPosition(chess.Board())
    plain = ET.fromstring(scene_xml(cell, position))  # nosec B314
    assert not list(plain.iter("mesh"))
    assert not [g for g in plain.iter("geom") if g.get("type") == "mesh"]

    meshed = ET.fromstring(scene_xml(cell, position, meshes=True))  # nosec B314
    assert {m.get("file") for m in meshed.iter("mesh")} == {m.file for m in pieces.PIECE_MESHES.values()}
    visuals = [g for g in meshed.iter("geom") if g.get("type") == "mesh"]
    assert len(visuals) == 34
    assert all(g.get("mesh") in {m.get("name") for m in meshed.iter("mesh")} for g in visuals)


def test_revision_is_pinned_to_a_full_commit() -> None:
    assert len(pieces.PIECES_REVISION) == 40 and set(pieces.PIECES_REVISION) <= set("0123456789abcdef")
    assert pieces.PIECES_REPO == "https://github.com/clarkerubber/Staunton-Pieces"


def test_ensure_downloads_once_then_reuses_the_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_fetch(path: str, sha256: str) -> bytes:
        calls.append(path)
        return b"stl"

    monkeypatch.setattr("lelab.utils.config.SIM_ASSETS_PATH", str(tmp_path / "assets"))
    monkeypatch.setattr(pieces, "_fetch", fake_fetch)
    root = pieces.ensure_piece_meshes()
    assert root == tmp_path / "assets" / "staunton"
    assert all(pieces.piece_mesh_path(s).is_file() for s in "pnbrqk")
    assert (root / "LICENSE").is_file()
    assert sorted(calls) == sorted([m.upstream for m in pieces.PIECE_MESHES.values()] + ["LICENSE"])
    assert not [f for f in root.iterdir() if f.name.endswith(".tmp")]
    assert pieces.ensure_piece_meshes() == root
    assert len(calls) == 7


def test_missing_meshes_raise_sim_unavailable_from_the_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(path: str, sha256: str) -> bytes:
        raise OSError("offline")

    monkeypatch.setattr("lelab.utils.config.SIM_ASSETS_PATH", str(tmp_path / "assets"))
    monkeypatch.setattr(pieces, "_fetch", failing)
    with pytest.raises(SimUnavailable, match="Staunton-Pieces"):
        pieces.ensure_piece_meshes()
    assert not pieces.piece_mesh_path("p").exists()
