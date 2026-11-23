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

from pathlib import Path

import pytest

from lelab.sim import assets
from lelab.sim.cell import CameraPose, PieceSize, SimCell, load_cell, save_cell


def test_default_cell_geometry() -> None:
    cell = SimCell()
    assert cell.half == pytest.approx(4 * cell.square_size)
    a1 = cell.location_xy("a1")
    h1 = cell.location_xy("h1")
    a8 = cell.location_xy("a8")
    # Rank 1 is nearest the arm (smallest x); file a is on the arm's left (+y).
    assert a1[0] == pytest.approx(cell.board_x0 + cell.square_size / 2)
    assert a1[1] > 0 > h1[1]
    assert a8[0] > a1[0]
    assert cell.location_xy("e4") == pytest.approx(
        (cell.board_x0 + 3.5 * cell.square_size, -0.5 * cell.square_size)
    )
    for name in ("bin", "park_white", "park_black"):
        x, y = cell.location_xy(name)
        (x0, y0), (x1, y1) = cell.location_rect(name)
        assert x0 < x < x1 and y0 < y < y1
        inner_y = min(abs(y0), abs(y1)) if y0 * y1 > 0 else 0.0
        assert inner_y > cell.half, f"{name} must sit beside the board"


def test_cell_round_trip(tmp_lerobot_home) -> None:
    assert load_cell() == SimCell()
    cell = SimCell(square_size=0.028, board_x0=0.12)
    save_cell(cell)
    assert load_cell() == cell
    assert SimCell.from_dict(cell.to_dict()) == cell


def test_cell_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        SimCell(square_size=0)
    with pytest.raises(ValueError):
        SimCell.from_dict({"square_size": "big"})
    with pytest.raises(ValueError):
        SimCell().location_xy("z9")


def test_cell_coerces_numeric_strings_to_float() -> None:
    cell = SimCell.from_dict({"square_size": "0.03", "bin_center": ["0.3", -0.17]})
    assert isinstance(cell.square_size, float)
    assert cell.half == pytest.approx(0.12)
    assert cell.bin_center == (0.3, -0.17)
    piece = SimCell.from_dict({"pieces": {"p": {"radius": "0.01", "height": 3}}}).pieces["p"]
    assert (piece.radius, piece.height) == (0.01, 3.0)
    assert isinstance(piece.height, float)


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"square_size": True}, "square_size"),
        ({"board_x0": 0}, "board_x0"),
        ({"board_x0": float("nan")}, "board_x0"),
        ({"board_x0": float("inf")}, "board_x0"),
        ({"board_thickness": -0.1}, "board_thickness"),
        ({"bin_half": (0.04, 0)}, "bin_half"),
        ({"bin_half": (0.04,)}, "bin_half"),
        ({"bin_center": (0.3, "far")}, "bin_center"),
        ({"bin_wall_height": 0}, "bin_wall_height"),
        ({"park_half": -1}, "park_half"),
        ({"park_white_center": (0.1, None)}, "park_white_center"),
    ],
)
def test_cell_rejects_invalid_fields(kwargs: dict, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        SimCell(**kwargs)


def test_piece_size_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="radius"):
        PieceSize(0, 0.03)
    with pytest.raises(ValueError, match="height"):
        PieceSize(0.01, -1)
    with pytest.raises(ValueError, match="radius"):
        PieceSize("wide", 0.03)
    with pytest.raises(ValueError, match="radius"):
        SimCell.from_dict({"pieces": {"p": {"radius": 0, "height": 0.03}}})


def test_camera_pose_validation() -> None:
    good = CameraPose((0.2, 0, 0.5), (0, -1, 0, 1, 0, 0), 45)
    assert good.pos == (0.2, 0.0, 0.5) and isinstance(good.fovy, float)
    with pytest.raises(ValueError, match="pos"):
        CameraPose((0.2, 0), (0, -1, 0, 1, 0, 0), 45)
    with pytest.raises(ValueError, match="xyaxes"):
        CameraPose((0.2, 0, 0.5), (0, -1, 0, 1, 0), 45)
    with pytest.raises(ValueError, match="xyaxes"):
        CameraPose((0.2, 0, 0.5), (0, 0, 0, 1, 0, 0), 45)
    with pytest.raises(ValueError, match="xyaxes"):
        CameraPose((0.2, 0, 0.5), (0, -1, 0, 0, 0, 0), 45)
    for fovy in (0, 180, -5, float("nan")):
        with pytest.raises(ValueError, match="fovy"):
            CameraPose((0.2, 0, 0.5), (0, -1, 0, 1, 0, 0), fovy)
    with pytest.raises(ValueError, match="pos"):
        SimCell.from_dict({"top_camera": {"pos": [0, 0], "xyaxes": [0, -1, 0, 1, 0, 0], "fovy": 45}})
    with pytest.raises(ValueError, match="fovy"):
        SimCell.from_dict({"side_camera": {"pos": [0, 0, 1], "xyaxes": [0, -1, 0, 1, 0, 0], "fovy": 0}})


def test_arm_assets_download_once_at_the_pinned_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["local_dir"]) / assets.ARM_DIR
        target.mkdir(parents=True)
        (target / "so101.xml").write_text("<mujoco/>")
        return kwargs["local_dir"]

    monkeypatch.setattr("lelab.utils.config.SIM_ASSETS_PATH", str(tmp_path / "assets"))
    monkeypatch.setattr(assets, "snapshot_download", fake_download)
    path = assets.arm_model_path()
    assert path.name == "so101.xml" and path.is_file()
    assert calls[0]["repo_id"] == assets.ASSETS_REPO
    assert calls[0]["revision"] == assets.ASSETS_REVISION
    assert calls[0]["allow_patterns"] == [f"{assets.ARM_DIR}/*"]
    assert assets.arm_model_path() == path
    assert len(calls) == 1


def test_arm_assets_raise_sim_unavailable_on_download_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(**kwargs):
        raise OSError("offline")

    monkeypatch.setattr("lelab.utils.config.SIM_ASSETS_PATH", str(tmp_path / "assets"))
    monkeypatch.setattr(assets, "snapshot_download", failing)
    with pytest.raises(assets.SimUnavailable, match="lerobot/so101-tabletop-mujoco"):
        assets.arm_model_path()
