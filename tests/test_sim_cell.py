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
from lelab.sim.cell import SimCell, load_cell, save_cell


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
        assert abs(y) > cell.half, f"{name} must sit beside the board"
        (x0, y0), (x1, y1) = cell.location_rect(name)
        assert x0 < x < x1 and y0 < y < y1


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
