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

"""Applying a camera fit to the cell, and the route that does it."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from lelab.chess.camera_pose import project
from lelab.server import app
from lelab.sim.camera_fit import CameraFitError, handle_camera_pose, world_corners
from lelab.sim.cell import CameraPose, SimCell, load_cell, save_cell

SIZE = (640, 480)


def side_clicks(cell: SimCell) -> list[list[float]]:
    pose = cell.side_camera
    return project(pose.pos, pose.xyaxes, pose.fovy, SIZE, world_corners(cell)).tolist()


def body(cell: SimCell, **extra) -> dict:
    return {
        "camera": "side",
        "image_size": list(SIZE),
        "board_corners": side_clicks(cell),
        "fovy": None,
        **extra,
    }


def test_world_corners_are_a1_h1_h8_a8_on_the_board_surface() -> None:
    cell = SimCell()
    corners = world_corners(cell)
    assert corners.shape == (4, 3)
    assert corners[0][:2] == pytest.approx((cell.board_x0, cell.half))  # a1: rank 1 edge, file a on the left
    assert corners[2][:2] == pytest.approx((cell.board_x0 + 8 * cell.square_size, -cell.half))
    assert set(corners[:, 2]) == {cell.board_thickness}


def test_fit_without_apply_leaves_the_cell_alone(tmp_lerobot_home) -> None:
    out = handle_camera_pose(body(SimCell()))
    assert out["applied"] is False
    assert out["rms_px"] < 0.5
    assert load_cell() == SimCell()


def test_apply_writes_the_fitted_pose_into_the_cell(tmp_lerobot_home) -> None:
    moved_pose = CameraPose((0.12, -0.30, 0.32), (0.9, -0.42, 0, 0.3, 0.64, 0.71), 47.0)
    moved = SimCell(side_camera=moved_pose)
    save_cell(SimCell())
    out = handle_camera_pose(body(moved, apply=True))
    assert out["applied"] is True
    written = load_cell().side_camera
    assert np.linalg.norm(np.array(written.pos) - moved.side_camera.pos) < 0.004
    assert written.fovy == pytest.approx(47.0, abs=0.6)
    assert load_cell().top_camera == SimCell().top_camera  # the other camera is untouched


def test_a_flat_field_of_view_keeps_the_cell_value_and_warns(tmp_lerobot_home) -> None:
    cell = SimCell()
    pose = cell.top_camera
    clicks = project(pose.pos, pose.xyaxes, pose.fovy, SIZE, world_corners(cell)).tolist()
    out = handle_camera_pose(
        {"camera": "top", "image_size": list(SIZE), "board_corners": clicks, "fovy": None}
    )
    assert out["fov_fitted"] is False
    assert out["fovy"] == cell.top_camera.fovy
    assert any("field of view" in w for w in out["warnings"])


@pytest.mark.parametrize(
    "patch",
    [
        {"camera": "wrist"},
        {"image_size": [1920, 1080]},
        {"board_corners": [[0, 0]] * 3},
        {"image_size": [0, 480]},
    ],
)
def test_bad_requests_are_refused(tmp_lerobot_home, patch: dict) -> None:
    with pytest.raises(CameraFitError):
        handle_camera_pose({**body(SimCell()), **patch})


def test_route_returns_400_with_the_reason(tmp_lerobot_home) -> None:
    response = TestClient(app).post("/chess/camera-pose", json={**body(SimCell()), "camera": "wrist"})
    assert response.status_code == 400
    assert "camera" in response.json()["message"]
