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

"""Pose fit from four board corners, on synthetic projections of the default sim cell."""

from __future__ import annotations

import numpy as np
import pytest

from lelab.chess.camera_pose import CameraPoseError, fit_camera, project
from lelab.sim.cell import SimCell

SIZE = (640, 480)
CELL = SimCell()
Z = CELL.board_thickness
X0, X1, H = CELL.board_x0, CELL.board_x0 + 8 * CELL.square_size, CELL.half
CORNERS = np.array([(X0, H, Z), (X0, -H, Z), (X1, -H, Z), (X1, H, Z)])  # a1, h1, h8, a8


def seen(pose, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return project(pose.pos, pose.xyaxes, pose.fovy, SIZE, CORNERS) + rng.normal(0, noise, (4, 2))


def test_the_oblique_side_camera_is_recovered_with_its_field_of_view() -> None:
    fit = fit_camera(CORNERS, seen(CELL.side_camera), SIZE)
    assert fit.fov_fitted
    assert fit.fovy == pytest.approx(CELL.side_camera.fovy, abs=0.6)
    assert np.linalg.norm(np.array(fit.pos) - CELL.side_camera.pos) < 0.003
    assert fit.rms_px < 0.1


def test_noisy_clicks_still_land_within_the_randomization_width() -> None:
    fit = fit_camera(CORNERS, seen(CELL.side_camera, noise=1.0), SIZE)
    assert fit.fovy == pytest.approx(CELL.side_camera.fovy, abs=3.0)
    assert np.linalg.norm(np.array(fit.pos) - CELL.side_camera.pos) < 0.03


def test_a_camera_looking_straight_down_cannot_tell_its_field_of_view() -> None:
    fit = fit_camera(CORNERS, seen(CELL.top_camera), SIZE, fallback_fovy=39.0)
    assert not fit.fov_fitted
    assert fit.fovy == 39.0


def test_a_given_field_of_view_is_kept_and_the_pose_follows() -> None:
    fit = fit_camera(CORNERS, seen(CELL.top_camera), SIZE, fovy=CELL.top_camera.fovy)
    assert not fit.fov_fitted
    assert fit.fovy == CELL.top_camera.fovy
    assert np.linalg.norm(np.array(fit.pos) - CELL.top_camera.pos) < 0.003


def test_the_fitted_pose_reprojects_onto_the_clicks() -> None:
    clicks = seen(CELL.side_camera, noise=0.5)
    fit = fit_camera(CORNERS, clicks, SIZE)
    again = project(fit.pos, fit.xyaxes, fit.fovy, SIZE, CORNERS)
    assert np.sqrt(np.mean(np.sum((again - clicks) ** 2, axis=1))) == pytest.approx(fit.rms_px)


@pytest.mark.parametrize("points", [np.zeros((3, 2)), np.full((4, 2), np.nan), np.zeros((4, 3))])
def test_unusable_clicks_raise(points: np.ndarray) -> None:
    with pytest.raises(CameraPoseError):
        fit_camera(CORNERS, points, SIZE)
