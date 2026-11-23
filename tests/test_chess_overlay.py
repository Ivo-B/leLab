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

import cv2
import numpy as np
import pytest

from lelab.chess.overlay import SOURCE_RGB, TARGET_RGB, GoalOverlayStep, draw_goal_markers
from tests.chess_render import calibration, empty_board


def _pixels_matching(frame: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    ys, xs = np.where(np.all(frame == np.array(rgb, dtype=np.uint8), axis=-1))
    return np.stack([xs, ys], axis=1)


def _all_inside(points: np.ndarray, polygon: np.ndarray) -> bool:
    poly = polygon.astype(np.float32)
    return all(cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for x, y in points)


def test_markers_land_inside_their_squares_and_nowhere_else() -> None:
    cal = calibration()
    frame = empty_board(cal)
    before = frame.copy()
    out = draw_goal_markers(frame, "e2", "bin", cal)
    assert np.array_equal(frame, before), "input frame must not be modified"
    red = _pixels_matching(out, SOURCE_RGB)
    green = _pixels_matching(out, TARGET_RGB)
    assert 150 < len(red) < 260  # disc of radius 8 is ~200 px
    assert len(green) > 50
    assert _all_inside(red, cal.square_polygon("e2"))
    assert _all_inside(green, cal.square_polygon("bin"))
    assert len(_pixels_matching(empty_board(cal), SOURCE_RGB)) == 0


def test_step_overlays_only_the_top_key_when_a_goal_is_set() -> None:
    cal = calibration()
    step = GoalOverlayStep(calibration=cal)
    frame = empty_board(cal)
    obs = {"top": frame, "wrist": frame.copy(), "shoulder_pan.pos": 1.0}

    untouched = step.observation(dict(obs))
    assert untouched["top"] is frame
    assert step.goal is None

    step.set_goal("g1", "f3")
    assert step.goal == ("g1", "f3")
    out = step.observation(dict(obs))
    assert len(_pixels_matching(out["top"], SOURCE_RGB)) > 0
    assert np.array_equal(out["wrist"], frame)
    assert out["shoulder_pan.pos"] == 1.0
    assert step.get_config() == {"camera_key": "top", "source": "g1", "target": "f3"}

    step.clear_goal()
    assert step.observation(dict(obs))["top"] is frame


def test_step_ignores_observations_without_the_camera_and_rejects_bad_squares() -> None:
    step = GoalOverlayStep(calibration=calibration())
    step.set_goal("a1", "h8")
    obs = {"wrist": np.zeros((4, 4, 3), np.uint8)}
    assert step.observation(obs) is obs
    assert not obs["wrist"].any()
    with pytest.raises(ValueError):
        step.set_goal("a1", "z9")
    assert step.goal == ("a1", "h8"), "a rejected goal must not change the current one"
    features = {"observation": {}}
    assert step.transform_features(features) is features
