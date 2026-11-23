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

import json

import cv2
import numpy as np
import pytest

from lelab.chess.calibration import (
    BOARD_CORNER_ORDER,
    LOCATION_NAMES,
    BoardCalibration,
    CalibrationError,
    load_calibration,
    save_calibration,
)
from tests.chess_render import calibration


def test_axis_aligned_board_maps_squares_to_40px_cells() -> None:
    cal = calibration()
    assert BOARD_CORNER_ORDER == ("a1", "h1", "h8", "a8")
    a1 = cal.square_polygon("a1")
    np.testing.assert_allclose(a1, [[40, 360], [80, 360], [80, 320], [40, 320]], atol=1e-3)
    h8 = cal.square_polygon("h8")
    np.testing.assert_allclose(h8, [[320, 80], [360, 80], [360, 40], [320, 40]], atol=1e-3)
    assert cal.square_center("e4") == pytest.approx((220.0, 220.0))
    assert cal.marker_radius("e4") == 8
    assert cal.square_center("bin") == pytest.approx((380.0, 330.0))


def test_every_centre_lies_in_exactly_its_own_polygon() -> None:
    cal = calibration()
    polygons = cal.polygons()
    assert set(polygons) == set(LOCATION_NAMES) and len(polygons) == 67
    for name in LOCATION_NAMES:
        cx, cy = cal.square_center(name)
        inside = [
            other
            for other, poly in polygons.items()
            if cv2.pointPolygonTest(poly.astype(np.float32), (cx, cy), False) >= 0
        ]
        assert inside == [name]


def test_perspective_corners_still_give_a_grid() -> None:
    cal = BoardCalibration(
        robot="arm",
        image_size=(640, 480),
        board_corners=[(120, 440), (520, 440), (440, 80), (200, 80)],
        bin=[(560, 300), (620, 300), (620, 400), (560, 400)],
        park_white=[(560, 60), (620, 60), (620, 120), (560, 120)],
        park_black=[(560, 140), (620, 140), (620, 200), (560, 200)],
    )
    e4 = cal.square_center("e4")
    d5 = cal.square_center("d5")
    assert e4[1] > d5[1]  # rank 4 is closer to the camera-bottom than rank 5
    assert e4[0] > d5[0]  # e is to the right of d


def test_scaled_to_rescales_every_point_and_keeps_thresholds() -> None:
    cal = calibration()
    scaled = cal.scaled_to(800, 400)
    assert scaled.image_size == (800, 400)
    assert scaled.square_center("e4") == pytest.approx((440.0, 220.0))
    assert scaled.pixel_diff_threshold == cal.pixel_diff_threshold
    assert cal.scaled_to(400, 400) is cal


def test_rejects_bad_shapes_and_names() -> None:
    good = calibration().to_dict()
    with pytest.raises(CalibrationError):
        BoardCalibration.from_dict({**good, "board_corners": good["board_corners"][:3]})
    with pytest.raises(CalibrationError):
        BoardCalibration.from_dict({**good, "robot": "../etc"})
    with pytest.raises(CalibrationError):
        BoardCalibration.from_dict({**good, "image_size": [0, 400]})
    with pytest.raises(CalibrationError):
        BoardCalibration.from_dict({k: v for k, v in good.items() if k != "bin"})
    with pytest.raises(ValueError):
        calibration().square_polygon("z9")


def test_round_trips_through_json_and_disk(tmp_lerobot_home) -> None:
    assert load_calibration() is None
    cal = calibration()
    save_calibration(cal)
    loaded = load_calibration()
    assert loaded is not None
    assert loaded.to_dict() == cal.to_dict()
    assert json.loads((tmp_lerobot_home / "garage" / "board_calibration.json").read_text())["robot"] == "arm"
    assert not list((tmp_lerobot_home / "garage").glob("*.tmp"))
