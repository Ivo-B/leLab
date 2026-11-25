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
"""The lens window's helpers; the window itself is checked by hand."""

from __future__ import annotations

import numpy as np

from lelab import color_calibration as cc, lens
from lelab.scripts import camera_lens_gui as gui

TWINS = [
    {"index": 0, "name": "USB Camera", "unique_id": "0x11300000abd8050"},
    {"index": 1, "name": "USB Camera", "unique_id": "0x11410000abd8050"},
]
TOP_LENS = {"fov": 135.0, "bend": 0.0, "zoom": 1.2}
SIDE_LENS = {"fov": 120.0, "bend": 0.01, "zoom": 1.1}


def test_the_grid_has_evenly_spaced_lines_without_the_edges() -> None:
    lines = gui.grid_lines(800, 400, divisions=4)
    assert (200.0, 0.0, 200.0, 400.0) in lines and (0.0, 300.0, 800.0, 300.0) in lines
    assert len(lines) == 6
    assert all(0 < x0 < 800 for x0, y0, x1, y1 in lines if x0 == x1)


def test_slider_values_become_a_checked_lens_or_a_message() -> None:
    values, message = gui.lens_or_message({"fov": 120.0, "bend": 0.0, "zoom": 1.0})
    assert values == {"fov": 120.0, "bend": 0.0, "zoom": 1.0} and message is None
    values, message = gui.lens_or_message({"fov": 200.0, "bend": -0.15, "zoom": 1.0})
    assert values is None and message and "folds" in message


def test_every_slider_range_is_inside_what_a_lens_accepts() -> None:
    for name, (low, high) in gui.SLIDER_RANGES.items():
        accepted_low, accepted_high = {
            "fov": lens.FOV_RANGE,
            "bend": lens.BEND_RANGE,
            "zoom": lens.ZOOM_RANGE,
        }[name]
        assert accepted_low <= low < high <= accepted_high
    assert lens.validate_lens(gui.DEFAULTS) == gui.DEFAULTS


def test_the_display_picture_is_straightened_at_window_size() -> None:
    frame = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    plain = gui.straighten_for_display(frame, (800, 450), None)
    assert plain.shape == (450, 800, 3)
    straightened = gui.straighten_for_display(frame, (800, 450), {"fov": 120.0, "bend": 0.0, "zoom": 0.5})
    assert straightened.shape == (450, 800, 3) and straightened[0, 0].max() == 0


def test_each_twin_keeps_its_own_saved_lens_and_forgets_only_its_own(monkeypatch) -> None:
    monkeypatch.setattr(lens, "list_cameras_cached", lambda: TWINS)
    top, side = cc.list_calibration_cameras(TWINS)
    gui.save_lens_for(top, TOP_LENS)
    gui.save_lens_for(side, SIDE_LENS)
    assert lens.load_saved() == {top.key: TOP_LENS, side.key: SIDE_LENS}
    assert gui.saved_lens_for(top, [top, side]) == TOP_LENS
    assert gui.saved_lens_for(side, [top, side]) == SIDE_LENS
    # The browser label is shared, so the preview can only find a lens by OpenCV index.
    by_name, by_index = lens.browser_lenses()
    assert by_name == {}
    assert by_index == {0: TOP_LENS, 1: SIDE_LENS}
    assert gui.forget_lens_for(top, [top, side])
    assert gui.saved_lens_for(top, [top, side]) is None
    assert gui.saved_lens_for(side, [top, side]) == SIDE_LENS


def test_a_name_only_lens_belongs_to_the_one_camera_with_that_name() -> None:
    (camera,) = cc.list_calibration_cameras(TWINS[:1])
    lens.save_lens("USB Camera", TOP_LENS)
    assert gui.saved_lens_for(camera, [camera]) == TOP_LENS
    assert gui.forget_lens_for(camera, [camera])
    assert lens.load_saved() == {}


def test_describe() -> None:
    assert (
        gui.describe({"fov": 119.6, "bend": 0.02, "zoom": 0.9})
        == "120° field of view · bend +0.020 · zoom 0.90"
    )
