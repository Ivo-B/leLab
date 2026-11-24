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

from lelab import lens
from lelab.scripts import camera_lens_gui as gui


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


def test_describe() -> None:
    assert (
        gui.describe({"fov": 119.6, "bend": 0.02, "zoom": 0.9})
        == "120° field of view · bend +0.020 · zoom 0.90"
    )
