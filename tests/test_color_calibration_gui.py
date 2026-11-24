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
"""The calibration window's helpers; the window itself is checked by hand."""

from __future__ import annotations

import numpy as np
import pytest

from lelab import color_calibration
from lelab.scripts import color_calibration_gui as gui
from lelab.utils import tk_preview


def test_display_scale_fits_without_enlarging() -> None:
    assert tk_preview.display_scale(1920, 1080) == pytest.approx(800 / 1920)
    assert tk_preview.display_scale(640, 480) == 1.0
    assert tk_preview.display_scale(800, 1200) == pytest.approx(0.5)


def test_to_ppm_has_header_and_rgb_bytes() -> None:
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[0, 0] = (1, 2, 3)
    data = tk_preview.to_ppm(rgb)
    assert data.startswith(b"P6 3 2 255\n")
    assert data[len(b"P6 3 2 255\n") :][:3] == b"\x01\x02\x03"
    assert len(data) == len(b"P6 3 2 255\n") + 18


def test_roi_from_drag_maps_window_to_frame_pixels() -> None:
    assert gui.roi_from_drag((300, 200), (100, 100), 0.5, 1920, 1080) == (200, 200, 400, 200)
    assert gui.roi_from_drag((100, 100), (103, 300), 0.5, 1920, 1080) is None
    assert gui.roi_from_drag((700, 400), (900, 700), 0.5, 1920, 1080) == (1400, 800, 400, 280)


def test_roi_round_trips_to_the_window() -> None:
    assert gui.roi_to_display((200, 200, 400, 200), 0.5) == (100, 100, 300, 200)


def test_readout() -> None:
    stats = color_calibration.PatchStats(blue=60, green=120, red=90, clipped=0, spread=0)
    assert gui.readout(stats) == "red/green 0.75 · blue/green 0.50"
