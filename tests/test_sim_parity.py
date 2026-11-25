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
"""Pure parts of the parity sheet; rendering is exercised by `python -m lelab.sim parity`."""

from __future__ import annotations

import numpy as np
import pytest

from lelab.sim.parity import blend, sheet

BLACK = np.zeros((480, 640, 3), np.uint8)
WHITE = np.full((480, 640, 3), 255, np.uint8)


def test_blend_is_the_mean_at_half_alpha() -> None:
    assert blend(BLACK, WHITE).mean() == pytest.approx(127.5, abs=1)


def test_blend_refuses_different_shapes() -> None:
    with pytest.raises(ValueError):
        blend(BLACK, WHITE[:100])


def test_sheet_has_three_columns_and_a_row_per_camera() -> None:
    out = sheet([("top", BLACK, WHITE), ("side", WHITE, BLACK)])
    assert out.shape == (2 * 480, 3 * 640, 3)
    assert out.dtype == np.uint8
    assert out[:, :640].mean() > 0  # the camera name is drawn on the first real picture
    assert (out[0:480, 640:1280] == 255).all()  # the sim column is untouched
