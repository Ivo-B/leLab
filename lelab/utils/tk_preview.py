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

"""Helpers for showing camera frames in a Tk window; no Tk import, so they run and test without a display."""

from __future__ import annotations

import numpy as np

MAX_DISPLAY = (800, 600)


def display_scale(width: int, height: int, limit: tuple[int, int] = MAX_DISPLAY) -> float:
    """Factor from frame pixels to window pixels that fits the frame in ``limit`` without enlarging it."""
    return min(1.0, limit[0] / width, limit[1] / height)


def to_ppm(rgb: np.ndarray) -> bytes:
    """An RGB image as binary PPM, which Tk's PhotoImage reads without Pillow."""
    height, width = rgb.shape[:2]
    return f"P6 {width} {height} 255\n".encode() + np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
