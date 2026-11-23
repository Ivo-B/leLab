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

"""Skips for the simulation tests: MuJoCo missing, no GL context, or no network for the arm assets."""

from __future__ import annotations

import importlib.util
import os
import traceback

import pytest

requires_mujoco = pytest.mark.skipif(
    importlib.util.find_spec("mujoco") is None, reason="lelab[sim] not installed"
)

GL_WORDS = ("GL", "OpenGL", "gladLoadGL", "context")


def _from_renderer_creation(error: BaseException) -> bool:
    """Whether `error` was raised inside `mujoco.Renderer(...)` (the only place a GL context is made)."""
    frames = traceback.extract_tb(error.__traceback__)
    return any(
        f"{os.sep}mujoco{os.sep}" in frame.filename
        and frame.filename.endswith("renderer.py")
        and frame.name == "__init__"
        for frame in frames
    )


def is_gl_context_error(error: BaseException) -> bool:
    """Whether `error` means no GL context could be created for `mujoco.Renderer` (skip, don't fail)."""
    import mujoco

    return _from_renderer_creation(error) and (
        isinstance(error, mujoco.FatalError)
        or any(word in f"{type(error).__name__} {error}" for word in GL_WORDS)
    )


def renderer_or_skip(model, height: int = 120, width: int = 160):
    import mujoco

    try:
        return mujoco.Renderer(model, height, width)
    except Exception as e:  # no GL context: headless CI, or the Cursor sandbox on macOS
        pytest.skip(f"no OpenGL context for MuJoCo rendering: {e}")


def arm_assets_or_skip():
    from lelab.sim.assets import SimUnavailable, arm_model_path

    try:
        return arm_model_path()
    except SimUnavailable as e:
        pytest.skip(str(e))
