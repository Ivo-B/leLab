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

"""The SO-101 arm model for the simulator.

MuJoCo Menagerie's `robotstudio_so101` (Apache-2.0), taken from LeRobot's SO-101
tabletop environment at a pinned revision so every machine simulates the same arm.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from huggingface_hub import snapshot_download

from lelab.utils import config

ASSETS_REPO = "lerobot/so101-tabletop-mujoco"
ASSETS_REVISION = "e68145d8bd793c27fa2c13e31369a616895c5acb"
ARM_DIR = "robotstudio_so101"


class SimUnavailable(RuntimeError):  # noqa: N818 - name fixed by the plan
    """MuJoCo, the arm assets or a GL context is missing."""


def require_mujoco() -> None:
    if importlib.util.find_spec("mujoco") is None:
        raise SimUnavailable("MuJoCo is not installed; install the extra: uv sync --extra sim")


def arm_model_path() -> Path:
    root = Path(config.SIM_ASSETS_PATH)
    model = root / ARM_DIR / "so101.xml"
    if model.is_file():
        return model
    try:
        snapshot_download(
            repo_id=ASSETS_REPO,
            revision=ASSETS_REVISION,
            allow_patterns=[f"{ARM_DIR}/*"],
            local_dir=str(root),
        )
    except Exception as e:
        raise SimUnavailable(
            f"Could not download the SO-101 model from {ASSETS_REPO}@{ASSETS_REVISION[:8]}: {e}"
        ) from e
    if not model.is_file():
        raise SimUnavailable(f"{ASSETS_REPO}@{ASSETS_REVISION[:8]} has no {ARM_DIR}/so101.xml")
    return model
