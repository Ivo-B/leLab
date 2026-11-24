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

"""``python -m lelab.scripts.rollout_gain``: LeRobot's rollout with the saved camera corrections applied.

LeLab starts inference in a subprocess. The parent passes the colour gains and the lens models of the
attached cameras in the ``LELAB_CAMERA_GAINS`` and ``LELAB_CAMERA_LENSES`` environment variables, and this
wrapper installs them before LeRobot's rollout opens the cameras. Every other argument is forwarded to
LeRobot unchanged.
"""

from __future__ import annotations

import os

from lelab import color_gain, lens


def main() -> None:
    gains = os.environ.get(color_gain.ENV_VAR, "")
    if gains:
        color_gain.install(color_gain.decode_from_env(gains))
    lenses = os.environ.get(lens.ENV_VAR, "")
    if lenses:
        lens.install(lens.decode_from_env(lenses))
    from lerobot.scripts.lerobot_rollout import main as lerobot_rollout_main

    lerobot_rollout_main()


if __name__ == "__main__":
    main()
