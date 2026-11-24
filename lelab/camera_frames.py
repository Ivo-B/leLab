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

"""One hook into LeRobot's camera frames, shared by the lens and colour corrections.

Recording, the live view and inference all read frames through ``OpenCVCamera._postprocess_image``. LeRobot
has no hook for changing them, so the first ``register`` wraps that method once. Each correction owns a named
stage holding a function per OpenCV camera index; the wrapper runs the stages in ``STAGES`` order on the
frame as the camera sent it (BGR, before LeRobot's colour conversion and rotation).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

FrameFunction = Callable[[np.ndarray], np.ndarray]

# The lens comes first: it moves pixels, and the colour gain then scales what is there.
STAGES = ("lens", "colour")

_active: dict[str, dict[int, FrameFunction]] = {stage: {} for stage in STAGES}
_HOOKED = "_lelab_frame_stages"


def register(stage: str, functions: Mapping[int, FrameFunction]) -> None:
    """Use these functions for the cameras at these indices in this stage. Replaces that stage's earlier map.

    The first call installs the wrapper; later calls only change the map, and an empty map switches the stage off.
    """
    if stage not in STAGES:
        raise ValueError(f"Unknown frame stage '{stage}'. Known: {', '.join(STAGES)}")
    from lerobot.cameras.opencv import OpenCVCamera

    _active[stage] = {int(index): function for index, function in functions.items()}
    if getattr(OpenCVCamera, _HOOKED, False):
        return

    original = OpenCVCamera._postprocess_image

    def _postprocess_image(self, image: np.ndarray) -> np.ndarray:
        if isinstance(self.index_or_path, int):
            for name in STAGES:
                function = _active[name].get(self.index_or_path)
                if function is not None:
                    image = function(image)
        return original(self, image)

    OpenCVCamera._postprocess_image = _postprocess_image
    setattr(OpenCVCamera, _HOOKED, True)
