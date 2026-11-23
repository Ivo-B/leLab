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

"""Goal markers: a red disc on the source and a green disc on the target.

The same drawing runs at record time (so the dataset video carries the markers)
and at inference time (so the policy sees what it was trained on). It is wrapped
in a LeRobot observation processor step so both paths share it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from lerobot.processor import ObservationProcessorStep

from .calibration import BoardCalibration

SOURCE_RGB = (255, 0, 0)
TARGET_RGB = (0, 255, 0)


def _disc(frame: np.ndarray, name: str, colour: tuple[int, int, int], cal: BoardCalibration) -> None:
    cx, cy = cal.square_center(name)
    cv2.circle(frame, (int(round(cx)), int(round(cy))), cal.marker_radius(name), colour, thickness=-1)


def draw_goal_markers(
    frame: np.ndarray, source: str, target: str, calibration: BoardCalibration
) -> np.ndarray:
    """A copy of the RGB `frame` with the two markers painted in."""
    height, width = frame.shape[:2]
    cal = calibration.scaled_to(width, height)
    out = np.ascontiguousarray(frame).copy()
    _disc(out, source, SOURCE_RGB, cal)
    _disc(out, target, TARGET_RGB, cal)
    return out


@dataclass
class GoalOverlayStep(ObservationProcessorStep):
    """Paints the current goal into `observation[camera_key]`, nothing else."""

    calibration: BoardCalibration
    camera_key: str = "top"
    source: str | None = field(default=None)
    target: str | None = field(default=None)

    @property
    def goal(self) -> tuple[str, str] | None:
        if self.source is None or self.target is None:
            return None
        return (self.source, self.target)

    def set_goal(self, source: str, target: str) -> None:
        # Validate both names before touching state, so a typo cannot leave a half-set goal.
        self.calibration.square_polygon(source)
        self.calibration.square_polygon(target)
        self.source, self.target = source, target

    def clear_goal(self) -> None:
        self.source = self.target = None

    def observation(self, observation: dict) -> dict:
        goal = self.goal
        frame = observation.get(self.camera_key)
        if goal is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            return observation
        observation[self.camera_key] = draw_goal_markers(frame, goal[0], goal[1], self.calibration)
        return observation

    def get_config(self) -> dict:
        return {"camera_key": self.camera_key, "source": self.source, "target": self.target}

    def transform_features(self, features):
        return features
