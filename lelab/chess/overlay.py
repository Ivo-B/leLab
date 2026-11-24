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


def _check_frame(frame: object, key: str = "frame") -> np.ndarray:
    """`frame` if it is an HxWx3 uint8 RGB array; a frame the markers cannot go on is an error, not a skip."""
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"{key} must be an HxWx3 uint8 numpy array, got {type(frame).__name__}")
    if frame.dtype != np.uint8:
        raise TypeError(f"{key} must be uint8 RGB, got dtype {frame.dtype}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"{key} must be HxWx3 (channels last, RGB), got shape {frame.shape}")
    return frame


def draw_goal_markers(
    frame: np.ndarray, source: str, target: str, calibration: BoardCalibration
) -> np.ndarray:
    """A copy of the RGB `frame` with the two markers painted in."""
    _check_frame(frame)
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
    # One attribute, so a reader on another thread (the live camera feed) sees the whole goal or none.
    _goal: tuple[str, str] | None = field(default=None, init=False, repr=False)

    @property
    def goal(self) -> tuple[str, str] | None:
        return self._goal

    @property
    def source(self) -> str | None:
        goal = self._goal
        return goal[0] if goal else None

    @property
    def target(self) -> str | None:
        goal = self._goal
        return goal[1] if goal else None

    def set_goal(self, source: str, target: str) -> None:
        # Validate both names before touching state, so a typo cannot leave a half-set goal.
        self.calibration.square_polygon(source)
        self.calibration.square_polygon(target)
        self._goal = (source, target)

    def clear_goal(self) -> None:
        self._goal = None

    def observation(self, observation: dict) -> dict:
        goal = self.goal
        if goal is None or self.camera_key not in observation:
            return observation
        frame = _check_frame(observation[self.camera_key], f"observation[{self.camera_key!r}]")
        observation[self.camera_key] = draw_goal_markers(frame, goal[0], goal[1], self.calibration)
        return observation

    def get_config(self) -> dict:
        goal = self._goal
        return {
            "camera_key": self.camera_key,
            "source": goal[0] if goal else None,
            "target": goal[1] if goal else None,
        }

    def transform_features(self, features):
        return features
