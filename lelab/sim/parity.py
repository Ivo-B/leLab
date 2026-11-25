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
"""Real references next to the simulated cell, to judge camera pose, field of view and wrist orientation."""

from __future__ import annotations

from pathlib import Path

import chess
import cv2
import numpy as np

from lelab import camera_reference
from lelab.chess.transfers import DATASET_SIZE

CAMERAS = ("top", "side", "wrist")


def blend(real: np.ndarray, sim: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if real.shape != sim.shape:
        raise ValueError(f"shapes differ: {real.shape} vs {sim.shape}")
    return cv2.addWeighted(real, 1 - alpha, sim, alpha, 0)


def sheet(rows: list[tuple[str, np.ndarray, np.ndarray]]) -> np.ndarray:
    """One row per camera: the real picture (named at the top-left), the sim render, and their blend."""
    lines = []
    for name, real, sim in rows:
        labelled = real.copy()
        cv2.putText(labelled, name, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        lines.append(np.concatenate([labelled, sim, blend(real, sim)], axis=1))
    return np.concatenate(lines, axis=0)


def _reference(robot: str, camera: str) -> np.ndarray | None:
    png = camera_reference.load_reference(robot, camera)
    if png is None:
        return None
    bgr = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    width, height = DATASET_SIZE
    if abs(rgb.shape[1] / rgb.shape[0] - width / height) > 0.01:
        raise ValueError(
            f"The {camera} reference is {rgb.shape[1]}x{rgb.shape[0]}, not 4:3; capture it again."
        )
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def run_parity(robot: str, out: Path) -> list[str]:
    """Write the parity sheet for the cameras that have a reference; returns the cameras it found."""
    from .cell import load_cell
    from .scene import SimPosition
    from .world import ChessWorld

    width, height = DATASET_SIZE
    world = ChessWorld(load_cell(), SimPosition(chess.Board()), width=width, height=height)
    try:
        rows, found = [], []
        for camera in CAMERAS:
            real = _reference(robot, camera)
            if real is not None:
                rows.append((camera, real, world.render(camera)))
                found.append(camera)
    finally:
        world.close()
    if rows:
        cv2.imwrite(str(out), cv2.cvtColor(sheet(rows), cv2.COLOR_RGB2BGR))
    return found
