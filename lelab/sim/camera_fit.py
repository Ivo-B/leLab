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

"""Write a fitted camera pose into the sim cell, from the Setup page's four clicked board corners."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from lelab.chess.camera_pose import CameraFit, CameraPoseError, fit_camera
from lelab.chess.transfers import DATASET_SIZE

from .cell import CameraPose, SimCell, load_cell, save_cell

CAMERA_FIELDS = {"top": "top_camera", "side": "side_camera"}
MAX_RMS_PX = 3.0


class CameraFitError(ValueError):
    """The request cannot be fitted."""


def world_corners(cell: SimCell) -> np.ndarray:
    """a1, h1, h8, a8 on the board surface, in the arm frame (as `ChessWorld.board_calibration` projects them)."""
    z = cell.board_thickness
    x0, x1, h = cell.board_x0, cell.board_x0 + 8 * cell.square_size, cell.half
    return np.array([(x0, h, z), (x0, -h, z), (x1, -h, z), (x1, h, z)], dtype=np.float64)


def apply_fit(cell: SimCell, camera: str, fit: CameraFit) -> SimCell:
    pose = CameraPose(fit.pos, fit.xyaxes, fit.fovy)
    return replace(cell, **{CAMERA_FIELDS[camera]: pose})


def _size(value: Any) -> tuple[int, int]:
    try:
        width, height = (int(v) for v in value)
    except (TypeError, ValueError) as e:
        raise CameraFitError("image_size must be [width, height]") from e
    if width <= 0 or height <= 0:
        raise CameraFitError("image_size must be positive")
    if abs(width / height - DATASET_SIZE[0] / DATASET_SIZE[1]) > 0.01:
        raise CameraFitError(
            f"The reference is {width}x{height}; the sim renders {DATASET_SIZE[0]}x{DATASET_SIZE[1]} (4:3). "
            "Set the camera to a 4:3 mode and capture the reference again."
        )
    return width, height


def handle_camera_pose(body: dict[str, Any]) -> dict[str, Any]:
    camera = body.get("camera")
    if camera not in CAMERA_FIELDS:
        raise CameraFitError(f"camera must be one of {sorted(CAMERA_FIELDS)}")
    size = _size(body.get("image_size"))
    corners = body.get("board_corners")
    if not isinstance(corners, list) or len(corners) != 4:
        raise CameraFitError("board_corners needs exactly four [x, y] points: a1, h1, h8, a8")
    cell = load_cell()
    current: CameraPose = getattr(cell, CAMERA_FIELDS[camera])
    fovy = body.get("fovy")
    try:
        fit = fit_camera(world_corners(cell), corners, size, fovy=fovy, fallback_fovy=current.fovy)
    except CameraPoseError as e:
        raise CameraFitError(str(e)) from e
    warnings: list[str] = []
    if fovy is None and not fit.fov_fitted:
        warnings.append(
            f"This view cannot tell its field of view from four corners; kept {fit.fovy:g}°. "
            "Pass the field of view of the straightened picture if it is not that."
        )
    if fit.rms_px > MAX_RMS_PX:
        warnings.append(
            f"The corners are {fit.rms_px:.1f} px from the fitted board; check the click order (a1, h1, h8, a8)."
        )
    applied = body.get("apply") is True
    if applied:
        save_cell(apply_fit(cell, camera, fit))
    return {
        "camera": camera,
        "pos": list(fit.pos),
        "xyaxes": list(fit.xyaxes),
        "fovy": fit.fovy,
        "rms_px": fit.rms_px,
        "fov_fitted": fit.fov_fitted,
        "warnings": warnings,
        "applied": applied,
    }
