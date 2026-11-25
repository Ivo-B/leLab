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

"""A camera's pose in the arm frame from clicked board corners.

The model is the one MuJoCo renders with and `ChessWorld.project` uses: a pinhole looking along the
camera's -z axis, square pixels, the principal point at the centre, `fovy` the vertical field of view.
The clicks must be on a straightened picture (the lens correction applied), or the model does not hold.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

FOV_GRID = np.arange(20.0, 100.01, 0.5)
FOV_FLAT_PX = 1.0  # the error must rise this much within FOV_PROBE_DEG of the best field of view
FOV_PROBE_DEG = 5.0


class CameraPoseError(ValueError):
    """The clicks or the points cannot give a pose."""


@dataclass(frozen=True)
class CameraFit:
    pos: tuple[float, float, float]
    xyaxes: tuple[float, float, float, float, float, float]
    fovy: float
    rms_px: float
    fov_fitted: bool


def _focal(fovy: float, height: int) -> float:
    return 0.5 * height / math.tan(math.radians(fovy) / 2)


def project(pos, xyaxes, fovy: float, size: tuple[int, int], points) -> np.ndarray:
    """World points (N, 3) -> pixels (N, 2) in a `size` = (width, height) picture."""
    width, height = size
    x = np.asarray(xyaxes[:3], dtype=np.float64)
    y = np.asarray(xyaxes[3:], dtype=np.float64)
    x = x / np.linalg.norm(x)
    y = y - x * (x @ y)  # MuJoCo orthogonalises xyaxes the same way
    y = y / np.linalg.norm(y)
    rot = np.stack([x, y, np.cross(x, y)], axis=1)
    local = (np.asarray(points, dtype=np.float64) - np.asarray(pos, dtype=np.float64)) @ rot
    focal = _focal(fovy, height)
    return np.stack(
        [width / 2 + focal * local[:, 0] / -local[:, 2], height / 2 - focal * local[:, 1] / -local[:, 2]],
        axis=1,
    )


def _solve(points3d: np.ndarray, points2d: np.ndarray, fovy: float, size: tuple[int, int]) -> CameraFit:
    width, height = size
    focal = _focal(fovy, height)
    k = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(points3d, points2d, k, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise CameraPoseError("The corners do not give a camera pose; click them again.")
    rvec, tvec = cv2.solvePnPRefineLM(points3d, points2d, k, None, rvec, tvec)
    rotation, _ = cv2.Rodrigues(rvec)
    to_world = rotation.T  # columns: the OpenCV camera's x (right), y (down), z (forward) in the world
    pos = tuple(float(v) for v in (-to_world @ tvec).ravel())
    xyaxes = tuple(float(v) for v in (*to_world[:, 0], *(-to_world[:, 1])))  # MuJoCo's y is up
    error = project(pos, xyaxes, fovy, size, points3d) - points2d
    rms = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
    return CameraFit(pos, xyaxes, float(fovy), rms, False)  # type: ignore[arg-type]


def fit_camera(
    points3d: Sequence[Sequence[float]],
    points2d: Sequence[Sequence[float]],
    size: tuple[int, int],
    fovy: float | None = None,
    fallback_fovy: float = 45.0,
) -> CameraFit:
    """The camera pose that puts `points3d` on `points2d`.

    With `fovy` given it is fixed. Without it the field of view is searched; a view that cannot tell it
    (straight down onto a plane) keeps `fallback_fovy` and reports `fov_fitted=False`."""
    p3 = np.asarray(points3d, dtype=np.float64)
    p2 = np.asarray(points2d, dtype=np.float64)
    if p3.ndim != 2 or p3.shape[1] != 3 or p2.shape != (len(p3), 2) or len(p3) < 4:
        raise CameraPoseError("Need at least four corners, as [x, y] pixel pairs.")
    if not (np.isfinite(p3).all() and np.isfinite(p2).all()):
        raise CameraPoseError("The corners must be finite numbers.")
    if fovy is not None:
        return _solve(p3, p2, float(fovy), size)
    errors = np.array([_solve(p3, p2, float(f), size).rms_px for f in FOV_GRID])
    best = int(np.argmin(errors))
    probe = int(round(FOV_PROBE_DEG / (FOV_GRID[1] - FOV_GRID[0])))
    lo, hi = max(best - probe, 0), min(best + probe, len(FOV_GRID) - 1)
    if max(errors[lo], errors[hi]) - errors[best] < FOV_FLAT_PX:
        return _solve(p3, p2, float(fallback_fovy), size)
    fit = _solve(p3, p2, float(FOV_GRID[best]), size)
    return CameraFit(fit.pos, fit.xyaxes, fit.fovy, fit.rms_px, True)
