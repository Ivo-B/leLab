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

"""Fisheye lens correction for one camera, applied to every frame LeRobot reads from it.

A wide-angle lens bends straight lines, which breaks everything that assumes a straight picture: the chess
board homography, the empty-board reference and the policy's view. The model is an equidistant fisheye with
one fine-tuning term, saved per camera key like the colour gain (``lelab-camera lens``,
``lelab-camera-lens``) and run in ``lelab.camera_frames`` before the colour gain.

Positions are measured from the picture centre in units of the half-diagonal ``R``, so one model fits any
resolution of the same aspect ratio. For an output pixel at offset ``o`` (``rho = |o|``)::

    theta_dmax = theta_max * (1 + bend * theta_max**2)  # the corner, source radius 1 by definition
    theta = atan(rho * theta_dmax / zoom)  # the ray angle the straightened picture shows
    r_source = theta * (1 + bend * theta**2) / theta_dmax
    source = centre + o / rho * r_source * R

The same formula runs in the browser (``frontend/src/lib/cameraLens.ts`` and the WebGL shader); the shared
numeric fixtures in the tests keep the copies from drifting apart.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from functools import lru_cache, partial
from typing import Any

import cv2
import numpy as np

from . import camera_frames
from .utils import config
from .utils.camera_store import CameraStore, clean_name

logger = logging.getLogger(__name__)

PARAMS = ("fov", "bend", "zoom")
FOV_RANGE = (10.0, 200.0)
BEND_RANGE = (-0.2, 0.5)
ZOOM_RANGE = (0.2, 3.0)
ENV_VAR = "LELAB_CAMERA_LENSES"
# The projection curve theta*(1 + bend*theta^2) must keep rising; below this slope it nearly folds back.
MIN_SLOPE = 0.05

Lens = dict[str, float]


class LensError(ValueError):
    """The camera name or the lens values are not valid."""


def normalize_name(name: str) -> str:
    return clean_name(name, LensError)


def validate_lens(values: Mapping[str, Any]) -> Lens:
    """The lens as floats, or raise if a value is missing, out of range, or the projection curve folds back."""
    if set(values) != set(PARAMS):
        raise LensError(f"Give exactly the values {', '.join(PARAMS)}")
    ranges = {"fov": FOV_RANGE, "bend": BEND_RANGE, "zoom": ZOOM_RANGE}
    checked: Lens = {}
    for name in PARAMS:
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise LensError(f"'{name}' must be a number, got {value!r}")
        number = float(value)
        low, high = ranges[name]
        if not math.isfinite(number) or not low <= number <= high:
            raise LensError(f"'{name}' must be between {low:g} and {high:g}, got {value}")
        checked[name] = number
    theta_max = math.radians(checked["fov"] / 2)
    if 1 + 3 * checked["bend"] * theta_max**2 < MIN_SLOPE:
        raise LensError(
            f"bend {checked['bend']:g} folds the picture back at a {checked['fov']:g}° field of view; raise bend or lower fov"
        )
    return checked


def parse_size(text: str) -> tuple[int, int]:
    """``"1920x1080"`` as ``(1920, 1080)``."""
    try:
        width, height = (int(part) for part in text.lower().split("x"))
    except ValueError:
        raise LensError(f"'{text}' is not a size like 1920x1080") from None
    if width <= 0 or height <= 0:
        raise LensError(f"The size must be positive, got '{text}'")
    return width, height


# -- saved lenses -----------------------------------------------------------------------------------

_store: CameraStore[Lens] = CameraStore(
    lambda: config.CAMERA_LENS_FILE, validate_lens, normalize_name, "lens"
)
load_saved = _store.load
save_lens = _store.save
forget_lens = _store.forget


# -- the model --------------------------------------------------------------------------------------


def _theta_dmax(fov: float, bend: float) -> tuple[float, float]:
    theta_max = math.radians(fov / 2)
    return theta_max, theta_max * (1 + bend * theta_max**2)


def _filled(lens: Mapping[str, float], width: int, height: int) -> bool:
    """True when every output pixel samples the middle of a real camera pixel, so nothing is black or blended."""
    # Only the picture's edge can leave the camera picture: the correction stretches radially outward.
    x, y = source_coordinates(lens, width, height)
    return bool(x.min() >= 0.5 and x.max() <= width - 0.5 and y.min() >= 0.5 and y.max() <= height - 0.5)


def fit_zoom(lens: Mapping[str, float], width: int, height: int) -> float:
    """The smallest zoom (the widest view) with no black borders for a picture of this size.

    The black appears first at the middle of the shorter edges, not at the corners, so the answer depends on
    the aspect ratio. The size only sets the aspect, so a small proxy picture is searched.
    """
    proxy = (
        (96, max(2, round(96 * height / width)))
        if width >= height
        else (max(2, round(96 * width / height)), 96)
    )
    low, high = ZOOM_RANGE
    if _filled({**lens, "zoom": low}, *proxy):
        return low
    if not _filled({**lens, "zoom": high}, *proxy):
        return high
    for _ in range(40):  # a larger zoom only ever pulls the samples inward, so the answer can be bisected
        middle = (low + high) / 2
        low, high = (low, middle) if _filled({**lens, "zoom": middle}, *proxy) else (middle, high)
    return round(high + 1e-4, 4)


def source_coordinates(lens: Mapping[str, float], width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """For every output pixel, where to sample the camera picture, as ``(x, y)`` arrays of shape ``(h, w)``.

    Coordinates are continuous with pixel ``i`` covering ``[i, i + 1)``, so a pixel's centre is ``i + 0.5``.
    """
    half_diagonal = math.hypot(width, height) / 2
    theta_max, theta_dmax = _theta_dmax(lens["fov"], lens["bend"])
    bend, zoom = lens["bend"], lens["zoom"]
    ox, oy = np.meshgrid(
        (np.arange(width) + 0.5 - width / 2) / half_diagonal,
        (np.arange(height) + 0.5 - height / 2) / half_diagonal,
    )
    rho = np.hypot(ox, oy)
    theta = np.arctan(rho * theta_dmax / zoom)
    r_source = theta * (1 + bend * theta**2) / theta_dmax
    # r_source / rho tends to 1 / zoom at the centre, where both are zero.
    scale = np.divide(r_source, rho, out=np.full_like(rho, 1 / zoom), where=rho > 1e-12)
    return width / 2 + ox * scale * half_diagonal, height / 2 + oy * scale * half_diagonal


@lru_cache(maxsize=16)
def _maps(width: int, height: int, fov: float, bend: float, zoom: float) -> tuple[np.ndarray, np.ndarray]:
    x, y = source_coordinates({"fov": fov, "bend": bend, "zoom": zoom}, width, height)
    # cv2.remap puts the centre of pixel i at i, one half pixel less than source_coordinates.
    fixed_point = cv2.convertMaps((x - 0.5).astype(np.float32), (y - 0.5).astype(np.float32), cv2.CV_16SC2)
    return fixed_point[0], fixed_point[1]


def apply_lens(image: np.ndarray, lens: Mapping[str, float]) -> np.ndarray:
    """Straighten a frame of any channel count; pixels the camera did not see are black."""
    height, width = image.shape[:2]
    map1, map2 = _maps(width, height, lens["fov"], lens["bend"], lens["zoom"])
    return cv2.remap(image, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# -- measuring the field of view from a stroke ------------------------------------------------------

MIN_STROKE_POINTS = 5
# A stroke must bow at least this far (in half-diagonals: about one pixel on a 1080p picture) to measure anything.
MIN_BOW = 0.001
# The best field of view must cut the bow to this fraction of the drawn bow, or the stroke was not a straight edge.
STRAIGHTEN_RATIO = 0.25
FIT_FOV_STEP = 0.25
FIT_FOV_RANGE = (10.0, 170.0)


def _line_residual(points: np.ndarray) -> float:
    """Sum of squared distances from the points to their best-fitting straight line."""
    centred = points - points.mean(axis=0)
    return float(np.linalg.svd(centred, compute_uv=False)[-1] ** 2)


def _straightened_positions(normalised: np.ndarray, fov: float, bend: float) -> np.ndarray:
    """Where camera points land in the straightened picture, zoom 1, in half-diagonal units (the inverse model)."""
    _, theta_dmax = _theta_dmax(fov, bend)
    radius = np.hypot(normalised[:, 0], normalised[:, 1])
    # Solve theta * (1 + bend * theta**2) = r_source * theta_dmax for theta by bisection; the curve is monotonic
    # for every lens that validate_lens accepts, and points beyond the picture's corner are clamped to it.
    target = np.minimum(radius, 1.0) * theta_dmax
    low = np.zeros_like(target)
    high = np.full_like(target, _theta_dmax(fov, bend)[0])
    for _ in range(60):
        middle = (low + high) / 2
        too_far = middle * (1 + bend * middle**2) > target
        high = np.where(too_far, middle, high)
        low = np.where(too_far, low, middle)
    theta = (low + high) / 2
    direction = np.divide(
        normalised, radius[:, None], out=np.zeros_like(normalised), where=radius[:, None] > 1e-12
    )
    return direction * (np.tan(theta) / theta_dmax)[:, None]


def fit_fov_to_line(
    points: Sequence[tuple[float, float]], width: int, height: int, bend: float = 0.0
) -> float:
    """The field of view at which a stroke drawn along a straight edge comes out straight.

    ``points`` are pixel positions in the picture as the camera shows it (not straightened), ``width`` x
    ``height`` in size. Zoom only scales a straight line, so it does not change straightness and is not searched;
    ``bend`` is used as given. Raises ``LensError`` when the stroke is too short, does not bow in the picture
    (a line through the middle stays straight under any lens), or no field of view straightens it.
    """
    stroke = np.asarray(points, dtype=float)
    if stroke.ndim != 2 or stroke.shape[1] != 2 or len(stroke) < MIN_STROKE_POINTS:
        raise LensError("Drag a longer stroke along a straight edge")
    half_diagonal = math.hypot(width, height) / 2
    normalised = (stroke - [width / 2, height / 2]) / half_diagonal

    drawn_bow = math.sqrt(_line_residual(normalised) / len(normalised))
    if drawn_bow < MIN_BOW:
        raise LensError(
            "This line does not bow in the picture, so the lens cannot be measured from it. Drag along a straight "
            "edge further out, such as the long side of the table."
        )

    best_fov, best_residual = None, math.inf
    for fov in np.arange(FIT_FOV_RANGE[0], FIT_FOV_RANGE[1] + FIT_FOV_STEP / 2, FIT_FOV_STEP):
        try:
            validate_lens({"fov": float(fov), "bend": bend, "zoom": 1.0})
        except LensError:
            continue  # this lens would fold the picture back; no picture from it can be straight
        residual = _line_residual(_straightened_positions(normalised, float(fov), bend))
        if residual < best_residual:
            best_fov, best_residual = float(fov), residual
    if best_fov is None or best_residual > STRAIGHTEN_RATIO * drawn_bow**2 * len(normalised):
        raise LensError(
            "Could not straighten that line with any field of view. Drag it again along one straight edge."
        )
    return round(best_fov, 2)


# -- attached cameras and the frame hook ------------------------------------------------------------


def lenses_for_session(cameras: Mapping[str, Mapping[str, Any]], listing: Sequence[Any]) -> dict[int, Lens]:
    """The lens saved for each resolved session camera, by OpenCV index. ``lens_correction: false`` skips one.

    ``listing`` is ``camera_identity.attached()``: each camera's ``camera_id`` and the ``key`` its lens is saved under.
    """
    saved = load_saved()
    keys = {camera.camera_id: camera.key for camera in listing}
    lenses: dict[int, Lens] = {}
    for camera in cameras.values():
        key = keys.get(str(camera.get("camera_id")))
        if camera.get("lens_correction", True) is not False and key in saved:
            lenses[int(camera["camera_index"])] = saved[key]
    return lenses


def install(index_map: Mapping[int, Lens]) -> None:
    """Straighten the frames of the OpenCV cameras at these indices. Replaces any earlier map; empty turns it off."""
    camera_frames.register(
        "lens", {int(index): partial(apply_lens, lens=dict(lens)) for index, lens in index_map.items()}
    )


def encode_for_env(index_map: Mapping[int, Lens]) -> str:
    return json.dumps({str(index): lens for index, lens in index_map.items()}, sort_keys=True)


def decode_from_env(text: str) -> dict[int, Lens]:
    return {int(index): validate_lens(lens) for index, lens in json.loads(text).items()}
