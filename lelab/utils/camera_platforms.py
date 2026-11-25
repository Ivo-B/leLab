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

"""Attached cameras in the order each platform's cv2 backend indexes them.

- macOS: AVFoundation, with each device's ``uniqueID`` (via a PyObjC subprocess);
- Windows: DirectShow FriendlyName (via pygrabber; recording is pinned to DSHOW);
- Linux: the v4l2 device name from sysfs.
"""

from __future__ import annotations

import contextlib
import logging
import platform
from typing import Any

from .camera_listing import list_cameras_in_cv2_order

logger = logging.getLogger(__name__)


def _generic_cv2_cameras(backend) -> list[dict[str, Any]]:
    """Last-resort enumeration: probe cv2 indices with placeholder names."""
    import cv2

    cameras: list[dict[str, Any]] = []
    for i in range(10):
        cap = cv2.VideoCapture(i, backend)
        opened = cap.isOpened()
        cap.release()
        if opened:
            cameras.append({"index": i, "name": f"Camera {i}", "available": True})
    return cameras


@contextlib.contextmanager
def _windows_com_initialized():
    """Initialize COM for the current Windows worker thread when available."""
    try:
        import comtypes
    except ImportError:
        yield
        return

    try:
        comtypes.CoInitialize()
    except OSError as e:
        logger.warning("Windows COM initialization failed: %s", e)
        yield
        return

    try:
        yield
    finally:
        comtypes.CoUninitialize()


def _windows_cameras() -> list[dict[str, Any]]:
    """Enumerate Windows cameras with their real DirectShow names.

    pygrabber lists DirectShow video devices in the same order cv2's DSHOW
    backend indexes them (which recording is pinned to), so the returned index
    matches what ``cv2.VideoCapture(i, CAP_DSHOW)`` opens. Falls back to generic
    names if pygrabber is unavailable.
    """
    with _windows_com_initialized():
        try:
            from pygrabber.dshow_graph import FilterGraph

            names = FilterGraph().get_input_devices()
        except Exception as e:  # ImportError, or a COM/DirectShow failure
            logger.warning("pygrabber unavailable; using generic camera names: %s", e)
            import cv2

            return _generic_cv2_cameras(cv2.CAP_DSHOW)
        return [{"index": i, "name": name, "available": True} for i, name in enumerate(names)]


def _v4l2_camera_name(index: int) -> str | None:
    """Real camera name for /dev/video{index} from sysfs (Linux, no deps)."""
    try:
        with open(f"/sys/class/video4linux/video{index}/name", encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def _linux_cameras() -> list[dict[str, Any]]:
    """Enumerate Linux cameras, naming each from sysfs (no extra deps)."""
    import cv2

    cameras: list[dict[str, Any]] = []
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        opened = cap.isOpened()
        cap.release()
        if not opened:
            continue
        cameras.append({"index": i, "name": _v4l2_camera_name(i) or f"Camera {i}", "available": True})
    return cameras


def list_cameras_for_platform() -> list[dict[str, Any]]:
    """Cameras in the order this platform's cv2 backend opens them; macOS entries carry ``unique_id``."""
    system = platform.system()
    if system == "Darwin":
        return [{**camera, "available": True} for camera in list_cameras_in_cv2_order()]
    if system == "Windows":
        return _windows_cameras()
    if system == "Linux":
        return _linux_cameras()
    import cv2

    return _generic_cv2_cameras(cv2.CAP_ANY)
