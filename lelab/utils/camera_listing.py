"""macOS camera enumeration in OpenCV's index order, shared by the server and the colour-gain code.

``cv2.VideoCapture(i)`` on macOS opens the device at position ``i`` of AVFoundation's list sorted by
``uniqueID``. The listing returns each index with the device's real name, which is also the label a browser
shows for it, so a name identifies a camera across the server, the recorder and the frontend.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

_USB_UNIQUE_ID = re.compile(r"^0x([0-9a-fA-F]{9,})$")


# Runs in a fresh Python — see list_cameras_in_cv2_order for why.
# Mirrors OpenCV's macOS enumeration: video + muxed devices sorted by
# uniqueID (cap_avfoundation_mac.mm), so the returned index matches what
# cv2.VideoCapture will open.
_AVF_ENUM_SCRIPT = """
import json, objc
from Foundation import NSBundle
bundle = NSBundle.bundleWithPath_("/System/Library/Frameworks/AVFoundation.framework")
bundle.load()
types = []
for name in (
    "AVCaptureDeviceTypeBuiltInWideAngleCamera",
    "AVCaptureDeviceTypeExternalUnknown",   # macOS < 14
    "AVCaptureDeviceTypeExternal",          # macOS >= 14
    "AVCaptureDeviceTypeContinuityCamera",  # macOS >= 14
    "AVCaptureDeviceTypeDeskViewCamera",    # macOS >= 13
):
    loaded = {}
    try:
        objc.loadBundleVariables(bundle, loaded, [(name, b"@")])
    except objc.error:
        continue
    if loaded.get(name) is not None:
        types.append(loaded[name])
cls = objc.lookUpClass("AVCaptureDeviceDiscoverySession")
devs = []
for mt in ("vide", "muxx"):
    devs.extend(cls.discoverySessionWithDeviceTypes_mediaType_position_(types, mt, 0).devices() or [])
devs.sort(key=lambda d: d.uniqueID())
print(json.dumps([
    {"index": i, "name": str(d.localizedName()), "unique_id": str(d.uniqueID())}
    for i, d in enumerate(devs)
]))
"""


def usb_id_from_unique_id(unique_id: str) -> tuple[int, int] | None:
    """The USB ``(vid, pid)`` of a camera from its AVFoundation unique id, or ``None`` if it is not USB.

    A USB camera's id is the hex location followed by vid and pid (``0x11300000abd8050`` is ``0abd:8050``).
    Built-in and Continuity cameras use UUID-like ids instead.
    """
    match = _USB_UNIQUE_ID.match(unique_id.strip())
    if not match:
        return None
    digits = match.group(1)
    return int(digits[-8:-4], 16), int(digits[-4:], 16)


def list_cameras_in_cv2_order() -> list[dict[str, Any]]:
    """Enumerate macOS cameras in a fresh Python subprocess.

    AVFoundation's in-process device cache doesn't refresh on USB
    hotplug. Both the deprecated ``+devicesWithMediaType:`` and a
    long-lived ``AVCaptureDeviceDiscoverySession`` go stale, because
    device-connection notifications are delivered via
    ``NSNotificationCenter`` on a thread that needs an active
    ``NSRunLoop`` — uvicorn workers don't run one. A fresh subprocess
    re-initializes AVFoundation, which reads IOKit's live device state
    at startup.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", _AVF_ENUM_SCRIPT],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("AVFoundation enumeration subprocess failed: %s", e)
        return []
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        logger.warning("AVFoundation enumeration returned invalid JSON: %s", e)
        return []


def index_by_name[V](cameras: list[dict[str, Any]], saved: Mapping[str, V], noun: str) -> dict[int, V]:
    """Map each OpenCV index to the value saved for the camera listed at that index.

    A saved name that matches two listed cameras is skipped, because the frames could not be told apart.
    ``noun`` ("Colour gain", "Lens") names the setting in the log lines.
    """
    by_name: dict[str, list[int]] = {}
    for camera in cameras:
        by_name.setdefault(str(camera.get("name", "")).strip(), []).append(int(camera["index"]))

    mapping: dict[int, V] = {}
    for name, value in saved.items():
        indices = by_name.get(name, [])
        if not indices:
            logger.info("%s for '%s' not applied: that camera is not attached", noun, name)
        elif len(indices) > 1:
            logger.warning("%s for '%s' not applied: %d cameras have that name", noun, name, len(indices))
        else:
            mapping[indices[0]] = value
            logger.info("%s for '%s' applies to camera index %d", noun, name, indices[0])
    return mapping
