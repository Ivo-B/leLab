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
import time
from collections import Counter
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


def usb_location_from_unique_id(unique_id: str) -> int | None:
    """The USB location of a camera (bus and port path, as IOKit's ``locationID``), or ``None`` if it is not USB.

    It is the hex digits before the vid and pid (``0x11300000abd8050`` is ``0x1130000``, bus 1, port 3), and
    differs for two identical cameras on different ports. libusb reports the same number for the device.
    """
    match = _USB_UNIQUE_ID.match(unique_id.strip())
    return int(match.group(1)[:-8], 16) if match else None


def port_path(location: int) -> str:
    """The USB bus and hub ports of a device the way Linux writes them: ``"1-4.1"`` for ``0x01141000``."""
    ports = [(location >> (20 - 4 * depth)) & 0xF for depth in range(6)]
    ports = ports[: max((i + 1 for i, port in enumerate(ports) if port), default=0)]
    return f"{location >> 24}-{'.'.join(str(port) for port in ports)}"


# -- camera keys ------------------------------------------------------------------------------------
#
# Settings saved per camera (colour gain, lens) are keyed by the camera's name and AVFoundation unique id,
# ``"USB Camera [0x11300000abd8050]"``: two cameras can share a name but never an id. A key without the
# bracketed id (saved before ids were used) is a legacy key and applies to a camera only when that name is
# unique among the attached ones.

_KEY = re.compile(r"^(?P<name>.*?)\s*\[(?P<id>[^\[\]]+)\]$")


class CameraKeyError(ValueError):
    """A camera named by the user cannot be turned into one key."""


def camera_key(name: str, unique_id: str) -> str:
    name = name.strip()
    unique_id = unique_id.strip()
    return f"{name} [{unique_id}]" if unique_id else name


def split_key(key: str) -> tuple[str, str | None]:
    """``(name, unique_id)`` of a key; the id is ``None`` for a legacy name-only key."""
    match = _KEY.match(key.strip())
    if not match:
        return key.strip(), None
    return match.group("name"), match.group("id")


def resolve_camera_key(wanted: str, cameras: list[dict[str, Any]]) -> str:
    """The key for a camera given by OpenCV index, full key or name, among the attached ``cameras``.

    A name that is not attached is returned as typed, so a setting can still be edited for an unplugged
    camera. A name shared by several attached cameras is an error that lists how to pick one.
    """
    text = wanted.strip()
    keys = {int(c["index"]): camera_key(str(c.get("name", "")), str(c.get("unique_id", ""))) for c in cameras}
    if text.isdigit():
        if int(text) not in keys:
            raise CameraKeyError(f"No camera at index {text}")
        return keys[int(text)]
    if text in keys.values():
        return text
    matching = [
        i for i, c in ((int(c["index"]), c) for c in cameras) if str(c.get("name", "")).strip() == text
    ]
    if len(matching) > 1:
        options = ", ".join(f"index {i} ({keys[i]})" for i in matching)
        raise CameraKeyError(f"{len(matching)} cameras are named '{text}'. Pick one: {options}")
    return keys[matching[0]] if matching else text


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


def index_by_key[V](cameras: list[dict[str, Any]], saved: Mapping[str, V], noun: str) -> dict[int, V]:
    """Map each OpenCV index to the value saved for the camera listed at that index.

    A key with an id applies to the camera with that id. A legacy name-only key applies to the camera with that
    name when exactly one is attached and no id key already covers it; with two the frames could not be told
    apart, so it is skipped. ``noun`` ("Colour gain", "Lens") names the setting in the log lines.
    """
    by_id = {str(c["unique_id"]): int(c["index"]) for c in cameras if c.get("unique_id")}
    by_name: dict[str, list[int]] = {}
    for camera in cameras:
        by_name.setdefault(str(camera.get("name", "")).strip(), []).append(int(camera["index"]))

    mapping: dict[int, V] = {}
    legacy: dict[str, V] = {}
    for key, value in saved.items():
        name, unique_id = split_key(key)
        if unique_id is None:
            legacy[name] = value
        elif unique_id not in by_id:
            logger.info("%s for '%s' not applied: that camera is not attached", noun, key)
        else:
            mapping[by_id[unique_id]] = value
            logger.info("%s for '%s' applies to camera index %d", noun, key, by_id[unique_id])

    for name, value in legacy.items():
        indices = by_name.get(name, [])
        if not indices:
            logger.info("%s for '%s' not applied: that camera is not attached", noun, name)
        elif len(indices) > 1:
            logger.warning(
                "%s for '%s' not applied: %d cameras have that name; calibrate each one again",
                noun,
                name,
                len(indices),
            )
        elif indices[0] in mapping:
            logger.info("%s for '%s' is superseded by the one saved for its port", noun, name)
        else:
            mapping[indices[0]] = value
            logger.info("%s for '%s' applies to camera index %d", noun, name, indices[0])
    return mapping


LISTING_TTL_SECONDS = 3.0
_listing_cache: tuple[float, list[dict[str, Any]]] | None = None


def list_cameras_cached() -> list[dict[str, Any]]:
    """``list_cameras_in_cv2_order`` for the browser's many small requests: one subprocess every few seconds."""
    global _listing_cache
    now = time.monotonic()
    if _listing_cache is None or now - _listing_cache[0] > LISTING_TTL_SECONDS:
        _listing_cache = (now, list_cameras_in_cv2_order())
    return _listing_cache[1]


def browser_settings[V](
    cameras: list[dict[str, Any]], saved: Mapping[str, V], noun: str
) -> tuple[dict[str, V], dict[int, V]]:
    """What the browser needs to show a saved setting: ``(by_name, by_index)``.

    The browser knows a camera by the label it shows, which two identical cameras share, so ``by_name`` holds only
    names that identify a single camera. ``by_index`` is exact: the OpenCV index of every attached camera that has
    a saved setting. With no listing (not macOS) the saved keys are shown by their bare names, again only when each
    name is saved once.
    """
    if not cameras:
        names = Counter(split_key(key)[0] for key in saved)
        return {split_key(key)[0]: v for key, v in saved.items() if names[split_key(key)[0]] == 1}, {}
    by_index = index_by_key(cameras, saved, noun)
    name_of = {int(c["index"]): str(c.get("name", "")).strip() for c in cameras}
    counts = Counter(name_of.values())
    return {name_of[i]: v for i, v in by_index.items() if counts[name_of[i]] == 1}, by_index
