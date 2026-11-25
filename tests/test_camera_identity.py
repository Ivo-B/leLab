"""One id per physical camera, shared by the web page and the colour and lens tools."""

from __future__ import annotations

import pytest

from lelab import camera_identity, color_gain, lens
from lelab.color_calibration import CalibrationCamera, list_calibration_cameras

LISTING = [
    {"index": 0, "name": "MacBook Pro Camera", "unique_id": "6C707041-05AC-0011-0008-000000000001"},
    {"index": 1, "name": "USB Camera", "unique_id": "0x1300000abd8050"},
    {"index": 2, "name": "USB Camera", "unique_id": "0x1410000abd8050"},
]


def test_twins_get_their_usb_port_in_the_label_and_the_key() -> None:
    cameras = list_calibration_cameras(LISTING)
    top, side = cameras[0], cameras[1]
    assert (top.camera_id, top.label, top.key) == (
        "0x1300000abd8050",
        "USB Camera (USB 0abd:8050@130000, index 1)",
        "USB Camera [0x1300000abd8050]",
    )
    assert side.label == "USB Camera (USB 0abd:8050@141000, index 2)"


def test_a_camera_without_an_os_id_is_known_by_its_index() -> None:
    assert CalibrationCamera(3, "Camera 3").camera_id == "index:3"


def test_describe_carries_the_corrections_saved_under_the_key() -> None:
    top = list_calibration_cameras(LISTING)[0]
    lens.save_lens(top.key, {"fov": 160.0, "bend": 0.0, "zoom": 1.3})
    color_gain.save_gains("USB Camera", {"blue": 1.5, "green": 1.0, "red": 1.2})
    assert camera_identity.describe(top) == {
        "camera_id": "0x1300000abd8050",
        "index": 1,
        "name": "USB Camera",
        "usb_id": "0abd:8050@130000",
        "label": "USB Camera (USB 0abd:8050@130000, index 1)",
        "key": "USB Camera [0x1300000abd8050]",
        "lens": {"fov": 160.0, "bend": 0.0, "zoom": 1.3},
        "gain": None,
    }


def test_a_session_camera_opens_the_index_its_id_has_now() -> None:
    swapped = [{**LISTING[1], "index": 2}, {**LISTING[2], "index": 1}]
    cameras = list_calibration_cameras(swapped)
    resolved = camera_identity.resolve_session_cameras(
        {"top": {"camera_id": "0x1300000abd8050", "width": 640}}, cameras
    )
    assert resolved == {"top": {"camera_id": "0x1300000abd8050", "width": 640, "camera_index": 2}}


def test_a_session_camera_without_an_id_must_be_picked_again() -> None:
    with pytest.raises(ValueError, match="Pick the camera for 'top' again"):
        camera_identity.resolve_session_cameras({"top": {"camera_index": 1}}, [])


def test_an_unplugged_camera_is_reported() -> None:
    with pytest.raises(camera_identity.CameraNotAttachedError, match="same USB port"):
        camera_identity.resolve_session_cameras({"top": {"camera_id": "0x9990000abd8050"}}, [])
