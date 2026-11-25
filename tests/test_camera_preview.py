"""Server-side previews: one reader per camera, corrected like the recording."""

from __future__ import annotations

import numpy as np
import pytest

from lelab import camera_preview as cp, color_gain, lens
from lelab.color_calibration import CalibrationCamera

TOP = CalibrationCamera(1, "USB Camera", (0x0ABD, 0x8050), "0x1300000abd8050", 0x130000)
CAMERAS = [TOP]


class FakeCapture:
    opened: list[int] = []

    def __init__(self, index: int, settings: cp.PreviewSettings) -> None:
        FakeCapture.opened.append(index)
        self.frame = np.full((settings.height, settings.width, 3), 100, np.uint8)

    def read(self):
        return True, self.frame.copy()

    def release(self) -> None:
        pass


def make_hub(busy=lambda: None) -> cp.PreviewHub:
    FakeCapture.opened = []
    return cp.PreviewHub(open_capture=FakeCapture, attached=lambda: CAMERAS, busy=busy)


def test_two_viewers_share_one_reader_and_it_closes_after_the_last() -> None:
    hub = make_hub()
    settings = cp.PreviewSettings(width=64, height=48)
    a, b = hub.stream(TOP.camera_id, settings), hub.stream(TOP.camera_id, settings)
    assert next(a).startswith(b"--frame") and next(b).startswith(b"--frame")
    assert FakeCapture.opened == [1]
    a.close()
    assert hub.open_cameras() == [TOP.camera_id]
    b.close()
    assert hub.open_cameras() == []


def test_a_snapshot_is_lens_then_gain_corrected(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(lens, "apply_lens", lambda f, model: calls.append("lens") or f)
    monkeypatch.setattr(color_gain, "apply_gains", lambda f, gains: calls.append("gain") or f + 1)
    lens.save_lens(TOP.key, {"fov": 120.0, "bend": 0.0, "zoom": 1.0})
    color_gain.save_gains(TOP.key, {"blue": 1.0, "green": 1.0, "red": 1.0})
    frame = make_hub().snapshot(TOP.camera_id, cp.PreviewSettings(width=64, height=48))
    assert frame.shape == (48, 64, 3) and frame[0, 0, 0] == 101
    assert calls[:2] == ["lens", "gain"]


def test_switched_off_corrections_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lens, "apply_lens", lambda f, model: pytest.fail("lens applied"))
    lens.save_lens(TOP.key, {"fov": 120.0, "bend": 0.0, "zoom": 1.0})
    make_hub().snapshot(TOP.camera_id, cp.PreviewSettings(width=64, height=48, lens=False))


def test_previews_are_refused_while_a_session_owns_the_cameras() -> None:
    hub = make_hub(busy=lambda: "Recording is using the cameras")
    with pytest.raises(cp.PreviewBusyError, match="Recording"):
        hub.check(TOP.camera_id)


def test_close_all_releases_every_reader() -> None:
    hub = make_hub()
    stream = hub.stream(TOP.camera_id, cp.PreviewSettings(width=64, height=48))
    next(stream)
    hub.close_all()
    assert hub.open_cameras() == []


def test_a_camera_that_delivers_another_size_is_reported() -> None:
    hub = cp.PreviewHub(
        open_capture=lambda i, s: FakeCapture(i, cp.PreviewSettings(width=1280, height=720)),
        attached=lambda: CAMERAS,
        busy=lambda: None,
    )
    with pytest.raises(cp.PreviewUnavailableError, match="1280×720.*640×480"):
        hub.snapshot(TOP.camera_id, cp.PreviewSettings())


def test_preview_settings_follow_the_profile_camera() -> None:
    settings = cp.PreviewSettings.from_camera(
        {"width": 1280, "height": 720, "fps": 15, "fourcc": "MJPG", "color_correction": False}
    )
    assert settings == cp.PreviewSettings(1280, 720, 15, "MJPG", lens=True, gain=False)


def test_preview_endpoints_answer_with_status_codes(client, monkeypatch: pytest.MonkeyPatch) -> None:
    from lelab import record

    assert client.get("/camera-preview/0x9990000abd8050").status_code == 404
    monkeypatch.setattr(record, "recording_active", True)
    r = client.get("/camera-snapshot/index:0")
    assert r.status_code == 409 and "Recording" in r.json()["message"]


def test_available_cameras_lists_what_the_colour_and_lens_tools_list(client) -> None:
    body = client.get("/available-cameras").json()
    assert body["status"] == "success"
    assert body["cameras"][0] == {
        "camera_id": "index:0",
        "index": 0,
        "name": "Camera 0",
        "usb_id": None,
        "label": "Camera 0 (no USB controls, index 0)",
        "key": "Camera 0",
        "lens": None,
        "gain": None,
    }
    assert client.get("/camera-lenses").status_code in (404, 405)
