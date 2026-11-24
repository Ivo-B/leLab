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

"""Tests for lelab.camera_controls, its /camera-controls endpoints and the lelab-camera command.

USB access is replaced by a fake camera built from the real IMX335 descriptor (see test_uvc.py).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lelab import camera_controls
from lelab.scripts import camera_controls as cli
from lelab.server import app
from lelab.utils import uvc

from .test_uvc import PU, WB_AUTO, WB_TEMP, make_camera

TOP = "0abd:8050"


@pytest.fixture
def controls_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "camera_controls.json"
    monkeypatch.setattr("lelab.utils.config.CAMERA_CONTROLS_FILE", str(path))
    return path


@pytest.fixture
def usb(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """One attached camera (the IMX335) behind the `uvc` module's public entry points."""
    camera, transport = make_camera()
    state = SimpleNamespace(camera=camera, transport=transport, attached=True, opened=0)

    @contextmanager
    def open_camera(vid: int, pid: int) -> Iterator[uvc.UvcCamera]:
        state.opened += 1
        if (vid, pid) != (0x0ABD, 0x8050) or not state.attached:
            raise uvc.CameraNotFoundError(f"No UVC camera {vid:04x}:{pid:04x} is attached")
        yield camera

    monkeypatch.setattr(uvc, "open_camera", open_camera)
    monkeypatch.setattr(
        uvc,
        "list_cameras",
        lambda: [uvc.UvcCameraInfo(0x0ABD, 0x8050, "USB Camera", camera.supported())],
    )
    return state


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# -- ids and validation -----------------------------------------------------------------------------


def test_device_id_is_lowercase_hex_vid_pid() -> None:
    assert camera_controls.device_id(0x0ABD, 0x8050) == "0abd:8050"
    assert camera_controls.parse_device_id("0ABD:8050") == (0x0ABD, 0x8050)


@pytest.mark.parametrize("text", ["", "0abd", "0abd:", "xyz:1234", "0abd:8050:1", "12345:1"])
def test_parse_device_id_rejects_malformed_ids(text: str) -> None:
    with pytest.raises(camera_controls.CameraControlsError):
        camera_controls.parse_device_id(text)


def test_validate_controls_rejects_unknown_names_and_non_integers() -> None:
    assert camera_controls.validate_controls({"hue": -10, "gamma": 300}) == {"hue": -10, "gamma": 300}
    for bad in ({"nope": 1}, {"hue": "a lot"}, {"hue": 1.5}, {"hue": True}, {}):
        with pytest.raises(camera_controls.CameraControlsError):
            camera_controls.validate_controls(bad)


# -- persistence ------------------------------------------------------------------------------------


def test_nothing_saved_yet(controls_file: Path) -> None:
    assert camera_controls.load_saved() == {}
    assert not controls_file.exists()


def test_set_controls_applies_and_saves_the_values_the_camera_reports(
    controls_file: Path, usb: SimpleNamespace
) -> None:
    applied = camera_controls.set_controls(
        TOP, {"white_balance_temperature_auto": 0, "white_balance_temperature": 1000}, save=True
    )
    # 1000 K is below the camera's floor: the clamped value is what gets applied and saved.
    assert applied == {"white_balance_temperature_auto": 0, "white_balance_temperature": 2800}
    assert json.loads(controls_file.read_text()) == {TOP: applied}
    assert not list(controls_file.parent.glob("*.tmp"))


def test_saving_again_merges_with_what_was_saved(controls_file: Path, usb: SimpleNamespace) -> None:
    camera_controls.set_controls(TOP, {"white_balance_temperature_auto": 0}, save=True)
    camera_controls.set_controls(TOP, {"hue": -60}, save=True)
    assert camera_controls.load_saved() == {TOP: {"white_balance_temperature_auto": 0, "hue": -60}}


def test_set_controls_without_save_leaves_no_file(controls_file: Path, usb: SimpleNamespace) -> None:
    camera_controls.set_controls(TOP, {"hue": 20})
    assert usb.transport.state[(PU, 0x06)]["cur"] == 20
    assert not controls_file.exists()


def test_a_failed_apply_saves_nothing(controls_file: Path, usb: SimpleNamespace) -> None:
    with pytest.raises(uvc.UnsupportedControlError):
        camera_controls.set_controls(TOP, {"gain": 5}, save=True)
    assert not controls_file.exists()


def test_a_corrupt_file_reads_as_nothing_saved(controls_file: Path) -> None:
    controls_file.write_text("{not json")
    assert camera_controls.load_saved() == {}


def test_set_controls_on_a_camera_that_is_not_attached(controls_file: Path, usb: SimpleNamespace) -> None:
    usb.attached = False
    with pytest.raises(uvc.CameraNotFoundError):
        camera_controls.set_controls(TOP, {"hue": 20}, save=True)
    assert not controls_file.exists()


# -- applying saved settings ------------------------------------------------------------------------


def test_apply_saved_with_nothing_saved_never_touches_usb(controls_file: Path, usb: SimpleNamespace) -> None:
    assert camera_controls.apply_saved() == {}
    assert usb.opened == 0


def test_apply_saved_restores_the_saved_values(controls_file: Path, usb: SimpleNamespace) -> None:
    controls_file.write_text(json.dumps({TOP: {"white_balance_temperature_auto": 0, "hue": 40}}))
    result = camera_controls.apply_saved()
    assert result == {TOP: {"white_balance_temperature_auto": 0, "hue": 40}}
    assert usb.transport.state[(PU, WB_AUTO)]["cur"] == 0
    assert usb.transport.state[(PU, 0x06)]["cur"] == 40


def test_apply_saved_reports_an_absent_camera_and_carries_on(
    controls_file: Path, usb: SimpleNamespace
) -> None:
    controls_file.write_text(json.dumps({"1234:5678": {"hue": 1}, TOP: {"white_balance_temperature": 3000}}))
    result = camera_controls.apply_saved()
    assert result[TOP] == {"white_balance_temperature": 3000}
    assert isinstance(result["1234:5678"], str) and "attached" in result["1234:5678"]
    assert usb.transport.state[(PU, WB_TEMP)]["cur"] == 3000


def test_apply_saved_ignores_a_malformed_entry(controls_file: Path, usb: SimpleNamespace) -> None:
    controls_file.write_text(json.dumps({"garbage": {"hue": 1}, TOP: {"hue": 5}}))
    result = camera_controls.apply_saved()
    assert result[TOP] == {"hue": 5}
    assert isinstance(result["garbage"], str)


# -- reset ------------------------------------------------------------------------------------------


def test_reset_restores_camera_defaults_and_forgets_the_saved_values(
    controls_file: Path, usb: SimpleNamespace
) -> None:
    camera_controls.set_controls(TOP, {"white_balance_temperature_auto": 0, "hue": 90}, save=True)
    restored = camera_controls.reset_controls(TOP)
    assert restored["white_balance_temperature_auto"] == 1
    assert restored["hue"] == -10
    assert camera_controls.load_saved() == {}


def test_reset_can_keep_the_saved_values(controls_file: Path, usb: SimpleNamespace) -> None:
    camera_controls.set_controls(TOP, {"hue": 90}, save=True)
    camera_controls.reset_controls(TOP, forget=False)
    assert camera_controls.load_saved() == {TOP: {"hue": 90}}


# -- listing ----------------------------------------------------------------------------------------


def test_list_cameras_describes_each_control_and_the_saved_values(
    controls_file: Path, usb: SimpleNamespace
) -> None:
    camera_controls.set_controls(TOP, {"hue": 30}, save=True)
    cameras = camera_controls.list_cameras()
    assert len(cameras) == 1
    top = cameras[0]
    assert top["id"] == TOP and top["name"] == "USB Camera"
    assert top["saved"] == {"hue": 30}
    by_name = {c["name"]: c for c in top["controls"]}
    assert by_name["white_balance_temperature"]["min"] == 2800
    assert by_name["hue"]["value"] == 30


# -- HTTP endpoints ---------------------------------------------------------------------------------


def test_get_camera_controls(client: TestClient, controls_file: Path, usb: SimpleNamespace) -> None:
    body = client.get("/camera-controls").json()
    assert body["status"] == "success" and body["available"] is True
    assert body["cameras"][0]["id"] == TOP
    assert body["saved"] == {}


def test_get_camera_controls_reports_unavailable_instead_of_failing(
    client: TestClient, controls_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable() -> None:
        raise uvc.UvcUnavailableError("only on macOS")

    monkeypatch.setattr(uvc, "list_cameras", unavailable)
    body = client.get("/camera-controls").json()
    assert body["available"] is False and body["cameras"] == []
    assert "macOS" in body["message"]


def test_put_camera_controls_applies_and_optionally_saves(
    client: TestClient, controls_file: Path, usb: SimpleNamespace
) -> None:
    payload = {"controls": {"white_balance_temperature_auto": 0, "white_balance_temperature": 2900}}
    response = client.put(f"/camera-controls/{TOP}", json={**payload, "save": True})
    assert response.status_code == 200
    assert response.json()["controls"] == {
        "white_balance_temperature_auto": 0,
        "white_balance_temperature": 2900,
    }
    assert camera_controls.load_saved()[TOP]["white_balance_temperature"] == 2900


def test_put_camera_controls_error_codes(
    client: TestClient, controls_file: Path, usb: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert client.put("/camera-controls/nonsense", json={"controls": {"hue": 1}}).status_code == 400
    assert client.put(f"/camera-controls/{TOP}", json={"controls": {"nope": 1}}).status_code == 400
    assert client.put(f"/camera-controls/{TOP}", json={"controls": {"gain": 1}}).status_code == 400
    assert client.put("/camera-controls/1234:5678", json={"controls": {"hue": 1}}).status_code == 404

    def unavailable(*_: object, **__: object) -> None:
        raise uvc.UvcUnavailableError("only on macOS")

    monkeypatch.setattr(camera_controls, "set_controls", unavailable)
    assert client.put(f"/camera-controls/{TOP}", json={"controls": {"hue": 1}}).status_code == 503


def test_post_apply_saved(client: TestClient, controls_file: Path, usb: SimpleNamespace) -> None:
    controls_file.write_text(json.dumps({TOP: {"hue": 12}}))
    body = client.post("/camera-controls/apply-saved").json()
    assert body["status"] == "success" and body["applied"] == {TOP: {"hue": 12}}


# -- lelab-camera command ---------------------------------------------------------------------------


def test_cli_set_parses_name_value_pairs_and_saves(
    controls_file: Path, usb: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        ["set", TOP, "white_balance_temperature_auto=0", "white_balance_temperature=2800", "--save"]
    )
    assert code == 0
    assert camera_controls.load_saved()[TOP] == {
        "white_balance_temperature_auto": 0,
        "white_balance_temperature": 2800,
    }
    assert "white_balance_temperature = 2800" in capsys.readouterr().out


def test_cli_exit_codes(controls_file: Path, usb: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    assert cli.main(["set", TOP, "hue"]) == 2  # no value
    assert cli.main(["set", TOP, "hue=abc"]) == 2
    assert cli.main(["set", "nonsense", "hue=1"]) == 2
    assert cli.main(["set", "1234:5678", "hue=1"]) == 4  # not attached

    def unavailable(*_: object, **__: object) -> None:
        raise uvc.UvcUnavailableError("only on macOS")

    monkeypatch.setattr(camera_controls, "set_controls", unavailable)
    assert cli.main(["set", TOP, "hue=1"]) == 3


def test_cli_list_prints_each_camera(
    controls_file: Path, usb: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert TOP in out and "USB Camera" in out and "white_balance_temperature" in out


def test_cli_apply_and_reset(
    controls_file: Path, usb: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    camera_controls.set_controls(TOP, {"hue": 77}, save=True)
    usb.transport.state[(PU, 0x06)]["cur"] = 0  # as after a replug
    assert cli.main(["apply"]) == 0
    assert usb.transport.state[(PU, 0x06)]["cur"] == 77
    assert cli.main(["reset", TOP]) == 0
    assert usb.transport.state[(PU, 0x06)]["cur"] == -10
    assert camera_controls.load_saved() == {}
    capsys.readouterr()
