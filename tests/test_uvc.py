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

"""Tests for lelab.utils.uvc: descriptor parsing and the control protocol against a fake camera.

The descriptors in `uvc_fixtures` were read from the two cameras on the demo Mac. The libusb transport is
covered by `test_uvc_live.py`, which needs the hardware.
"""

from __future__ import annotations

import pytest

from lelab.utils import uvc

from .uvc_fixtures import IMX335_TOP_CONFIG, INNOMAKER_WRIST_CONFIG

PU = 2  # processing unit id of the IMX335 fixture
CT = 1  # camera terminal id of the IMX335 fixture
WB_TEMP, WB_AUTO, HUE = 0x0A, 0x0B, 0x06


class FakeTransport:
    """A camera in memory: each (unit, selector) holds cur/min/max/res/def and its byte length.

    Controls the test does not describe get a 0..100 range so `reset()` and `describe()` work on all of them.
    """

    def __init__(self) -> None:
        self.state: dict[tuple[int, int], dict] = {}
        self.writes: list[tuple[int, int, bytes]] = []

    def define(self, unit: int, selector: int, *, length: int, signed: bool = False, **values: int) -> None:
        base = {"cur": 0, "min": 0, "max": 100, "res": 1, "def": 0}
        self.state[(unit, selector)] = {**base, **values, "len": length, "signed": signed}

    def _entry(self, unit: int, selector: int) -> dict:
        return self.state.setdefault(
            (unit, selector),
            {"cur": 0, "min": 0, "max": 100, "res": 1, "def": 0, "len": 2, "signed": False},
        )

    def control_in(self, request_type: int, request: int, value: int, index: int, length: int) -> bytes:
        entry = self._entry(index >> 8, value >> 8)
        if request == uvc.GET_INFO:
            return bytes([0x03])
        key = {
            uvc.GET_CUR: "cur",
            uvc.GET_MIN: "min",
            uvc.GET_MAX: "max",
            uvc.GET_RES: "res",
            uvc.GET_DEF: "def",
        }[request]
        return int(entry[key]).to_bytes(entry["len"], "little", signed=entry["signed"])[:length]

    def control_out(self, request_type: int, request: int, value: int, index: int, data: bytes) -> None:
        assert request == uvc.SET_CUR
        unit, selector = index >> 8, value >> 8
        entry = self._entry(unit, selector)
        entry["cur"] = int.from_bytes(data, "little", signed=entry["signed"])
        self.writes.append((unit, selector, bytes(data)))


def make_camera() -> tuple[uvc.UvcCamera, FakeTransport]:
    transport = FakeTransport()
    transport.define(PU, WB_TEMP, length=2, min=2800, max=6500, res=10, cur=4600, **{"def": 4600})
    transport.define(PU, WB_AUTO, length=1, min=0, max=1, cur=1, **{"def": 1})
    transport.define(PU, HUE, length=2, signed=True, min=-180, max=180, cur=-10, **{"def": -10})
    transport.define(CT, 0x02, length=1, min=0, max=255, cur=8, **{"def": 8})  # exposure_auto
    transport.define(CT, 0x04, length=4, min=50, max=10000, res=10, cur=166, **{"def": 166})
    info = uvc.parse_video_control(IMX335_TOP_CONFIG)
    assert info is not None
    return uvc.UvcCamera(transport, info, 0x0ABD, 0x8050, "USB Camera"), transport


def test_parses_the_waveshare_imx335_descriptor() -> None:
    info = uvc.parse_video_control(IMX335_TOP_CONFIG)
    assert info is not None
    assert (info.interface, info.processing_unit, info.camera_terminal) == (0, PU, CT)
    assert info.pu_controls == b"\x7f\x15"
    assert info.ct_controls == b"\x0e\x2a\x00"


def test_parses_the_innomaker_wrist_descriptor() -> None:
    info = uvc.parse_video_control(INNOMAKER_WRIST_CONFIG)
    assert info is not None
    assert (info.interface, info.processing_unit, info.camera_terminal) == (0, PU, CT)
    assert info.pu_controls == b"\x7f\x17"


def test_a_configuration_without_a_video_control_interface_is_not_a_camera() -> None:
    assert uvc.parse_video_control(IMX335_TOP_CONFIG[:9]) is None
    assert uvc.parse_video_control(b"") is None


def test_a_truncated_descriptor_does_not_raise() -> None:
    for cut in (20, 60, 120, 200):
        uvc.parse_video_control(IMX335_TOP_CONFIG[:cut])


def test_supported_controls_follow_the_descriptor_bitmaps() -> None:
    camera, _ = make_camera()
    supported = camera.supported()
    assert {"white_balance_temperature", "white_balance_temperature_auto", "hue", "saturation"} <= set(
        supported
    )
    assert {"exposure_auto", "exposure_time_absolute"} <= set(supported)
    assert "gain" not in supported  # bit 9 of the IMX335's processing-unit bitmap is clear


def test_read_decodes_signed_and_unsigned_values() -> None:
    camera, _ = make_camera()
    assert camera.read("white_balance_temperature") == 4600
    assert camera.read("hue") == -10


def test_write_returns_the_value_the_camera_reports_back() -> None:
    camera, transport = make_camera()
    assert camera.write("white_balance_temperature", 2800) == 2800
    assert transport.writes[-1] == (PU, WB_TEMP, (2800).to_bytes(2, "little"))


def test_write_encodes_negative_values_as_twos_complement() -> None:
    camera, transport = make_camera()
    assert camera.write("hue", -60) == -60
    assert transport.writes[-1][2] == b"\xc4\xff"


def test_write_clamps_range_controls_to_the_camera_limits() -> None:
    camera, _ = make_camera()
    assert camera.write("white_balance_temperature", 9000) == 6500
    assert camera.write("white_balance_temperature", 100) == 2800


def test_apply_switches_the_auto_control_off_before_writing_its_dependent() -> None:
    camera, transport = make_camera()
    applied = camera.apply({"white_balance_temperature": 2800, "white_balance_temperature_auto": 0})
    assert applied == {"white_balance_temperature_auto": 0, "white_balance_temperature": 2800}
    assert [selector for _, selector, _ in transport.writes] == [WB_AUTO, WB_TEMP]


def test_apply_rejects_unsupported_controls_before_writing_anything() -> None:
    camera, transport = make_camera()
    with pytest.raises(uvc.UnsupportedControlError, match="gain"):
        camera.apply({"white_balance_temperature_auto": 0, "gain": 5})
    with pytest.raises(uvc.UnsupportedControlError, match="no_such_control"):
        camera.apply({"no_such_control": 1})
    assert transport.writes == []


def test_describe_reports_range_default_and_current_value() -> None:
    camera, _ = make_camera()
    assert camera.describe("white_balance_temperature") == {
        "name": "white_balance_temperature",
        "kind": "range",
        "value": 4600,
        "default": 4600,
        "min": 2800,
        "max": 6500,
        "step": 10,
        "writable": True,
    }
    assert camera.describe("white_balance_temperature_auto")["kind"] == "bool"


def test_reset_restores_defaults_with_auto_modes_dropped_first_and_restored_last() -> None:
    camera, transport = make_camera()
    camera.apply({"white_balance_temperature_auto": 0, "white_balance_temperature": 2800, "hue": 90})
    transport.writes.clear()

    restored = camera.reset()

    assert restored["white_balance_temperature_auto"] == 1
    assert restored["white_balance_temperature"] == 4600
    assert restored["hue"] == -10
    # The first writes put every auto control into manual mode, the last restore the auto defaults.
    manual_gate_writes = [w for w in transport.writes[:2] if w[1] in (WB_AUTO, 0x02)]
    assert len(manual_gate_writes) == 2
    assert transport.writes[-2][1] in (WB_AUTO, 0x02) and transport.writes[-1][1] in (WB_AUTO, 0x02)
    assert transport.state[(PU, WB_AUTO)]["cur"] == 1
    assert transport.state[(CT, 0x02)]["cur"] == 8


def test_every_public_call_is_unavailable_off_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lelab.utils.uvc.platform.system", lambda: "Linux")
    with pytest.raises(uvc.UvcUnavailableError, match="macOS"):
        uvc.list_cameras()
    with pytest.raises(uvc.UvcUnavailableError, match="macOS"), uvc.open_camera(0x0ABD, 0x8050):
        pass


# -- two identical cameras ----------------------------------------------------------------------------


class _Descriptor:
    bDeviceClass = 0xEF  # noqa: N815 - libusb field names
    iProduct = 2  # noqa: N815

    def __init__(self, vid: int, pid: int) -> None:
        self.idVendor, self.idProduct = vid, pid


def _two_identical_cameras(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Two IMX335s on different ports. Returns the devices that were opened, in order."""
    from contextlib import contextmanager

    opened: list[int] = []
    first, second = 0x01130000, 0x01141000

    @contextmanager
    def devices(vid=None, pid=None):
        yield [(first, _Descriptor(0x0ABD, 0x8050), first), (second, _Descriptor(0x0ABD, 0x8050), second)]

    @contextmanager
    def transport(device):
        opened.append(device)
        fake = FakeTransport()
        fake.config_descriptor = lambda: IMX335_TOP_CONFIG  # type: ignore[attr-defined]
        fake.product_name = lambda index: "USB Camera"  # type: ignore[attr-defined]
        yield fake

    monkeypatch.setattr(uvc, "_usb_devices", devices)
    monkeypatch.setattr(uvc, "_open_transport", transport)
    return opened


def test_open_camera_picks_the_device_at_the_given_location(monkeypatch: pytest.MonkeyPatch) -> None:
    opened = _two_identical_cameras(monkeypatch)
    with uvc.open_camera(0x0ABD, 0x8050, 0x01141000):
        pass
    with uvc.open_camera(0x0ABD, 0x8050, 0x01130000):
        pass
    assert opened == [0x01141000, 0x01130000]


def test_open_camera_will_not_guess_between_identical_cameras(monkeypatch: pytest.MonkeyPatch) -> None:
    opened = _two_identical_cameras(monkeypatch)
    with (
        pytest.raises(
            uvc.CameraNotFoundError, match="2 cameras 0abd:8050.*0abd:8050@1130000.*0abd:8050@1141000"
        ),
        uvc.open_camera(0x0ABD, 0x8050),
    ):
        pass
    assert opened == []


def test_open_camera_reports_a_location_with_no_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    _two_identical_cameras(monkeypatch)
    with (
        pytest.raises(uvc.CameraNotFoundError, match="at 1150000"),
        uvc.open_camera(0x0ABD, 0x8050, 0x01150000),
    ):
        pass
