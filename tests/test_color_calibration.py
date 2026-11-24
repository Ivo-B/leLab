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

from __future__ import annotations

import numpy as np
import pytest

from lelab import color_calibration as cc, color_gain
from lelab.utils.camera_listing import usb_id_from_unique_id

LISTING = [
    {"index": 0, "name": "MacBook Pro Camera", "unique_id": "6C707041-05AC-0011-0008-000000000001"},
    {"index": 1, "name": "USB Camera", "unique_id": "0x11300000abd8050"},
    {"index": 2, "name": "Innomaker-U20CAM-720P  ", "unique_id": "0x11410000c456367"},
]


def _frame(blue: float, green: float, red: float, size: tuple[int, int] = (60, 80)) -> np.ndarray:
    frame = np.empty((*size, 3), dtype=np.uint8)
    frame[:] = np.clip(np.rint([blue, green, red]), 0, 255).astype(np.uint8)
    return frame


class FakeCamera:
    """A white sheet seen through a camera whose cast follows its white-balance setting; neutral at 4000 K."""

    def __init__(self, base: float = 150.0) -> None:
        self.base = base
        self.kelvin = 6500
        self.reads = 0

    def read(self) -> np.ndarray:
        self.reads += 1
        factor = self.kelvin / 4000
        return _frame(self.base / factor, self.base, self.base * factor)


class FakeWhiteBalance:
    def __init__(self, camera: FakeCamera) -> None:
        self.camera = camera
        self.sets: list[int] = []
        self.saves: list[int] = []

    def range(self) -> tuple[int, int, int]:
        return 2800, 6500, 10

    def set(self, kelvin: int) -> int:
        self.sets.append(kelvin)
        self.camera.kelvin = kelvin
        return kelvin

    def save(self, kelvin: int) -> int:
        self.saves.append(kelvin)
        self.camera.kelvin = kelvin
        return kelvin


# -- cameras ----------------------------------------------------------------------------------------


def test_usb_id_from_unique_id() -> None:
    assert usb_id_from_unique_id("0x11300000abd8050") == (0x0ABD, 0x8050)
    assert usb_id_from_unique_id("0x11410000c456367") == (0x0C45, 0x6367)
    assert usb_id_from_unique_id("6C707041-05AC-0011-0008-000000000001") is None
    assert usb_id_from_unique_id("0x1234") is None


def test_list_calibration_cameras_puts_usb_cameras_first() -> None:
    cameras = cc.list_calibration_cameras(LISTING)
    assert [camera.index for camera in cameras] == [1, 2, 0]
    assert cameras[0].device == "0abd:8050"
    assert cameras[1].name == "Innomaker-U20CAM-720P"
    assert cameras[2].device is None
    assert "no USB controls" in cameras[2].label


def test_find_camera() -> None:
    cameras = cc.list_calibration_cameras(LISTING)
    assert cc.find_camera(cameras, None).name == "USB Camera"
    assert cc.find_camera(cameras, "0").name == "MacBook Pro Camera"
    assert cc.find_camera(cameras, " Innomaker-U20CAM-720P ").index == 2
    with pytest.raises(cc.CalibrationError, match="'USB Camera' \\(1\\)"):
        cc.find_camera(cameras, "Logitech")
    with pytest.raises(cc.CalibrationError, match="No cameras"):
        cc.find_camera([], None)


# -- measuring --------------------------------------------------------------------------------------


def test_default_and_clamped_roi() -> None:
    assert cc.default_roi(640, 480) == (224, 168, 192, 144)
    assert cc.clamp_roi((600, -5, 100, 50), 640, 480) == (600, 0, 40, 50)


def test_measure_patch_reads_only_the_box() -> None:
    frame = _frame(10, 10, 10, size=(100, 100))
    frame[40:60, 40:60] = (100, 150, 200)
    stats = cc.measure_patch([frame, frame], (40, 40, 20, 20))
    assert (stats.blue, stats.green, stats.red) == (100, 150, 200)
    assert stats.clipped == 0 and stats.spread == 0
    assert stats.red_ratio == pytest.approx(200 / 150)


def test_measure_patch_counts_clipped_and_uneven_pixels() -> None:
    frame = _frame(200, 200, 200, size=(10, 10))
    frame[:5] = (255, 120, 120)
    stats = cc.measure_patch([frame], (0, 0, 10, 10))
    assert stats.clipped == pytest.approx(0.5)
    assert stats.spread > 0.25


def test_measure_patch_needs_frames() -> None:
    with pytest.raises(cc.CalibrationError):
        cc.measure_patch([], (0, 0, 1, 1))


def test_neutrality() -> None:
    assert cc.neutrality(cc.PatchStats(120, 120, 120, 0, 0)) == 0
    assert cc.neutrality(cc.PatchStats(60, 120, 240, 0, 0)) == pytest.approx(2 * np.log(2))


def test_gains_from_patch() -> None:
    gains, warnings = cc.gains_from_patch(cc.PatchStats(blue=80, green=120, red=100, clipped=0, spread=0.05))
    assert gains == {"blue": 1.5, "green": 1.0, "red": 1.2}
    assert warnings == []


@pytest.mark.parametrize(
    ("stats", "message"),
    [
        (cc.PatchStats(30, 40, 35, 0, 0), "too dark"),
        (cc.PatchStats(20, 150, 200, 0, 0), "white sheet"),
        (cc.PatchStats(250, 255, 250, 0.4, 0), "overexposed"),
        (cc.PatchStats(51, 240, 200, 0, 0), "too strong"),
    ],
)
def test_gains_from_patch_refuses_bad_patches(stats: cc.PatchStats, message: str) -> None:
    with pytest.raises(cc.CalibrationError, match=message):
        cc.gains_from_patch(stats)


def test_gains_from_patch_warns_about_doubtful_patches() -> None:
    _, warnings = cc.gains_from_patch(cc.PatchStats(120, 120, 120, clipped=0.05, spread=0.4))
    assert len(warnings) == 2


def test_temperature_steps() -> None:
    assert cc.temperature_steps(2800, 6500, 10, 7) == [2800, 3420, 4030, 4650, 5270, 5880, 6500]
    assert cc.temperature_steps(2800, 2800, 10, 7) == [2800]


# -- calibration ------------------------------------------------------------------------------------


def test_run_sweeps_to_the_most_neutral_temperature_without_saving() -> None:
    camera = FakeCamera()
    white_balance = FakeWhiteBalance(camera)
    seen, progress = [], []
    calibrator = cc.Calibrator(
        camera, white_balance, settle_frames=1, sample_frames=2, on_frame=lambda frame: seen.append(frame)
    )
    result = calibrator.run((10, 10, 20, 20), progress=lambda fraction, message: progress.append(fraction))

    assert result.temperature == 4030
    assert len(result.sweep) == 7
    assert white_balance.sets[-1] == 4030 and white_balance.saves == []
    assert result.after.red_ratio == pytest.approx(1, abs=0.01)
    assert result.after.blue_ratio == pytest.approx(1, abs=0.01)
    assert progress == sorted(progress) and progress[-1] == 1.0
    assert len(seen) == camera.reads


def test_run_without_white_balance_only_measures_the_gain() -> None:
    camera = FakeCamera()
    camera.kelvin = 5000
    result = cc.Calibrator(camera, None, settle_frames=1, sample_frames=2).run((0, 0, 80, 60))
    assert result.temperature is None and result.sweep == []
    assert result.gains["red"] == pytest.approx(0.8, abs=0.01)
    assert result.after.red_ratio == pytest.approx(1, abs=0.01)
    assert any("no USB white-balance" in warning for warning in result.warnings)


def test_run_refuses_a_dark_patch_before_sweeping() -> None:
    camera = FakeCamera(base=20)
    white_balance = FakeWhiteBalance(camera)
    with pytest.raises(cc.CalibrationError, match="too dark"):
        cc.Calibrator(camera, white_balance, settle_frames=0, sample_frames=1).run((0, 0, 10, 10))
    assert white_balance.sets == []


def test_save_keeps_temperature_and_gain() -> None:
    camera = FakeCamera()
    white_balance = FakeWhiteBalance(camera)
    calibrator = cc.Calibrator(camera, white_balance, settle_frames=0, sample_frames=1)
    result = calibrator.run((0, 0, 80, 60))
    gains = calibrator.save(result, "USB Camera ")
    assert white_balance.saves == [4030]
    assert color_gain.load_saved() == {"USB Camera": gains}


@pytest.fixture
def fake_hardware(monkeypatch: pytest.MonkeyPatch) -> FakeWhiteBalance:
    camera = FakeCamera()
    camera.close = lambda: None  # type: ignore[attr-defined]
    white_balance = FakeWhiteBalance(camera)
    monkeypatch.setattr(cc, "list_cameras_in_cv2_order", lambda: LISTING)
    monkeypatch.setattr(cc, "OpenCVFrameSource", lambda index: camera)
    monkeypatch.setattr(cc, "white_balance_for", lambda found: white_balance if found.device else None)
    return white_balance


def test_cli_calibrate_saves_white_balance_and_gain(
    fake_hardware: FakeWhiteBalance, capsys: pytest.CaptureFixture[str]
) -> None:
    from lelab.scripts import camera_controls as cli

    assert cli.main(["calibrate"]) == 0
    out = capsys.readouterr().out
    assert "USB Camera (USB 0abd:8050, index 1)" in out and "4030 K" in out
    assert fake_hardware.saves == [4030]
    assert set(color_gain.load_saved()) == {"USB Camera"}


def test_cli_calibrate_dry_run_and_no_sweep(
    fake_hardware: FakeWhiteBalance, capsys: pytest.CaptureFixture[str]
) -> None:
    from lelab.scripts import camera_controls as cli

    assert cli.main(["calibrate", "1", "--no-sweep", "--dry-run", "--roi", "0,0,40,30"]) == 0
    assert "Not saved" in capsys.readouterr().out
    assert fake_hardware.sets == [] and fake_hardware.saves == []
    assert color_gain.load_saved() == {}


def test_cli_calibrate_exit_codes(
    fake_hardware: FakeWhiteBalance, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from lelab.scripts import camera_controls as cli

    assert cli.main(["calibrate", "--roi", "1,2,3"]) == 2
    assert cli.main(["calibrate", "Logitech"]) == 4
    fake_hardware.camera.base = 20
    assert cli.main(["calibrate", "--no-sweep"]) == 6
    assert "too dark" in capsys.readouterr().err


def test_describe_result_mentions_temperature_and_gain() -> None:
    camera = FakeCamera()
    result = cc.Calibrator(camera, FakeWhiteBalance(camera), settle_frames=0, sample_frames=1).run(
        (0, 0, 80, 60)
    )
    text = "\n".join(cc.describe_result(result))
    assert "4030 K" in text and "Gain: blue" in text
