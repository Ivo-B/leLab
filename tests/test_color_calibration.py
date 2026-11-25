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

from types import SimpleNamespace

import numpy as np
import pytest

from lelab import color_calibration as cc, color_gain
from lelab.utils.camera_listing import CameraKeyError, usb_id_from_unique_id

LISTING = [
    {"index": 0, "name": "MacBook Pro Camera", "unique_id": "6C707041-05AC-0011-0008-000000000001"},
    {"index": 1, "name": "USB Camera", "unique_id": "0x11300000abd8050"},
    {"index": 2, "name": "Innomaker-U20CAM-720P  ", "unique_id": "0x11410000c456367"},
]
TWINS = [
    {"index": 0, "name": "USB Camera", "unique_id": "0x11300000abd8050"},
    {"index": 1, "name": "USB Camera", "unique_id": "0x11410000abd8050"},
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


class BrightCamera(FakeCamera):
    """A white sheet at `base` levels (green), moved up or down one level per brightness unit, clipped at 255.

    At `boost_at` Kelvin the picture is `boost` levels brighter, as a white-balance change can shift brightness.
    """

    def __init__(self, base: float, boost_at: int | None = None, boost: float = 80.0) -> None:
        super().__init__()
        self.base, self.boost_at, self.boost = base, boost_at, boost
        self.level = 0
        self.kelvin = 4000  # starts neutral, as under the camera's own automatic white balance

    def read(self) -> np.ndarray:
        self.reads += 1
        green = self.base + self.level + (self.boost if self.kelvin == self.boost_at else 0.0)
        factor = self.kelvin / 4000
        return _frame(green / factor, green, green * factor)


class FakeBrightness:
    def __init__(self, camera: BrightCamera, low: int = -64, high: int = 64, step: int = 1) -> None:
        self.camera, self.low, self.high, self.step = camera, low, high, step
        self.started = 0
        self.restored = 0
        self.sets: list[int] = []
        self.saves: list[int] = []

    def range(self) -> tuple[int, int, int]:
        return self.low, self.high, self.step

    def start(self) -> int:
        self.started += 1
        return self.camera.level

    def restore(self) -> None:
        self.restored += 1

    def set(self, value: int) -> int:
        self.sets.append(value)
        self.camera.level = value
        return value

    def save(self, value: int) -> int:
        self.saves.append(value)
        return value


class FakeWhiteBalance:
    def __init__(self, camera: FakeCamera) -> None:
        self.camera = camera
        self.sets: list[int] = []
        self.saves: list[int] = []
        self.remembered = 0
        self.restored = 0

    def range(self) -> tuple[int, int, int]:
        return 2800, 6500, 10

    def remember(self) -> None:
        self.remembered += 1

    def restore(self) -> None:
        self.restored += 1

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
    assert cameras[0].device == "0abd:8050@1130000"
    assert cameras[0].key == "USB Camera [0x11300000abd8050]"
    assert cameras[1].name == "Innomaker-U20CAM-720P"
    assert cameras[2].device is None
    assert "no USB controls" in cameras[2].label
    assert cameras[0].label == "USB Camera (USB 0abd:8050@1130000, index 1)"


def test_two_cameras_with_one_name_have_different_devices_and_labels() -> None:
    top, side = cc.list_calibration_cameras(TWINS)
    assert top.name == side.name
    assert top.device == "0abd:8050@1130000" and side.device == "0abd:8050@1141000"
    assert top.key != side.key
    assert "0abd:8050@1130000" in top.label and "0abd:8050@1141000" in side.label


def test_find_camera_asks_for_an_index_when_two_cameras_share_a_name() -> None:
    cameras = cc.list_calibration_cameras(TWINS)
    with pytest.raises(CameraKeyError, match="2 cameras are named 'USB Camera'.*index 0.*index 1"):
        cc.find_camera(cameras, "USB Camera")
    assert cc.find_camera(cameras, "1").device == "0abd:8050@1141000"
    assert cc.find_camera(cameras, cameras[1].key) is cameras[1]


def test_each_twin_keeps_its_own_saved_gain_and_forgets_only_its_own() -> None:
    top, side = cc.list_calibration_cameras(TWINS)
    color_gain.save_gains(top.key, {"blue": 1.8, "green": 1.0, "red": 1.4})
    color_gain.save_gains(side.key, {"blue": 1.2, "green": 1.0, "red": 1.1})
    assert cc.saved_gains_for(top, [top, side])["blue"] == 1.8
    assert cc.saved_gains_for(side, [top, side])["blue"] == 1.2
    cc.forget_gain_for(top)
    assert cc.saved_gains_for(top, [top, side]) is None
    assert cc.saved_gains_for(side, [top, side])["blue"] == 1.2


def test_a_name_only_gain_is_not_used() -> None:
    (camera,) = cc.list_calibration_cameras(TWINS[:1])
    color_gain.save_gains("USB Camera", {"blue": 1.8, "green": 1.0, "red": 1.4})
    assert cc.saved_gains_for(camera, [camera]) is None


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
    gains = calibrator.save(result, "USB Camera [0x11300000abd8050]")
    assert white_balance.saves == [4030]
    assert color_gain.load_saved() == {"USB Camera [0x11300000abd8050]": gains}


# -- brightness -------------------------------------------------------------------------------------

BOX = (0, 0, 80, 60)


def _calibrator(camera: BrightCamera, brightness: FakeBrightness | None = None, wb: bool = True):
    return cc.Calibrator(
        camera,
        FakeWhiteBalance(camera) if wb else None,
        brightness=brightness,
        settle_frames=0,
        sample_frames=1,
    )


def test_a_clipped_sheet_is_brought_down_with_the_brightness_and_the_exposure_is_left_alone() -> None:
    camera = BrightCamera(base=260)  # clipped everywhere at brightness 0
    brightness = FakeBrightness(camera)
    result = _calibrator(camera, brightness).run(BOX)
    assert result.brightness == camera.level < 0
    assert result.before.clipped <= 0.01
    assert brightness.started == 1


def test_a_dim_sheet_is_raised_toward_the_target() -> None:
    camera = BrightCamera(base=70)
    brightness = FakeBrightness(camera)
    result = _calibrator(camera, brightness).run(BOX)
    assert result.brightness is not None and result.brightness > 0
    assert cc.GREEN_LOW <= result.before.green <= cc.GREEN_HIGH


def test_a_sheet_that_clips_while_being_raised_settles_below_the_clipping_point() -> None:
    camera = BrightCamera(base=100)
    # Above 150 the sheet clips; a raise to the target (140) is fine, so the fit stops there without oscillating.
    brightness = FakeBrightness(camera)
    result = _calibrator(camera, brightness).run(BOX)
    assert 100 < result.before.green <= 150
    assert len(brightness.sets) <= cc.FIT_STEPS * 2


def test_a_sheet_already_in_range_keeps_the_brightness_it_has() -> None:
    camera = BrightCamera(base=140)
    brightness = FakeBrightness(camera)
    result = _calibrator(camera, brightness).run(BOX)
    assert brightness.sets == [] and result.brightness == 0


def test_the_brightness_stays_inside_the_cameras_range_and_then_the_sheet_is_refused() -> None:
    camera = BrightCamera(base=900)  # clipped even with the brightness all the way down
    brightness = FakeBrightness(camera)
    with pytest.raises(cc.CalibrationError, match="brightness turned all the way down"):
        _calibrator(camera, brightness).run(BOX)
    assert camera.level == -64
    assert min(brightness.sets) >= -64


def test_the_brightness_is_fitted_again_after_the_white_balance_changes_the_picture() -> None:
    camera = BrightCamera(base=140, boost_at=4030, boost=90)  # the best temperature is 90 levels brighter
    brightness = FakeBrightness(camera)
    result = _calibrator(camera, brightness).run(BOX)
    assert result.temperature == 4030
    assert result.brightness is not None and result.brightness < 0
    assert result.before.clipped <= 0.01 and result.before.green <= cc.GREEN_HIGH


def test_a_sheet_too_dim_even_at_full_brightness_still_measures_but_warns() -> None:
    camera = BrightCamera(base=40)  # at most 104 with +64
    result = _calibrator(camera, FakeBrightness(camera)).run(BOX)
    assert result.brightness == 64
    assert any("dim" in warning and "Add light" in warning for warning in result.warnings)


def test_a_failed_calibration_puts_the_camera_back() -> None:
    camera = BrightCamera(base=-100)  # black even at full brightness
    brightness = FakeBrightness(camera)
    white_balance = FakeWhiteBalance(camera)
    calibrator = cc.Calibrator(camera, white_balance, brightness=brightness, settle_frames=0, sample_frames=1)
    with pytest.raises(cc.CalibrationError, match="too dark"):
        calibrator.run(BOX)
    assert white_balance.remembered == 1
    assert brightness.restored == 1 and white_balance.restored == 1


def test_a_good_calibration_leaves_the_camera_calibrated_until_restore_is_called() -> None:
    camera = BrightCamera(base=260)
    brightness, white_balance = FakeBrightness(camera), FakeWhiteBalance(camera)
    calibrator = cc.Calibrator(camera, white_balance, brightness=brightness, settle_frames=0, sample_frames=1)
    calibrator.run(BOX)
    assert brightness.restored == 0 and white_balance.restored == 0
    calibrator.restore()
    assert brightness.restored == 1 and white_balance.restored == 1


def test_save_keeps_the_brightness_with_the_temperature_and_gain() -> None:
    camera = BrightCamera(base=260)
    brightness, white_balance = FakeBrightness(camera), FakeWhiteBalance(camera)
    calibrator = cc.Calibrator(camera, white_balance, brightness=brightness, settle_frames=0, sample_frames=1)
    result = calibrator.run(BOX)
    calibrator.save(result, "USB Camera [0x11300000abd8050]")
    assert brightness.saves == [result.brightness] and white_balance.saves == [4030]


def test_without_a_brightness_control_a_clipped_sheet_asks_for_less_light() -> None:
    camera = FakeCamera(base=300)
    with pytest.raises(cc.CalibrationError, match="Dim the light or lower the exposure"):
        cc.Calibrator(camera, None, settle_frames=0, sample_frames=1).run(BOX)


class _UsbCamera:
    """The two calls `UvcBrightness` makes on an open camera."""

    def __init__(self, values: dict[str, int]) -> None:
        self.values = values

    def read(self, name: str) -> int:
        return self.values[name]

    def describe(self, name: str) -> dict[str, int]:
        return {"min": -64, "max": 64, "step": 1}


@pytest.fixture
def usb_brightness(monkeypatch: pytest.MonkeyPatch):
    from contextlib import contextmanager

    state = SimpleNamespace(camera=_UsbCamera({"brightness": 0}), writes=[])

    @contextmanager
    def open_camera(vid, pid, location=None):
        yield state.camera

    def set_controls(device, controls, save=False):
        state.writes.append((dict(controls), save))
        return dict(controls)

    monkeypatch.setattr(cc.uvc, "open_camera", open_camera)
    monkeypatch.setattr(cc.camera_controls, "set_controls", set_controls)
    return state


def test_the_usb_brightness_leaves_the_exposure_mode_alone_and_restores_the_old_value(usb_brightness) -> None:
    usb_brightness.camera.values["brightness"] = 5
    brightness = cc.UvcBrightness("0abd:8050@1130000")
    assert brightness.start() == 5
    brightness.set(-8)
    brightness.save(-8)
    brightness.restore()
    assert usb_brightness.writes == [
        ({"brightness": -8}, False),
        ({"brightness": -8}, True),
        ({"brightness": 5}, False),
    ]


def test_restore_without_start_writes_nothing(usb_brightness) -> None:
    cc.UvcBrightness("0abd:8050@1130000").restore()
    assert usb_brightness.writes == []


class YellowStartCamera(FakeCamera):
    """A white sheet that the camera's automatic white balance leaves almost blue-less until it is swept."""

    def read(self) -> np.ndarray:
        self.reads += 1
        if self.kelvin == 6500:  # where the camera starts
            return _frame(5, 150, 190)
        return super().read()


def test_a_sheet_the_automatic_white_balance_left_blue_less_is_still_calibrated() -> None:
    camera = YellowStartCamera()
    result = cc.Calibrator(camera, FakeWhiteBalance(camera), settle_frames=0, sample_frames=1).run(BOX)
    assert result.temperature == 4030
    assert result.after.blue_ratio == pytest.approx(1, abs=0.05)


def test_without_a_white_balance_control_a_blue_less_box_is_still_refused() -> None:
    camera = YellowStartCamera()
    with pytest.raises(cc.CalibrationError, match="strongly coloured"):
        cc.Calibrator(camera, None, settle_frames=0, sample_frames=1).run(BOX)


def test_a_dim_but_neutral_sheet_is_not_called_coloured() -> None:
    cc.check_exposure(cc.PatchStats(blue=48, green=70, red=62, clipped=0, spread=0))


def test_a_box_that_stays_strongly_coloured_after_the_sweep_is_refused() -> None:
    with pytest.raises(cc.CalibrationError, match="strongly coloured"):
        cc.gains_from_patch(cc.PatchStats(blue=20, green=150, red=200, clipped=0, spread=0))


def test_describe_result_mentions_the_brightness() -> None:
    camera = BrightCamera(base=140)
    result = _calibrator(camera, FakeBrightness(camera)).run(BOX)
    assert any(line.startswith("Brightness +0") for line in cc.describe_result(result))


@pytest.fixture
def fake_hardware(monkeypatch: pytest.MonkeyPatch) -> FakeWhiteBalance:
    camera = FakeCamera()
    camera.close = lambda: None  # type: ignore[attr-defined]
    white_balance = FakeWhiteBalance(camera)
    monkeypatch.setattr(cc, "list_cameras_in_cv2_order", lambda: LISTING)
    monkeypatch.setattr(cc, "OpenCVFrameSource", lambda index: camera)
    monkeypatch.setattr(cc, "white_balance_for", lambda found: white_balance if found.device else None)
    monkeypatch.setattr(cc, "brightness_for", lambda found: None)
    return white_balance


def test_cli_calibrate_saves_white_balance_and_gain(
    fake_hardware: FakeWhiteBalance, capsys: pytest.CaptureFixture[str]
) -> None:
    from lelab.scripts import camera_controls as cli

    assert cli.main(["calibrate"]) == 0
    out = capsys.readouterr().out
    assert "USB Camera (USB 0abd:8050@1130000, index 1)" in out and "4030 K" in out
    assert fake_hardware.saves == [4030]
    assert set(color_gain.load_saved()) == {"USB Camera [0x11300000abd8050]"}


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
