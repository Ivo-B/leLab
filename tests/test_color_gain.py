"""Per-channel camera colour gain: the math, saved per camera name, and the hook into LeRobot's frame path."""

from __future__ import annotations

import json
import types

import numpy as np
import pytest

from lelab import color_gain
from lelab.scripts import camera_controls as cli

WHITE_PAPER = {"blue": 1.7868, "green": 1.0, "red": 1.4464}


@pytest.fixture(autouse=True)
def _no_gain_left_installed():
    yield
    color_gain.install({})


# -- validation and naming --------------------------------------------------------------------------


def test_gains_need_exactly_the_three_channels() -> None:
    with pytest.raises(color_gain.ColorGainError, match="exactly the gains"):
        color_gain.validate_gains({"blue": 1.0, "red": 1.0})
    with pytest.raises(color_gain.ColorGainError, match="exactly the gains"):
        color_gain.validate_gains({**WHITE_PAPER, "alpha": 1.0})


@pytest.mark.parametrize("bad", [0, -1.0, 4.5, float("nan"), float("inf"), True, "1.5", None])
def test_gain_values_must_be_sensible_multipliers(bad) -> None:
    with pytest.raises(color_gain.ColorGainError):
        color_gain.validate_gains({**WHITE_PAPER, "red": bad})


def test_names_ignore_the_stray_spaces_the_os_adds() -> None:
    assert color_gain.normalize_name("Innomaker-U20CAM-720P  ") == "Innomaker-U20CAM-720P"
    with pytest.raises(color_gain.ColorGainError):
        color_gain.normalize_name("   ")


# -- saved per name ---------------------------------------------------------------------------------


def test_save_then_load_then_forget() -> None:
    color_gain.save_gains("USB Camera  ", WHITE_PAPER)
    assert color_gain.load_saved() == {"USB Camera": WHITE_PAPER}

    assert color_gain.forget_gains("USB Camera") is True
    assert color_gain.load_saved() == {}
    assert color_gain.forget_gains("USB Camera") is False


def test_a_bad_entry_in_the_file_is_ignored_not_fatal(tmp_path, monkeypatch) -> None:
    path = tmp_path / "gains.json"
    path.write_text(json.dumps({"USB Camera": WHITE_PAPER, "Broken": {"red": 0}}))
    monkeypatch.setattr("lelab.utils.config.CAMERA_GAINS_FILE", str(path))
    assert color_gain.load_saved() == {"USB Camera": WHITE_PAPER}


def test_an_unreadable_file_means_no_gains(tmp_path, monkeypatch) -> None:
    path = tmp_path / "gains.json"
    path.write_text("{not json")
    monkeypatch.setattr("lelab.utils.config.CAMERA_GAINS_FILE", str(path))
    assert color_gain.load_saved() == {}


# -- the arithmetic ---------------------------------------------------------------------------------


def test_gains_apply_per_bgr_channel_and_clip() -> None:
    frame = np.zeros((1, 3, 3), dtype=np.uint8)
    frame[0, 0] = (100, 100, 100)  # B, G, R
    frame[0, 1] = (200, 200, 200)
    frame[0, 2] = (10, 240, 0)

    out = color_gain.apply_gains(frame, WHITE_PAPER)

    assert out.dtype == np.uint8 and out.shape == frame.shape
    assert tuple(out[0, 0]) == (179, 100, 145)  # blue x1.79, red x1.45
    assert tuple(out[0, 1]) == (255, 200, 255)  # clipped, not wrapped
    assert tuple(out[0, 2]) == (18, 240, 0)


def test_the_measured_white_paper_comes_out_neutral() -> None:
    # Median paper pixel from the 2800 K frame, BGR: blue 136, green 243, red 168.
    paper = np.full((1, 1, 3), (136, 243, 168), dtype=np.uint8)
    out = color_gain.apply_gains(paper, WHITE_PAPER)
    assert tuple(out[0, 0]) == (243, 243, 243)


# -- matching to the current indices ----------------------------------------------------------------

CAMERAS = [
    {"index": 0, "name": "USB Camera", "unique_id": "a"},
    {"index": 1, "name": "Innomaker-U20CAM-720P  ", "unique_id": "b"},
    {"index": 2, "name": "MacBook Pro Camera", "unique_id": "c"},
]


def test_a_saved_name_maps_to_the_index_it_is_attached_at() -> None:
    mapping = color_gain.index_gains(CAMERAS, {"USB Camera": WHITE_PAPER})
    assert mapping == {0: WHITE_PAPER}


def test_the_index_follows_the_camera_when_the_order_changes() -> None:
    # After a replug the listing reports the camera at a new index.
    reordered = [{**CAMERAS[2], "index": 0}, {**CAMERAS[0], "index": 1}]
    assert color_gain.index_gains(reordered, {"USB Camera": WHITE_PAPER}) == {1: WHITE_PAPER}


def test_a_camera_that_is_not_attached_gets_no_gain() -> None:
    assert color_gain.index_gains(CAMERAS[1:], {"USB Camera": WHITE_PAPER}) == {}


def test_two_cameras_with_the_same_name_are_skipped_rather_than_guessed() -> None:
    twins = [{"index": 0, "name": "USB Camera"}, {"index": 1, "name": "USB Camera"}]
    assert color_gain.index_gains(twins, {"USB Camera": WHITE_PAPER}) == {}


def test_no_saved_gains_means_no_listing_and_no_gains(monkeypatch) -> None:
    called = []
    monkeypatch.setattr(color_gain, "list_cameras_in_cv2_order", lambda: called.append(1) or CAMERAS)
    monkeypatch.setattr(color_gain.platform, "system", lambda: "Darwin")
    assert color_gain.gains_for_attached_cameras() == {}
    assert called == []


def test_gains_are_only_looked_up_on_macos(monkeypatch) -> None:
    color_gain.save_gains("USB Camera", WHITE_PAPER)
    monkeypatch.setattr(color_gain.platform, "system", lambda: "Linux")
    assert color_gain.gains_for_attached_cameras() == {}


def test_a_camera_can_switch_its_colour_correction_off(monkeypatch) -> None:
    monkeypatch.setattr(
        color_gain,
        "gains_for_attached_cameras",
        lambda: {0: WHITE_PAPER, 1: {"blue": 1.0, "green": 1.0, "red": 1.0}},
    )
    session = {"top": {"camera_index": 0, "color_correction": False}, "wrist": {"camera_index": 1}}
    assert color_gain.gains_for_session(session) == {1: {"blue": 1.0, "green": 1.0, "red": 1.0}}


def test_a_camera_that_omits_the_flag_keeps_its_correction(monkeypatch) -> None:
    monkeypatch.setattr(color_gain, "gains_for_attached_cameras", lambda: {0: WHITE_PAPER})
    assert color_gain.gains_for_session({"top": {"camera_index": 0}}) == {0: WHITE_PAPER}


def test_attached_gains_come_from_the_macos_listing(monkeypatch) -> None:
    color_gain.save_gains("USB Camera", WHITE_PAPER)
    monkeypatch.setattr(color_gain.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(color_gain, "list_cameras_in_cv2_order", lambda: CAMERAS)
    assert color_gain.gains_for_attached_cameras() == {0: WHITE_PAPER}


# -- the hook into LeRobot --------------------------------------------------------------------------


def _camera_stub(index: int, color_mode, shape: tuple[int, int]) -> types.SimpleNamespace:
    height, width = shape
    return types.SimpleNamespace(
        index_or_path=index,
        color_mode=color_mode,
        capture_height=height,
        capture_width=width,
        rotation=None,
    )


def test_install_corrects_only_the_camera_at_the_mapped_index() -> None:
    from lerobot.cameras.configs import ColorMode
    from lerobot.cameras.opencv import OpenCVCamera

    color_gain.install({0: WHITE_PAPER})
    frame = np.full((2, 2, 3), (136, 243, 168), dtype=np.uint8)  # BGR

    corrected = OpenCVCamera._postprocess_image(_camera_stub(0, ColorMode.BGR, (2, 2)), frame)
    untouched = OpenCVCamera._postprocess_image(_camera_stub(1, ColorMode.BGR, (2, 2)), frame)

    assert tuple(corrected[0, 0]) == (243, 243, 243)
    assert tuple(untouched[0, 0]) == (136, 243, 168)


def test_the_correction_runs_before_colour_conversion_to_rgb() -> None:
    from lerobot.cameras.configs import ColorMode
    from lerobot.cameras.opencv import OpenCVCamera

    color_gain.install({0: {"blue": 2.0, "green": 1.0, "red": 1.0}})
    frame = np.full((1, 1, 3), (10, 20, 30), dtype=np.uint8)  # BGR: blue is 10

    rgb = OpenCVCamera._postprocess_image(_camera_stub(0, ColorMode.RGB, (1, 1)), frame)

    assert tuple(rgb[0, 0]) == (30, 20, 20)  # blue doubled, then converted to RGB order


def test_install_replaces_the_map_and_an_empty_map_turns_it_off() -> None:
    from lerobot.cameras.configs import ColorMode
    from lerobot.cameras.opencv import OpenCVCamera

    frame = np.full((1, 1, 3), (100, 100, 100), dtype=np.uint8)
    color_gain.install({0: WHITE_PAPER})
    color_gain.install({})
    out = OpenCVCamera._postprocess_image(_camera_stub(0, ColorMode.BGR, (1, 1)), frame)
    assert tuple(out[0, 0]) == (100, 100, 100)


def test_the_environment_round_trip_keeps_the_index_map() -> None:
    mapping = {0: WHITE_PAPER, 3: {"blue": 1.0, "green": 1.0, "red": 1.2}}
    assert color_gain.decode_from_env(color_gain.encode_for_env(mapping)) == mapping


def test_the_rollout_wrapper_installs_the_env_map_before_lerobot_runs(monkeypatch) -> None:
    import lerobot.scripts.lerobot_rollout as lerobot_rollout
    from lelab.scripts import rollout_gain

    seen = {}
    monkeypatch.setenv(color_gain.ENV_VAR, color_gain.encode_for_env({0: WHITE_PAPER}))
    monkeypatch.setattr(color_gain, "install", lambda mapping: seen.setdefault("installed", mapping))
    monkeypatch.setattr(lerobot_rollout, "main", lambda: seen.setdefault("ran", True))

    rollout_gain.main()

    assert seen == {"installed": {0: WHITE_PAPER}, "ran": True}


# -- HTTP and CLI -----------------------------------------------------------------------------------


def test_the_gains_endpoint_returns_the_saved_gains(client, monkeypatch) -> None:
    monkeypatch.setattr(color_gain, "list_cameras_cached", lambda: CAMERAS)
    monkeypatch.setattr(color_gain.platform, "system", lambda: "Darwin")
    color_gain.save_gains("USB Camera", WHITE_PAPER)
    body = client.get("/camera-gains").json()
    assert body == {"status": "success", "gains": {"USB Camera": WHITE_PAPER}, "by_index": {"0": WHITE_PAPER}}


def test_the_gains_endpoint_keeps_two_cameras_with_one_name_apart(client, monkeypatch) -> None:
    twins = [
        {"index": 0, "name": "USB Camera", "unique_id": "0x11300000abd8050"},
        {"index": 1, "name": "USB Camera", "unique_id": "0x11410000abd8050"},
    ]
    other = {"blue": 1.2, "green": 1.0, "red": 1.1}
    monkeypatch.setattr(color_gain, "list_cameras_cached", lambda: twins)
    monkeypatch.setattr(color_gain.platform, "system", lambda: "Darwin")
    color_gain.save_gains("USB Camera [0x11300000abd8050]", WHITE_PAPER)
    color_gain.save_gains("USB Camera [0x11410000abd8050]", other)
    body = client.get("/camera-gains").json()
    # The browser's label is shared, so by name it can say nothing; by index it is exact.
    assert body["gains"] == {}
    assert body["by_index"] == {"0": WHITE_PAPER, "1": other}


def test_the_gains_endpoint_without_a_listing_shows_bare_names(client, monkeypatch) -> None:
    monkeypatch.setattr(color_gain.platform, "system", lambda: "Linux")
    color_gain.save_gains("USB Camera [0x11300000abd8050]", WHITE_PAPER)
    body = client.get("/camera-gains").json()
    assert body["gains"] == {"USB Camera": WHITE_PAPER} and body["by_index"] == {}


def test_cli_sets_shows_and_forgets_a_gain(capsys) -> None:
    assert cli.main(["gain", "USB Camera", "--blue", "1.7868", "--red", "1.4464"]) == cli.EXIT_OK
    assert color_gain.load_saved() == {"USB Camera": {"blue": 1.7868, "green": 1.0, "red": 1.4464}}

    assert cli.main(["gain"]) == cli.EXIT_OK
    assert "USB Camera: blue=1.7868 green=1.0000 red=1.4464" in capsys.readouterr().out

    assert cli.main(["gain", "USB Camera", "--forget"]) == cli.EXIT_OK
    assert color_gain.load_saved() == {}


def test_cli_rejects_a_gain_without_red_or_blue(capsys) -> None:
    assert cli.main(["gain", "USB Camera", "--blue", "1.7"]) == cli.EXIT_INVALID
    assert "Give --blue and --red" in capsys.readouterr().err
    assert color_gain.load_saved() == {}


def test_cli_rejects_a_non_positive_gain(capsys) -> None:
    assert cli.main(["gain", "USB Camera", "--blue", "0", "--red", "1.4"]) == cli.EXIT_INVALID
    assert "must be above 0" in capsys.readouterr().err


def test_cli_forget_leaves_a_name_only_gain_that_two_cameras_share(monkeypatch, capsys) -> None:
    twins = [
        {"index": 0, "name": "USB Camera", "unique_id": "0x11300000abd8050"},
        {"index": 1, "name": "USB Camera", "unique_id": "0x11410000abd8050"},
    ]
    monkeypatch.setattr(cli, "list_cameras_in_cv2_order", lambda: twins)
    color_gain.save_gains("USB Camera", WHITE_PAPER)
    assert cli.main(["gain", "1", "--forget"]) == cli.EXIT_OK
    assert "No colour gain saved" in capsys.readouterr().out
    assert color_gain.load_saved() == {"USB Camera": WHITE_PAPER}


def test_cli_sets_a_gain_for_one_of_two_cameras_by_index(monkeypatch, capsys) -> None:
    twins = [
        {"index": 0, "name": "USB Camera", "unique_id": "0x11300000abd8050"},
        {"index": 1, "name": "USB Camera", "unique_id": "0x11410000abd8050"},
    ]
    monkeypatch.setattr(cli, "list_cameras_in_cv2_order", lambda: twins)
    assert cli.main(["gain", "USB Camera", "--blue", "1.5", "--red", "1.3"]) == cli.EXIT_INVALID
    assert "Pick one" in capsys.readouterr().err
    assert cli.main(["gain", "1", "--blue", "1.5", "--red", "1.3"]) == cli.EXIT_OK
    assert list(color_gain.load_saved()) == ["USB Camera [0x11410000abd8050]"]
