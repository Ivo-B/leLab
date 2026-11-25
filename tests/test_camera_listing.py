"""Identifying cameras by their AVFoundation id, so two cameras with one name stay apart."""

from __future__ import annotations

import logging

import pytest

from lelab import color_gain
from lelab.utils import camera_listing as cl

TOP = {"index": 0, "name": "USB Camera", "unique_id": "0x11300000abd8050"}
SIDE = {"index": 1, "name": "USB Camera", "unique_id": "0x11410000abd8050"}
OTHER = {"index": 2, "name": "Innomaker-U20CAM-720P  ", "unique_id": "0x11440000c456367"}
TWINS = [TOP, SIDE, OTHER]

TOP_KEY = "USB Camera [0x11300000abd8050]"
SIDE_KEY = "USB Camera [0x11410000abd8050]"

GAIN_TOP = {"blue": 1.8, "green": 1.0, "red": 1.4}
GAIN_SIDE = {"blue": 1.2, "green": 1.0, "red": 1.1}


def test_a_key_is_the_name_and_the_unique_id() -> None:
    assert cl.camera_key(" USB Camera ", "0x11300000abd8050") == TOP_KEY
    assert cl.camera_key("USB Camera", "") == "USB Camera"


def test_a_key_splits_back_into_name_and_id() -> None:
    assert cl.split_key(TOP_KEY) == ("USB Camera", "0x11300000abd8050")
    assert cl.split_key("USB Camera") == ("USB Camera", None)
    assert cl.split_key("Cam [a] [b]") == ("Cam [a]", "b")


def test_the_usb_location_is_the_digits_before_vid_and_pid() -> None:
    assert cl.usb_location_from_unique_id("0x11300000abd8050") == 0x1130000
    assert cl.usb_location_from_unique_id("0x11410000abd8050") == 0x1141000
    assert cl.usb_location_from_unique_id("6C707041-05AC-0011-0008-000000000001") is None


def test_two_cameras_with_one_name_each_get_their_own_saved_value() -> None:
    saved = {TOP_KEY: GAIN_TOP, SIDE_KEY: GAIN_SIDE}
    assert color_gain.index_gains(TWINS, saved) == {0: GAIN_TOP, 1: GAIN_SIDE}


def test_the_value_follows_the_camera_when_the_index_changes() -> None:
    swapped = [{**SIDE, "index": 0}, {**TOP, "index": 1}]
    assert color_gain.index_gains(swapped, {TOP_KEY: GAIN_TOP}) == {1: GAIN_TOP}


def test_a_key_for_a_camera_on_another_port_is_not_applied() -> None:
    moved = {**TOP, "unique_id": "0x11500000abd8050"}
    assert color_gain.index_gains([moved], {TOP_KEY: GAIN_TOP}) == {}


def test_a_legacy_name_applies_when_one_camera_has_that_name() -> None:
    assert color_gain.index_gains([TOP, OTHER], {"USB Camera": GAIN_TOP}) == {0: GAIN_TOP}


def test_a_legacy_name_is_skipped_when_two_cameras_have_it(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert color_gain.index_gains(TWINS, {"USB Camera": GAIN_TOP}) == {}
    assert "2 cameras have that name" in caplog.text


def test_an_id_entry_wins_over_a_legacy_name_entry() -> None:
    saved = {"USB Camera": GAIN_SIDE, TOP_KEY: GAIN_TOP}
    assert color_gain.index_gains([TOP, OTHER], saved) == {0: GAIN_TOP}


def test_a_legacy_name_still_covers_the_twin_without_an_id_entry() -> None:
    # Only one of the two cameras is attached, so the old name-only entry is unambiguous again.
    assert color_gain.index_gains([SIDE], {"USB Camera": GAIN_SIDE, TOP_KEY: GAIN_TOP}) == {1: GAIN_SIDE}


# -- turning what the user typed into a key ----------------------------------------------------------


def test_a_unique_name_resolves_to_the_attached_cameras_key() -> None:
    assert (
        cl.resolve_camera_key("Innomaker-U20CAM-720P", TWINS) == "Innomaker-U20CAM-720P [0x11440000c456367]"
    )


def test_an_index_resolves_to_that_cameras_key() -> None:
    assert cl.resolve_camera_key("1", TWINS) == SIDE_KEY


def test_a_full_key_resolves_to_itself() -> None:
    assert cl.resolve_camera_key(SIDE_KEY, TWINS) == SIDE_KEY


def test_a_shared_name_asks_for_an_index() -> None:
    with pytest.raises(cl.CameraKeyError, match="2 cameras are named 'USB Camera'.*index 0.*index 1"):
        cl.resolve_camera_key("USB Camera", TWINS)


def test_an_unknown_name_is_kept_as_typed() -> None:
    assert cl.resolve_camera_key("Some Camera", TWINS) == "Some Camera"


def test_an_index_that_is_not_attached_is_an_error() -> None:
    with pytest.raises(cl.CameraKeyError, match="No camera at index 9"):
        cl.resolve_camera_key("9", TWINS)
