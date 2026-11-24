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

"""Runs the libusb transport against a real camera. Skipped unless a USB id is given:

    LELAB_UVC_LIVE=0abd:8050 uv run pytest tests/test_uvc_live.py -q

The camera's white balance is restored afterwards.
"""

from __future__ import annotations

import os

import pytest

from lelab import camera_controls
from lelab.utils import uvc

LIVE = os.environ.get("LELAB_UVC_LIVE", "")

pytestmark = pytest.mark.skipif(not LIVE, reason="set LELAB_UVC_LIVE=<vid:pid> to test against a camera")

WHITE_BALANCE = ("white_balance_temperature_auto", "white_balance_temperature")


def test_the_camera_is_listed_with_its_white_balance_control() -> None:
    vid, pid = camera_controls.parse_device_id(LIVE)
    found = {(c.vid, c.pid): c for c in uvc.list_cameras()}
    assert (vid, pid) in found
    assert "white_balance_temperature" in found[(vid, pid)].controls


def test_white_balance_is_written_and_read_back_and_survives_reopening() -> None:
    vid, pid = camera_controls.parse_device_id(LIVE)
    with uvc.open_camera(vid, pid) as camera:
        original = {name: camera.read(name) for name in WHITE_BALANCE}
        low = camera.describe("white_balance_temperature")["min"]
        target = low + 200  # distinct from the floor, so a no-op write cannot pass
        try:
            applied = camera.apply({"white_balance_temperature_auto": 0, "white_balance_temperature": target})
            assert applied == {"white_balance_temperature_auto": 0, "white_balance_temperature": target}

            with uvc.open_camera(vid, pid) as again:  # a new handle sees the value the camera kept
                assert again.read("white_balance_temperature") == target
                assert again.read("white_balance_temperature_auto") == 0
        finally:
            camera.apply(original)


def test_out_of_range_values_are_clamped_by_the_camera_limits() -> None:
    vid, pid = camera_controls.parse_device_id(LIVE)
    with uvc.open_camera(vid, pid) as camera:
        original = {name: camera.read(name) for name in WHITE_BALANCE}
        try:
            camera.apply({"white_balance_temperature_auto": 0})
            limits = camera.describe("white_balance_temperature")
            assert camera.write("white_balance_temperature", limits["max"] + 5000) == limits["max"]
        finally:
            camera.apply(original)
