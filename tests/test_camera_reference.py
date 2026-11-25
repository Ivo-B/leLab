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

"""Tests for lelab.camera_reference and its /robots/{name}/camera-reference endpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from lelab import camera_reference
from lelab.server import app

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


@pytest.fixture
def references_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "camera_references"
    monkeypatch.setattr("lelab.utils.config.CAMERA_REFERENCES_PATH", str(path))
    return path


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_save_then_load_round_trips_and_overwrites(references_dir: Path) -> None:
    assert camera_reference.load_reference("arm", "front") is None
    camera_reference.save_reference("arm", "front", PNG)
    assert camera_reference.load_reference("arm", "front") == PNG
    newer = PNG + b"\x01"
    camera_reference.save_reference("arm", "front", newer)
    assert camera_reference.load_reference("arm", "front") == newer
    assert not list(references_dir.rglob("*.tmp"))


def test_save_rejects_non_png_and_oversized(references_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(camera_reference.InvalidReferenceError):
        camera_reference.save_reference("arm", "front", b"not a png")
    monkeypatch.setattr(camera_reference, "MAX_REFERENCE_BYTES", 10)
    with pytest.raises(camera_reference.InvalidReferenceError):
        camera_reference.save_reference("arm", "front", PNG)
    assert not references_dir.exists()


@pytest.mark.parametrize(
    "robot,camera", [("..", "front"), ("arm", "../front"), ("a/b", "front"), ("arm", "")]
)
def test_names_cannot_escape_the_references_dir(references_dir: Path, robot: str, camera: str) -> None:
    with pytest.raises(camera_reference.InvalidReferenceError):
        camera_reference.save_reference(robot, camera, PNG)


def test_delete_reports_whether_anything_was_removed(references_dir: Path) -> None:
    assert camera_reference.delete_reference("arm", "front") is False
    camera_reference.save_reference("arm", "front", PNG)
    assert camera_reference.delete_reference("arm", "front") is True
    assert camera_reference.load_reference("arm", "front") is None


@pytest.fixture
def robots_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "robots"
    monkeypatch.setattr("lelab.utils.config.ROBOTS_PATH", str(path))
    return path


def _save_arm(**camera: object) -> None:
    from lelab.utils.config import save_robot_record

    base = {"id": "c1", "name": "top", "type": "opencv", "width": 64, "height": 48}
    save_robot_record("arm", {"name": "arm", "cameras": [{**base, **camera}]})


def test_endpoints_get_delete(references_dir: Path, client: TestClient) -> None:
    url = "/robots/arm/camera-reference/front"
    assert client.get(url).status_code == 404
    camera_reference.save_reference("arm", "front", PNG)
    got = client.get(url)
    assert got.status_code == 200
    assert got.headers["content-type"] == "image/png"
    assert got.content == PNG
    assert client.delete(url).status_code == 200
    assert client.delete(url).status_code == 404


def test_a_browser_frame_can_no_longer_be_uploaded(references_dir: Path, client: TestClient) -> None:
    assert client.put("/robots/arm/camera-reference/front", content=PNG).status_code == 405


def test_capture_saves_the_server_snapshot_of_the_profile_camera(
    references_dir: Path, robots_dir: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lelab import camera_preview

    _save_arm(camera_id="0x1300000abd8050", lens_correction=False)
    seen: dict = {}

    def snapshot(camera_id: str, settings: camera_preview.PreviewSettings) -> np.ndarray:
        seen.update(camera_id=camera_id, settings=settings)
        return np.zeros((48, 64, 3), np.uint8)

    monkeypatch.setattr(camera_preview.hub, "snapshot", snapshot)
    r = client.post("/robots/arm/camera-reference/top/capture")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert seen["camera_id"] == "0x1300000abd8050"
    assert (seen["settings"].width, seen["settings"].lens, seen["settings"].gain) == (64, False, True)
    assert client.get("/robots/arm/camera-reference/top").content == r.content


def test_capture_asks_to_pick_a_camera_without_an_id(
    references_dir: Path, robots_dir: Path, client: TestClient
) -> None:
    _save_arm()
    r = client.post("/robots/arm/camera-reference/top/capture")
    assert r.status_code == 400 and "Pick the camera" in r.json()["message"]


def test_capture_of_an_unplugged_camera_is_404(
    references_dir: Path, robots_dir: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lelab import camera_preview

    _save_arm(camera_id="0x1300000abd8050")
    monkeypatch.setattr(camera_preview.hub, "_attached", lambda: [])
    monkeypatch.setattr(camera_preview.hub, "_busy", lambda: None)
    r = client.post("/robots/arm/camera-reference/top/capture")
    assert r.status_code == 404 and "not attached" in r.json()["message"]
