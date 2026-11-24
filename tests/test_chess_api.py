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

from pathlib import Path

import cv2
import pytest
from fastapi.testclient import TestClient

from lelab import camera_reference
from lelab.chess.game import GameState
from lelab.server import app
from lelab.utils import config
from tests.chess_render import calibration, empty_board, render


def png(frame) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()


@pytest.fixture
def client(tmp_lerobot_home) -> TestClient:
    return TestClient(app)


def test_calibration_round_trip_and_errors(client: TestClient) -> None:
    assert client.get("/chess/board-calibration").status_code == 404
    body = calibration().to_dict()
    bad = client.put("/chess/board-calibration", json={**body, "bin": []})
    assert bad.status_code == 400
    assert bad.json()["message"]
    malformed = client.put(
        "/chess/board-calibration", content=b"{", headers={"Content-Type": "application/json"}
    )
    assert malformed.status_code == 400
    assert client.put("/chess/board-calibration", json=body).json() == {"status": "success"}
    assert client.get("/chess/board-calibration").json() == body


def test_board_state_needs_calibration_and_reference(client: TestClient) -> None:
    cal = calibration()
    frame = png(render(cal, GameState().occupancy()))
    missing = client.post("/chess/board-state", content=frame, headers={"Content-Type": "image/png"})
    assert missing.status_code == 409
    client.put("/chess/board-calibration", json=cal.to_dict())
    no_reference = client.post("/chess/board-state", content=frame, headers={"Content-Type": "image/png"})
    assert no_reference.status_code == 409
    assert "reference" in no_reference.json()["message"].lower()


def test_board_state_reports_a_corrupt_calibration(client: TestClient) -> None:
    path = Path(config.BOARD_CALIBRATION_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json")
    frame = png(empty_board(calibration()))
    response = client.post("/chess/board-state", content=frame, headers={"Content-Type": "image/png"})
    assert response.status_code == 500
    assert response.json()["status"] == "error"
    assert response.json()["message"]


def test_board_state_rejects_an_oversized_frame(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(camera_reference, "MAX_REFERENCE_BYTES", 1024)
    response = client.post("/chess/board-state", content=b"x" * 1025, headers={"Content-Type": "image/png"})
    assert response.status_code == 400
    assert "too large" in response.json()["message"]


def test_board_state_reads_the_position(client: TestClient) -> None:
    cal = calibration()
    client.put("/chess/board-calibration", json=cal.to_dict())
    camera_reference.save_reference("arm", "top", png(empty_board(cal)))
    frame = png(render(cal, {**GameState().occupancy(), "bin": "black"}))
    response = client.post("/chess/board-state", content=frame, headers={"Content-Type": "image/png"})
    assert response.status_code == 200
    body = response.json()
    assert body["robot"] == "arm"
    assert body["squares"] == GameState().occupancy()
    assert body["extras"] == {"bin": "black", "park_white": "empty", "park_black": "empty"}
    garbage = client.post("/chess/board-state", content=b"nope", headers={"Content-Type": "image/png"})
    assert garbage.status_code == 400
