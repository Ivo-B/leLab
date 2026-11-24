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
"""Tests for the /sim-generate and /sim-evaluate routes. The heavy functions are patched: no MuJoCo, no arm."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from lelab import server


@pytest.fixture(autouse=True)
def idle_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        server, "_sim_generate_state", {"running": False, "saved": 0, "episodes": 0, "error": None}
    )
    monkeypatch.setattr(server, "_sim_evaluate_state", {"running": False, "report": None, "error": None})


@pytest.fixture
def client() -> TestClient:
    return TestClient(server.app)


def wait_until_idle(client: TestClient, path: str) -> dict[str, Any]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = client.get(path).json()
        if not status["running"]:
            return status
        time.sleep(0.01)
    raise AssertionError(f"{path} still running")


def blocking_generate(release: threading.Event, calls: list[tuple]):
    def generate_dataset(repo_id, episodes, **kwargs):
        calls.append((repo_id, episodes, kwargs))
        kwargs["progress"](2, episodes, 3)
        release.wait(5)
        return SimpleNamespace(saved=episodes, root=Path("/tmp/unused"), dropped={})

    return generate_dataset


def test_status_is_idle_before_any_job(client: TestClient) -> None:
    assert client.get("/sim-generate").json() == {"running": False, "saved": 0, "episodes": 0, "error": None}
    assert client.get("/sim-evaluate").json() == {"running": False, "report": None, "error": None}


def test_generate_runs_in_the_background_and_reports_progress(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, calls = threading.Event(), []
    monkeypatch.setattr(server, "_load_generate_dataset", lambda: blocking_generate(release, calls))

    response = client.post("/sim-generate", json={"episodes": 30, "seed": 0, "randomize": True})
    assert response.status_code == 200
    assert response.json() == {"started": True}

    deadline = time.monotonic() + 5
    while client.get("/sim-generate").json()["saved"] != 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client.get("/sim-generate").json() == {"running": True, "saved": 2, "episodes": 30, "error": None}

    release.set()
    assert wait_until_idle(client, "/sim-generate") == {
        "running": False,
        "saved": 30,
        "episodes": 30,
        "error": None,
    }
    repo_id, episodes, kwargs = calls[0]
    assert repo_id != "local/so101-chess-sim"  # the M1 dataset is never regenerated
    assert (episodes, kwargs["seed"], kwargs["randomize"]) == (30, 0, True)


def test_generate_refuses_a_second_start_while_running(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, calls = threading.Event(), []
    monkeypatch.setattr(server, "_load_generate_dataset", lambda: blocking_generate(release, calls))
    assert (
        client.post("/sim-generate", json={"episodes": 5, "seed": 1, "randomize": False}).status_code == 200
    )

    second = client.post("/sim-generate", json={"episodes": 5, "seed": 2, "randomize": False})
    assert second.status_code == 409
    assert "already running" in second.json()["detail"]

    release.set()
    wait_until_idle(client, "/sim-generate")
    assert len(calls) == 1


def test_generate_failure_is_reported_and_frees_the_slot(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(*args, **kwargs):
        raise RuntimeError("Saved 0/5 episodes in 110 attempts")

    monkeypatch.setattr(server, "_load_generate_dataset", lambda: failing)
    client.post("/sim-generate", json={"episodes": 5, "seed": 0, "randomize": True})

    status = wait_until_idle(client, "/sim-generate")
    assert status["error"] == "Saved 0/5 episodes in 110 attempts"
    assert client.post("/sim-generate", json={"episodes": 5, "seed": 0, "randomize": True}).status_code == 200
    wait_until_idle(client, "/sim-generate")


def test_evaluate_runs_and_returns_the_report(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple] = []

    def evaluate_checkpoint(checkpoint, *, seeds):
        calls.append((checkpoint, seeds))
        return SimpleNamespace(to_dict=lambda: {"name": "so101-chess-sim-eval", "success_rate": 0.5})

    monkeypatch.setattr(server, "_load_evaluate_checkpoint", lambda: evaluate_checkpoint)

    response = client.post("/sim-evaluate", json={"checkpoint": "/tmp/run/pretrained_model", "seeds": 3})
    assert response.json() == {"started": True}

    status = wait_until_idle(client, "/sim-evaluate")
    assert status == {
        "running": False,
        "report": {"name": "so101-chess-sim-eval", "success_rate": 0.5},
        "error": None,
    }
    assert calls == [(Path("/tmp/run/pretrained_model"), 3)]


def test_evaluate_refuses_a_second_start_while_running(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()

    def evaluate_checkpoint(checkpoint, *, seeds):
        release.wait(5)
        return SimpleNamespace(to_dict=lambda: {})

    monkeypatch.setattr(server, "_load_evaluate_checkpoint", lambda: evaluate_checkpoint)
    assert client.post("/sim-evaluate", json={"checkpoint": "a", "seeds": 3}).status_code == 200

    assert client.post("/sim-evaluate", json={"checkpoint": "b", "seeds": 3}).status_code == 409
    # A generation job counts too: one simulation job at a time.
    assert client.post("/sim-generate", json={"episodes": 5, "seed": 0, "randomize": True}).status_code == 409

    release.set()
    wait_until_idle(client, "/sim-evaluate")


def test_evaluate_failure_is_reported(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(checkpoint, *, seeds):
        raise FileNotFoundError("no checkpoint at /nope")

    monkeypatch.setattr(server, "_load_evaluate_checkpoint", lambda: failing)
    client.post("/sim-evaluate", json={"checkpoint": "/nope", "seeds": 3})

    status = wait_until_idle(client, "/sim-evaluate")
    assert status["report"] is None
    assert status["error"] == "no checkpoint at /nope"


def test_requests_are_validated(client: TestClient) -> None:
    assert client.post("/sim-generate", json={"episodes": 0, "seed": 0, "randomize": True}).status_code == 422
    assert client.post("/sim-evaluate", json={"checkpoint": "", "seeds": 3}).status_code == 422
    assert client.post("/sim-evaluate", json={"checkpoint": "a", "seeds": 0}).status_code == 422
