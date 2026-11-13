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

import time

import pytest
from fastapi.testclient import TestClient

from lelab.agent.loop import OrderResult
from lelab.agent.session import reset_session
from lelab.server import app

client = TestClient(app)


class BlockingLoop:
    def __init__(self, planner, executor, stop=None) -> None:
        del planner, executor
        self.stop = stop

    def run(self, text, policy_family, policy_host):
        del text, policy_family, policy_host
        assert self.stop is not None
        self.stop.wait(timeout=2)
        return OrderResult("stopped", None, {"bolt": 1}, [], 0)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.setattr("lelab.agent.session.OrderLoop", BlockingLoop)
    monkeypatch.setattr("lelab.agent.session.export_trace", lambda *args, **kwargs: None)
    reset_session()
    yield
    reset_session()


def test_forge_off_is_400() -> None:
    response = client.post(
        "/orders", json={"text": "one bolt", "policy_family": "smolvla", "policy_host": "local"}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Orders need Weights & Biases"


def test_second_order_is_409_and_stop_finishes_the_worker(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "local-test-key")
    monkeypatch.setenv("WANDB_ENTITY", "team")
    body = {"text": "one bolt", "policy_family": "smolvla", "policy_host": "local"}
    first = client.post("/orders", json=body)
    assert first.status_code == 200
    assert first.json()["status"] == "running"
    assert first.json()["counts"] == {"bolt": 1}
    second = client.post("/orders", json=body)
    assert second.status_code == 409
    stopped = client.post("/orders/current/stop")
    assert stopped.status_code == 200
    status = {}
    for _ in range(20):
        status = client.get("/orders/current").json()
        if status["status"] == "stopped":
            break
        time.sleep(0.05)
    assert status["status"] == "stopped"
    assert client.post("/orders/current/answer", json={"message": "go"}).status_code == 400
