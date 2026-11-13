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

from lelab.utils.forge_config import ForgeConfig, forge_configured, load_forge_config


def test_missing_credentials_are_not_configured(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    cfg = load_forge_config()
    assert cfg == ForgeConfig(api_key=None, entity=None, project="so101-kitting")
    assert forge_configured(cfg) is False


def test_key_and_entity_configure_forge(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "local-test-key")
    monkeypatch.setenv("WANDB_ENTITY", "coreweave-demo")
    monkeypatch.setenv("WANDB_PROJECT", "so101")
    cfg = load_forge_config()
    assert cfg.api_key == "local-test-key"
    assert cfg.entity == "coreweave-demo"
    assert cfg.project == "so101"
    assert forge_configured(cfg) is True


def test_blank_entity_is_not_configured(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "local-test-key")
    monkeypatch.setenv("WANDB_ENTITY", "  ")
    assert forge_configured(load_forge_config()) is False


def test_forge_status_hides_the_api_key(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from lelab.server import app

    monkeypatch.setenv("WANDB_API_KEY", "local-test-key")
    monkeypatch.setenv("WANDB_ENTITY", "coreweave-demo")
    client = TestClient(app)
    response = client.get("/forge-status")
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "forge_configured": True,
        "entity": "coreweave-demo",
        "project": "so101-kitting",
    }
    assert "local-test-key" not in response.text
