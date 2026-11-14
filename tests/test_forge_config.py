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

import os

from dotenv import load_dotenv

from lelab.utils.forge_config import (
    ForgeConfig,
    forge_configured,
    load_forge_config,
    sandbox_configured,
    save_forge_credentials,
    upsert_dotenv,
)


def test_dotenv_file_fills_missing_env_vars(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("WANDB_API_KEY=from-dotenv\nWANDB_ENTITY=dotenv-entity\n")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    load_dotenv(env_file, override=True)
    cfg = load_forge_config()
    assert cfg.api_key == "from-dotenv"
    assert cfg.entity == "dotenv-entity"
    assert forge_configured(cfg) is True



def test_missing_credentials_are_not_configured(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
    monkeypatch.delenv("CWSANDBOX_TRAIN_IMAGE", raising=False)
    monkeypatch.delenv("CWSANDBOX_MAX_LIFETIME_SECONDS", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
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
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
    monkeypatch.delenv("CWSANDBOX_TRAIN_IMAGE", raising=False)
    client = TestClient(app)
    response = client.get("/forge-status")
    assert response.status_code == 200
    body = response.json()
    assert body["forge_configured"] is True
    assert body["entity"] == "coreweave-demo"
    assert body["project"] == "so101-kitting"
    assert body["sandbox_configured"] is False
    assert body["sandbox_image"] is None
    assert body["fields"] == {
        "api_key": True,
        "entity": True,
        "project": False,
        "sandbox_api_key": False,
        "sandbox_image": False,
    }
    assert "local-test-key" not in response.text


def test_sandbox_settings_come_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("CWSANDBOX_API_KEY", "sandbox-test-key")
    monkeypatch.setenv("CWSANDBOX_TRAIN_IMAGE", "registry.example/lelab-train:cuda")
    monkeypatch.setenv("CWSANDBOX_MAX_LIFETIME_SECONDS", "7200")
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    cfg = load_forge_config()
    assert cfg.sandbox_api_key == "sandbox-test-key"
    assert cfg.sandbox_image == "registry.example/lelab-train:cuda"
    assert cfg.sandbox_lifetime_seconds == 7200
    assert cfg.sandbox_cpu == "8"
    assert cfg.sandbox_memory == "32Gi"
    assert cfg.sandbox_volume_size == "100Gi"
    assert cfg.hf_token == "hf-test-token"
    assert sandbox_configured(cfg) is True


def test_bad_lifetime_uses_four_hours(monkeypatch) -> None:
    monkeypatch.setenv("CWSANDBOX_MAX_LIFETIME_SECONDS", "0")
    assert load_forge_config().sandbox_lifetime_seconds == 14400


def test_sandbox_status_hides_both_keys(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from lelab.server import app

    monkeypatch.setenv("WANDB_API_KEY", "local-test-key")
    monkeypatch.setenv("WANDB_ENTITY", "coreweave-demo")
    monkeypatch.setenv("CWSANDBOX_API_KEY", "sandbox-test-key")
    monkeypatch.setenv("CWSANDBOX_TRAIN_IMAGE", "registry.example/lelab-train:cuda")
    client = TestClient(app)
    response = client.get("/forge-status")
    assert response.status_code == 200
    assert response.json()["sandbox_configured"] is True
    assert "sandbox-test-key" not in response.text
    assert "local-test-key" not in response.text


def test_judge_model_defaults_and_can_be_overridden(monkeypatch) -> None:
    monkeypatch.delenv("FORGE_JUDGE_MODEL", raising=False)
    assert load_forge_config().judge_model == "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B"
    monkeypatch.setenv("FORGE_JUDGE_MODEL", "google/gemma-4-26B-A4B-it")
    assert load_forge_config().judge_model == "google/gemma-4-26B-A4B-it"


def test_upsert_dotenv_preserves_comments_and_other_keys(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("# keep me\nHF_TOKEN=hf-keep\nWANDB_ENTITY=old\n")
    upsert_dotenv(env_file, {"WANDB_ENTITY": "new", "WANDB_API_KEY": "secret"})
    text = env_file.read_text()
    assert "# keep me" in text
    assert "HF_TOKEN=hf-keep" in text
    assert "WANDB_ENTITY=new" in text
    assert "WANDB_API_KEY=secret" in text


def test_save_forge_credentials_updates_env_and_hides_secrets(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
    monkeypatch.delenv("CWSANDBOX_TRAIN_IMAGE", raising=False)
    status = save_forge_credentials(
        api_key="ui-secret-key",
        entity="coreweave-demo",
        sandbox_api_key="sandbox-secret",
        sandbox_image="registry.example/lelab-train:cuda",
        env_path=env_file,
    )
    assert status["forge_configured"] is True
    assert status["entity"] == "coreweave-demo"
    assert status["project"] == "so101-kitting"
    assert status["sandbox_configured"] is True
    assert status["sandbox_image"] == "registry.example/lelab-train:cuda"
    assert status["fields"]["api_key"] is True
    assert status["fields"]["sandbox_api_key"] is True
    assert "ui-secret-key" not in str(status)
    assert "sandbox-secret" not in str(status)
    assert os.environ["WANDB_API_KEY"] == "ui-secret-key"
    text = env_file.read_text()
    assert "WANDB_API_KEY=ui-secret-key" in text
    assert "CWSANDBOX_API_KEY=sandbox-secret" in text


def test_forge_credentials_endpoint_never_echoes_secrets(tmp_path, monkeypatch) -> None:
    import lelab.utils.forge_config as forge_config
    from fastapi.testclient import TestClient

    from lelab.server import app

    env_file = tmp_path / ".env"
    monkeypatch.setattr(forge_config, "_ENV_FILE", env_file)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    client = TestClient(app)
    response = client.post(
        "/forge-credentials",
        json={
            "api_key": "post-secret-key",
            "entity": "coreweave-demo",
            "sandbox_api_key": "post-sandbox-secret",
            "sandbox_image": "registry.example/lelab-train:cuda",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["forge_configured"] is True
    assert body["sandbox_configured"] is True
    assert "post-secret-key" not in response.text
    assert "post-sandbox-secret" not in response.text


def test_forge_status_includes_empty_registry_when_unconfigured(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    from fastapi.testclient import TestClient

    from lelab.server import app

    body = TestClient(app).get("/forge-status").json()
    assert body["registry"] == {"ready": False, "collections": [], "error": None}


def test_promote_endpoint_rejects_when_forge_is_off(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    from fastapi.testclient import TestClient

    from lelab.server import app

    response = TestClient(app).post(
        "/registry/promote",
        json={"policy_family": "smolvla", "skill": None},
    )
    assert response.status_code == 400
