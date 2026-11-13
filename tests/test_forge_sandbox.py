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

from lelab.forge.sandbox import (
    REMOTE_OUTPUT_DIR,
    SANDBOX_TAG,
    redact_secrets,
    sandbox_create_kwargs,
    training_command,
)
from lelab.train import TrainingRequest
from lelab.utils.forge_config import ForgeConfig


def _cfg() -> ForgeConfig:
    return ForgeConfig(
        api_key="wandb-secret",
        entity="coreweave-demo",
        project="so101-kitting",
        sandbox_api_key="sandbox-secret",
        sandbox_image="registry.example/lelab-train:cuda",
        sandbox_lifetime_seconds=14400,
        hf_token="hf-secret",
    )


def test_create_kwargs_request_one_serverless_gpu() -> None:
    kwargs = sandbox_create_kwargs(_cfg(), "bolt")
    assert kwargs["args"] == ["sleep", "infinity"]
    assert kwargs["container_image"] == "registry.example/lelab-train:cuda"
    assert kwargs["placement_mode"] == "serverless"
    assert kwargs["tags"] == [SANDBOX_TAG]
    assert kwargs["max_lifetime_seconds"] == 14400
    assert kwargs["resources"] == {"cpu": "8", "memory": "32Gi", "gpu": 1}
    assert "type" not in kwargs["resources"]
    assert kwargs["volumes"] == [{"name": "work", "mount_path": "/work", "size": "100Gi"}]
    env = kwargs["environment_variables"]
    assert env["WANDB_API_KEY"] == "wandb-secret"
    assert env["WANDB_ENTITY"] == "coreweave-demo"
    assert env["WANDB_PROJECT"] == "so101-kitting"
    assert env["FORGE_SKILL"] == "bolt"
    assert env["HF_TOKEN"] == "hf-secret"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert "CWSANDBOX_API_KEY" not in env
    assert "sandbox-secret" not in str(kwargs)


def test_blank_skill_is_an_empty_env_value() -> None:
    env = sandbox_create_kwargs(_cfg(), None)["environment_variables"]
    assert env["FORGE_SKILL"] == ""


def test_redact_secrets_removes_both_keys() -> None:
    text = redact_secrets("failed wandb-secret and sandbox-secret", _cfg())
    assert text == "failed [redacted] and [redacted]"


def test_training_command_uses_the_sandbox_output_and_the_wandb_wrapper() -> None:
    request = TrainingRequest(
        dataset_repo_id="me/so101",
        policy_type="act",
        tracking="wandb",
        skill="bolt",
    )
    cmd = training_command(request)
    assert cmd[:4] == ["python", "-m", "lelab.forge.remote.train_entry", "--dataset.repo_id"]
    assert "me/so101" in cmd
    assert "--output_dir" in cmd
    assert cmd[cmd.index("--output_dir") + 1] == REMOTE_OUTPUT_DIR
    assert "--trackio.enable" in cmd
    assert "true" in cmd
    assert "--trackio.space_id" not in cmd
