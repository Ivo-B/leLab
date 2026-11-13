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

import json

from lelab.train import TrainingRequest, build_training_command
from lelab.utils.forge_config import ForgeConfig

SANDBOX_TAG = "lelab"
REMOTE_OUTPUT_DIR = "/work/output"


def sandbox_create_kwargs(cfg: ForgeConfig, skill: str | None) -> dict[str, object]:
    env: dict[str, str] = {"FORGE_SKILL": skill or "", "PYTHONUNBUFFERED": "1"}
    if cfg.api_key is not None:
        env["WANDB_API_KEY"] = cfg.api_key
    if cfg.entity is not None:
        env["WANDB_ENTITY"] = cfg.entity
    env["WANDB_PROJECT"] = cfg.project
    if cfg.hf_token is not None:
        env["HF_TOKEN"] = cfg.hf_token
    return {
        "args": ["sleep", "infinity"],
        "container_image": cfg.sandbox_image,
        "placement_mode": "serverless",
        "tags": [SANDBOX_TAG],
        "max_lifetime_seconds": cfg.sandbox_lifetime_seconds,
        "resources": {"cpu": cfg.sandbox_cpu, "memory": cfg.sandbox_memory, "gpu": 1},
        "volumes": [{"name": "work", "mount_path": "/work", "size": cfg.sandbox_volume_size}],
        "environment_variables": env,
    }


def redact_secrets(text: str, cfg: ForgeConfig) -> str:
    for secret in (cfg.api_key, cfg.sandbox_api_key, cfg.hf_token):
        if secret:
            text = text.replace(secret, "[redacted]")
    return text


def training_command(config: TrainingRequest) -> list[str]:
    pinned = config.model_copy(update={"policy_device": config.policy_device or "cuda"})
    return build_training_command(pinned, REMOTE_OUTPUT_DIR, "python")


def dumps_command(command: list[str]) -> bytes:
    return json.dumps(command).encode()
