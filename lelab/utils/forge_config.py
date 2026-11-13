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
from dataclasses import dataclass


@dataclass(frozen=True)
class ForgeConfig:
    api_key: str | None
    entity: str | None
    project: str
    sandbox_api_key: str | None = None
    sandbox_image: str | None = None
    sandbox_lifetime_seconds: int = 14400
    sandbox_cpu: str = "8"
    sandbox_memory: str = "32Gi"
    sandbox_volume_size: str = "100Gi"
    hf_token: str | None = None
    judge_model: str = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B"


def _blank_to_none(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return value.strip()


def _positive_int(value: str | None, default: int) -> int:
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value.strip())
    except ValueError:
        return default
    if parsed <= 0 or parsed > 2_592_000:
        return default
    return parsed


def load_forge_config() -> ForgeConfig:
    return ForgeConfig(
        api_key=_blank_to_none(os.environ.get("WANDB_API_KEY")),
        entity=_blank_to_none(os.environ.get("WANDB_ENTITY")),
        project=_blank_to_none(os.environ.get("WANDB_PROJECT")) or "so101-kitting",
        sandbox_api_key=_blank_to_none(os.environ.get("CWSANDBOX_API_KEY")),
        sandbox_image=_blank_to_none(os.environ.get("CWSANDBOX_TRAIN_IMAGE")),
        sandbox_lifetime_seconds=_positive_int(os.environ.get("CWSANDBOX_MAX_LIFETIME_SECONDS"), 14400),
        hf_token=_blank_to_none(os.environ.get("HF_TOKEN"))
        or _blank_to_none(os.environ.get("HUGGING_FACE_HUB_TOKEN")),
        judge_model=_blank_to_none(os.environ.get("FORGE_JUDGE_MODEL"))
        or "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B",
    )


def forge_configured(cfg: ForgeConfig) -> bool:
    return cfg.api_key is not None and cfg.entity is not None


def sandbox_configured(cfg: ForgeConfig) -> bool:
    return cfg.sandbox_api_key is not None and cfg.sandbox_image is not None
