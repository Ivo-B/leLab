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


def _blank_to_none(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return value.strip()


def load_forge_config() -> ForgeConfig:
    return ForgeConfig(
        api_key=_blank_to_none(os.environ.get("WANDB_API_KEY")),
        entity=_blank_to_none(os.environ.get("WANDB_ENTITY")),
        project=_blank_to_none(os.environ.get("WANDB_PROJECT")) or "so101-kitting",
    )


def forge_configured(cfg: ForgeConfig) -> bool:
    return cfg.api_key is not None and cfg.entity is not None
