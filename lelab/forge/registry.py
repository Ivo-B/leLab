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

import re
from pathlib import Path

_SKILL_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def collection_name(policy_type: str, skill: str | None) -> str:
    if policy_type == "smolvla":
        return "so101-kitting-smolvla"
    if policy_type == "act":
        if skill is None or _SKILL_SLUG.fullmatch(skill) is None:
            raise ValueError("ACT registry linking needs a skill slug such as bolt, bracket, or gear")
        return f"so101-kitting-act-{skill}"
    raise ValueError(f"No registry collection for policy type {policy_type}")


def registry_path(entity: str, collection: str) -> str:
    return f"{entity}/wandb-registry-model/{collection}"


def episode_counts(dataset_dir: Path) -> dict[str, int]:
    from lelab.episode_media import read_episode_index

    counts: dict[str, int] = {}
    for row in read_episode_index(dataset_dir):
        for task in row.tasks:
            counts[task] = counts.get(task, 0) + 1
    return counts
