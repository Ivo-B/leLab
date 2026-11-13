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

import pytest

from lelab.forge.registry import collection_name, episode_counts, registry_path


def test_smolvla_has_one_collection() -> None:
    assert collection_name("smolvla", None) == "so101-kitting-smolvla"
    assert collection_name("smolvla", "bolt") == "so101-kitting-smolvla"


def test_act_collection_uses_the_skill_slug() -> None:
    assert collection_name("act", "bracket") == "so101-kitting-act-bracket"


def test_act_without_a_slug_is_rejected() -> None:
    with pytest.raises(ValueError, match="skill slug"):
        collection_name("act", None)
    with pytest.raises(ValueError, match="skill slug"):
        collection_name("act", "Pick the bolt")


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="groot"):
        collection_name("groot", "bolt")


def test_registry_path() -> None:
    assert registry_path("coreweave-demo", "so101-kitting-smolvla") == (
        "coreweave-demo/wandb-registry-model/so101-kitting-smolvla"
    )


def test_episode_counts_sum_tasks(tmp_path: Path, monkeypatch) -> None:
    from lelab import episode_media

    class Row:
        def __init__(self, tasks: tuple[str, ...]) -> None:
            self.tasks = tasks

    monkeypatch.setattr(
        episode_media,
        "read_episode_index",
        lambda directory: [Row(("pick the bolt",)), Row(("pick the bolt",)), Row(("pick the bracket",))],
    )
    assert episode_counts(tmp_path) == {"pick the bolt": 2, "pick the bracket": 1}
