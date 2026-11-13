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

import pytest

from lelab.jobs import JobRegistry, JobTarget
from lelab.train import TrainingRequest


def test_wandb_job_requires_forge_config(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    registry = JobRegistry(tmp_path)
    with pytest.raises(ValueError, match="WANDB_API_KEY and WANDB_ENTITY"):
        registry.start(TrainingRequest(dataset_repo_id="me/so101", tracking="wandb"))


def test_wandb_job_refuses_hf_cloud(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "local-test-key")
    monkeypatch.setenv("WANDB_ENTITY", "coreweave-demo")
    registry = JobRegistry(tmp_path)
    with pytest.raises(ValueError, match="runs on this machine"):
        registry.start(
            TrainingRequest(dataset_repo_id="me/so101", tracking="wandb"),
            JobTarget(runner="hf_cloud", flavor="a10g-small"),
        )


def test_local_runner_exports_the_skill(monkeypatch, tmp_path) -> None:
    from lelab.jobs import LocalJobRunner

    seen = {}

    def capture(self, cmd, thread_name, extra_env=None) -> None:
        seen["cmd"] = cmd
        seen["env"] = extra_env

    monkeypatch.setattr(LocalJobRunner, "_spawn", capture)
    runner = LocalJobRunner.__new__(LocalJobRunner)
    runner.start(
        "job", TrainingRequest(dataset_repo_id="me/so101", tracking="wandb", skill="gear"), str(tmp_path)
    )
    assert seen["cmd"][2] == "lelab.forge.remote.train_entry"
    assert seen["env"]["FORGE_SKILL"] == "gear"
    assert "FORGE_SKILL" not in os.environ
