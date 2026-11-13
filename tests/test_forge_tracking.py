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

import inspect
from pathlib import Path

from lelab.forge.tracking import log_dataset_if_configured
from lerobot.common.trackio_utils import TrackioLogger


def test_unconfigured_forge_does_not_open_a_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    assert log_dataset_if_configured(tmp_path, "me/so101") is None


def test_configured_forge_logs_one_dataset_artifact(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "local-test-key")
    monkeypatch.setenv("WANDB_ENTITY", "coreweave-demo")
    monkeypatch.setenv("WANDB_PROJECT", "so101-kitting")

    class Artifact:
        def __init__(self, name, type, metadata) -> None:  # noqa: A002
            self.name = name
            self.type = type
            self.metadata = metadata
            self.dirs: list[str] = []

        def add_dir(self, path: str) -> None:
            self.dirs.append(path)

    class Run:
        url = "https://wandb.ai/coreweave-demo/so101-kitting/runs/abc"
        logged = None

        def log_artifact(self, artifact) -> None:
            self.logged = artifact

        def finish(self) -> None:
            self.finished = True

    run = Run()
    created: dict = {}

    def init(**kwargs):
        created.update(kwargs)
        return run

    import wandb

    monkeypatch.setattr(wandb, "init", init)
    monkeypatch.setattr(wandb, "Artifact", Artifact)
    monkeypatch.setattr(
        "lelab.forge.tracking.episode_counts",
        lambda directory: {"pick the bolt": 2},
    )

    url = log_dataset_if_configured(tmp_path, "me/so101")

    assert url == run.url
    assert created["entity"] == "coreweave-demo"
    assert created["project"] == "so101-kitting"
    assert created["job_type"] == "record"
    assert run.logged.name == "so101-kitting"
    assert run.logged.type == "dataset"
    assert run.logged.metadata == {"repo_id": "me/so101", "episode_counts": {"pick the bolt": 2}}
    assert run.logged.dirs == [str(tmp_path)]
    assert run.finished is True


def test_a_wandb_error_does_not_escape(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "local-test-key")
    monkeypatch.setenv("WANDB_ENTITY", "coreweave-demo")
    import wandb

    def explode(**kwargs):
        raise RuntimeError("wandb down")

    monkeypatch.setattr(wandb, "init", explode)
    assert log_dataset_if_configured(tmp_path, "me/so101") is None


def test_a_failed_artifact_log_still_finishes_the_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "local-test-key")
    monkeypatch.setenv("WANDB_ENTITY", "coreweave-demo")
    monkeypatch.setenv("WANDB_PROJECT", "so101-kitting")

    class Artifact:
        def __init__(self, name, type, metadata) -> None:  # noqa: A002
            self.name = name
            self.type = type
            self.metadata = metadata

        def add_dir(self, path: str) -> None:
            self.dirs = [path]

    class Run:
        url = "https://wandb.ai/coreweave-demo/so101-kitting/runs/abc"

        def log_artifact(self, artifact) -> None:
            raise RuntimeError("artifact failed")

        def finish(self) -> None:
            self.finished = True

    run = Run()

    def init(**kwargs):
        return run

    import wandb

    monkeypatch.setattr(wandb, "init", init)
    monkeypatch.setattr(wandb, "Artifact", Artifact)
    monkeypatch.setattr(
        "lelab.forge.tracking.episode_counts",
        lambda directory: {"pick the bolt": 2},
    )

    assert log_dataset_if_configured(tmp_path, "me/so101") is None
    assert run.finished is True


class _Cfg:
    def __init__(self) -> None:
        self.trackio = type("Trackio", (), {"project": "so101-kitting"})()
        self.policy = type("Policy", (), {"type": "act"})()

    def to_dict(self) -> dict:
        return {"policy": {"type": "act"}}


def test_wandb_logger_matches_trackio_parameters() -> None:
    from lelab.forge.tracking import WandbLogger

    for name in ("__init__", "log_dict", "log_policy", "log_video", "finish"):
        expected = list(inspect.signature(getattr(TrackioLogger, name)).parameters)
        actual = list(inspect.signature(getattr(WandbLogger, name)).parameters)
        assert actual == expected, name


def test_finish_links_the_last_checkpoint_as_candidate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "local-test-key")
    monkeypatch.setenv("WANDB_ENTITY", "coreweave-demo")
    monkeypatch.setenv("FORGE_SKILL", "bolt")
    from lelab.forge.tracking import WandbLogger

    class Artifact:
        def __init__(self, name, type) -> None:  # noqa: A002
            self.name = name
            self.type = type
            self.files: list[Path] = []

        def add_file(self, path) -> None:
            self.files.append(Path(path))

    class Run:
        url = "https://wandb.ai/coreweave-demo/so101-kitting/runs/train1"
        linked = None

        def log_artifact(self, artifact) -> None:
            self.logged = artifact

        def link_artifact(self, artifact, target, aliases) -> None:
            self.linked = (artifact, target, aliases)

        def log(self, data, step=None) -> None:
            self.rows.append((data, step))

        def finish(self) -> None:
            self.finished = True

    run = Run()
    run.rows = []
    import wandb

    monkeypatch.setattr(wandb, "init", lambda **kwargs: run)
    monkeypatch.setattr(wandb, "Artifact", Artifact)
    weights = tmp_path / "ckpt" / "pretrained_model" / "model.safetensors"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")

    logger = WandbLogger(_Cfg())
    logger.log_dict({"loss": 0.5}, step=10, mode="train")
    logger.log_policy(weights.parent.parent)
    logger.finish()

    assert run.rows == [({"train/loss": 0.5}, 10)]
    assert run.logged.type == "model"
    assert run.logged.files == [weights]
    assert run.linked == (
        run.logged,
        "coreweave-demo/wandb-registry-model/so101-kitting-act-bolt",
        ["candidate"],
    )
    assert run.finished is True


def test_finish_skips_the_link_when_act_has_no_skill(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "local-test-key")
    monkeypatch.setenv("WANDB_ENTITY", "coreweave-demo")
    monkeypatch.delenv("FORGE_SKILL", raising=False)
    import wandb

    from lelab.forge.tracking import WandbLogger

    class Artifact:
        def __init__(self, name, type) -> None:  # noqa: A002
            pass

        def add_file(self, path) -> None:
            pass

    class Run:
        url = "https://wandb.ai/run"
        linked = None

        def log_artifact(self, artifact) -> None:
            pass

        def link_artifact(self, artifact, target, aliases) -> None:
            self.linked = target

        def finish(self) -> None:
            pass

    monkeypatch.setattr(wandb, "init", lambda **kwargs: Run())
    monkeypatch.setattr(wandb, "Artifact", Artifact)
    weights = tmp_path / "pretrained_model" / "model.safetensors"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    logger = WandbLogger(_Cfg())
    logger.log_policy(tmp_path)
    logger.finish()
    assert logger._run.linked is None
