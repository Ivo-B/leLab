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

import sys
import time

import pytest

from lelab.jobs import JobRecord, JobRegistry, JobTarget
from lelab.train import TrainingRequest


def _set_forge_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "wandb-secret")
    monkeypatch.setenv("WANDB_ENTITY", "coreweave-demo")
    monkeypatch.setenv("CWSANDBOX_API_KEY", "sandbox-secret")
    monkeypatch.setenv("CWSANDBOX_TRAIN_IMAGE", "registry.example/lelab-train:cuda")


def _wait_until_settled(registry: JobRegistry, record: JobRecord) -> JobRecord:
    for _ in range(100):
        record = registry.get(record.id)
        if record.state != "running":
            break
        time.sleep(0.05)
    return record


def test_coreweave_requires_sandbox_settings(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
    monkeypatch.delenv("CWSANDBOX_TRAIN_IMAGE", raising=False)
    registry = JobRegistry(tmp_path)
    request = TrainingRequest(dataset_repo_id="me/so101", policy_type="act", tracking="off")
    with pytest.raises(ValueError, match="CWSANDBOX_API_KEY and CWSANDBOX_TRAIN_IMAGE"):
        registry.start(request, JobTarget(runner="coreweave"))


@pytest.mark.parametrize("tracking", ["off", "trackio"])
def test_coreweave_without_wandb_is_rejected(monkeypatch, tmp_path, tracking) -> None:
    _set_forge_env(monkeypatch)
    created: list[dict] = []

    class Client:
        def create(self, kwargs):
            created.append(kwargs)
            raise AssertionError("no Sandbox may be created")

    monkeypatch.setattr("lelab.runners.cwsandbox_client.CwsandboxClient", Client)
    registry = JobRegistry(tmp_path)
    request = TrainingRequest(dataset_repo_id="me/so101", policy_type="act", tracking=tracking)
    with pytest.raises(ValueError) as excinfo:
        registry.start(request, JobTarget(runner="coreweave"))
    assert str(excinfo.value) == (
        "CoreWeave Sandbox training needs Weights & Biases tracking, "
        "so the checkpoint is saved before the Sandbox is deleted."
    )
    assert created == []


def test_wandb_check_runs_before_the_sandbox_check(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.setenv("CWSANDBOX_API_KEY", "sandbox-secret")
    monkeypatch.setenv("CWSANDBOX_TRAIN_IMAGE", "registry.example/lelab-train:cuda")
    registry = JobRegistry(tmp_path)
    request = TrainingRequest(dataset_repo_id="me/so101", policy_type="act", tracking="wandb")
    with pytest.raises(ValueError, match="Weights & Biases tracking needs WANDB_API_KEY and WANDB_ENTITY"):
        registry.start(request, JobTarget(runner="coreweave"))


def test_wandb_still_refuses_hf_cloud(monkeypatch, tmp_path) -> None:
    _set_forge_env(monkeypatch)
    registry = JobRegistry(tmp_path)
    request = TrainingRequest(dataset_repo_id="me/so101", policy_type="act", tracking="wandb")
    with pytest.raises(ValueError, match="Weights & Biases tracking runs on this machine"):
        registry.start(request, JobTarget(runner="hf_cloud", flavor="a10g-small"))


def test_coreweave_job_uses_the_sandbox_runner(monkeypatch, tmp_path) -> None:
    _set_forge_env(monkeypatch)
    registry = JobRegistry(tmp_path)
    created = {}

    class Client:
        def create(self, kwargs):
            created["kwargs"] = kwargs
            raise RuntimeError("capacity")

    monkeypatch.setattr("lelab.runners.cwsandbox_client.CwsandboxClient", Client)
    request = TrainingRequest(dataset_repo_id="me/so101", policy_type="smolvla", tracking="wandb")
    record = registry.start(request, JobTarget(runner="coreweave"))
    assert record.runner == "coreweave"
    record = _wait_until_settled(registry, record)
    assert record.state == "failed"
    assert "capacity" in (record.error_message or "")
    assert "Run this training on this machine instead." in (record.error_message or "")
    assert "wandb-secret" not in (record.error_message or "")
    assert "sandbox-secret" not in (record.error_message or "")
    assert created["kwargs"]["resources"]["gpu"] == 1


def test_coreweave_job_deletes_the_sandbox_after_a_normal_exit(monkeypatch, tmp_path) -> None:
    _set_forge_env(monkeypatch)
    registry = JobRegistry(tmp_path)

    class Process:
        stdout = iter(["training started"])
        returncode = 0

    class Handle:
        stopped = False

        def write_file(self, path, contents):
            pass

        def exec(self, command):
            return Process()

        def stop(self):
            self.stopped = True

    handle = Handle()

    class Client:
        def create(self, kwargs):
            return handle

    monkeypatch.setattr("lelab.runners.cwsandbox_client.CwsandboxClient", Client)
    request = TrainingRequest(dataset_repo_id="me/so101", policy_type="act", tracking="wandb")
    record = registry.start(request, JobTarget(runner="coreweave"))
    record = _wait_until_settled(registry, record)
    assert record.state == "done"
    assert handle.stopped is True


def test_interrupted_coreweave_record_is_not_reattached(tmp_path) -> None:
    registry = JobRegistry(tmp_path)
    record = JobRecord(
        id="act_me_so101_2026-01-01_00-00-00",
        name="ACT · me/so101",
        state="running",
        config=TrainingRequest(dataset_repo_id="me/so101", policy_type="act"),
        output_dir=str(tmp_path / "run"),
        started_at=time.time(),
        runner="coreweave",
    )
    registry._write_meta(record)
    registry.shutdown()

    reloaded = JobRegistry(tmp_path)
    loaded = reloaded.get(record.id)
    assert loaded.state == "interrupted"
    assert loaded.error_message == "The Sandbox was stopped because the server restarted."
    assert loaded.ended_at is not None
    assert record.id not in reloaded._runners
    reloaded.shutdown()


def test_sweep_returns_zero_without_sandbox_settings(monkeypatch) -> None:
    monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
    monkeypatch.delenv("CWSANDBOX_TRAIN_IMAGE", raising=False)
    monkeypatch.delitem(sys.modules, "cwsandbox", raising=False)
    from lelab.runners.cwsandbox_client import sweep_lelab_sandboxes

    assert sweep_lelab_sandboxes() == 0
    assert "cwsandbox" not in sys.modules


def test_sweep_stops_every_lelab_sandbox_and_continues_after_a_failure(monkeypatch) -> None:
    _set_forge_env(monkeypatch)
    import types

    stopped: list[str] = []

    class Future:
        def __init__(self, value=None, error=None):
            self._value, self._error = value, error

        def result(self):
            if self._error is not None:
                raise self._error
            return self._value

    class FakeSandbox:
        def __init__(self, name, fail=False):
            self.name, self.fail = name, fail

        def stop(self, missing_ok=False):
            if self.fail:
                return Future(error=RuntimeError("boom sandbox-secret"))
            stopped.append(self.name)
            return Future()

    class FakeSandboxCls:
        listed: dict = {}

        @staticmethod
        def list(**kwargs):
            FakeSandboxCls.listed = kwargs
            return Future([FakeSandbox("a"), FakeSandbox("b", fail=True), FakeSandbox("c")])

    fake = types.SimpleNamespace(
        Sandbox=FakeSandboxCls, AuthStrategy=types.SimpleNamespace(COREWEAVE_API_KEY="cw-key")
    )
    monkeypatch.setitem(sys.modules, "cwsandbox", fake)
    from lelab.runners.cwsandbox_client import sweep_lelab_sandboxes

    assert sweep_lelab_sandboxes() == 2
    assert stopped == ["a", "c"]
    assert FakeSandboxCls.listed["tags"] == ["lelab"]
    assert FakeSandboxCls.listed["auth"] == "cw-key"


def test_shutdown_stops_running_coreweave_jobs_then_sweeps(monkeypatch) -> None:
    from lelab import server

    calls: list[str] = []

    class Registry:
        def running_job_ids(self, runner):
            calls.append(f"ids:{runner}")
            return ["job-a", "job-b"]

        def stop(self, job_id):
            calls.append(f"stop:{job_id}")
            if job_id == "job-a":
                raise RuntimeError("already gone")

    monkeypatch.setattr(server, "job_registry", Registry())
    monkeypatch.setattr(
        "lelab.runners.cwsandbox_client.sweep_lelab_sandboxes", lambda: calls.append("sweep") or 1
    )
    server._stop_coreweave_jobs()
    server._sweep_sandboxes_safely("shutdown")
    assert calls == ["ids:coreweave", "stop:job-a", "stop:job-b", "sweep"]
