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
import logging
import time
from pathlib import Path

import pytest

from lelab.jobs import TrainingMetrics
from lelab.runners.sandbox import SandboxJobRunner
from lelab.train import TrainingRequest


@pytest.fixture(autouse=True)
def forge_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "wandb-secret")
    monkeypatch.setenv("WANDB_ENTITY", "coreweave-demo")
    monkeypatch.setenv("CWSANDBOX_API_KEY", "sandbox-secret")
    monkeypatch.setenv("CWSANDBOX_TRAIN_IMAGE", "registry.example/lelab-train:cuda")


class FakeProcess:
    def __init__(self, lines: list[str], code: int) -> None:
        self.stdout = iter(lines)
        self.returncode = code


class FailingStdout:
    def __iter__(self) -> "FailingStdout":
        return self

    def __next__(self) -> str:
        raise RuntimeError("stream wandb-secret")


class FailingStreamProcess:
    def __init__(self) -> None:
        self.stdout = FailingStdout()
        self.returncode = 0


class FakeHandle:
    def __init__(self, process: FakeProcess) -> None:
        self.process = process
        self.written: list[tuple[str, bytes]] = []
        self.stopped = False
        self.commands: list[list[str]] = []

    def write_file(self, path: str, contents: bytes) -> None:
        self.written.append((path, contents))

    def exec(self, command: list[str]) -> FakeProcess:
        self.commands.append(command)
        return self.process

    def stop(self) -> None:
        self.stopped = True


class FailingStopHandle(FakeHandle):
    def stop(self) -> None:
        raise RuntimeError("stop failed wandb-secret")


class FakeClient:
    def __init__(self, handle: FakeHandle | None = None, error: Exception | None = None) -> None:
        self.handle = handle
        self.error = error
        self.created: list[dict] = []

    def create(self, kwargs: dict) -> FakeHandle:
        self.created.append(kwargs)
        if self.error is not None:
            raise self.error
        assert self.handle is not None
        return self.handle


def wait_until_finished(runner: SandboxJobRunner) -> None:
    deadline = 0
    while runner.is_running() and deadline < 100:
        time.sleep(0.02)
        deadline += 1
    assert not runner.is_running()


def test_runner_streams_stdout_and_uploads_the_command(tmp_path: Path) -> None:
    handle = FakeHandle(FakeProcess(["Training:   1%| | 1/10 [00:01<00:09,  1.00step/s]\n"], 0))
    client = FakeClient(handle)
    log_path = tmp_path / "log.jsonl"
    runner = SandboxJobRunner(TrainingMetrics(), log_path, client)
    request = TrainingRequest(dataset_repo_id="me/so101", policy_type="act", tracking="wandb", skill="bolt")
    runner.start("job-1", request, str(tmp_path / "run"))
    wait_until_finished(runner)
    assert runner.returncode() == 0
    assert handle.written[0][0] == "/work/train_argv.json"
    assert b"lelab.forge.remote.train_entry" in handle.written[0][1]
    assert json.loads(handle.written[0][1]) == handle.commands[0]
    assert handle.commands[0][0:3] == ["python", "-m", "lelab.forge.remote.train_entry"]
    assert client.created[0]["tags"] == ["lelab"]
    assert client.created[0]["resources"]["gpu"] == 1
    assert "Training:" in log_path.read_text()
    lines = runner.stream_log_lines()
    assert any("Training:" in line.message for line in lines)


def test_create_failure_is_not_retried_and_offers_local(tmp_path: Path) -> None:
    client = FakeClient(error=RuntimeError("CWSANDBOX_RUNNER_UNAVAILABLE"))
    runner = SandboxJobRunner(TrainingMetrics(), tmp_path / "log.jsonl", client)
    request = TrainingRequest(dataset_repo_id="me/so101", policy_type="smolvla", tracking="wandb")
    runner.start("job-2", request, str(tmp_path / "run"))
    wait_until_finished(runner)
    assert runner.returncode() == 1
    message = runner.failure_message()
    assert message is not None
    assert "CWSANDBOX_RUNNER_UNAVAILABLE" in message
    assert "Run this training on this machine instead." in message
    assert len(client.created) == 1


def test_create_failure_redacts_secrets(tmp_path: Path) -> None:
    client = FakeClient(error=RuntimeError("boom wandb-secret sandbox-secret"))
    log_path = tmp_path / "log.jsonl"
    runner = SandboxJobRunner(TrainingMetrics(), log_path, client)
    request = TrainingRequest(dataset_repo_id="me/so101", policy_type="act", tracking="wandb")
    runner.start("job-4", request, str(tmp_path / "run"))
    wait_until_finished(runner)
    message = runner.failure_message()
    assert message is not None
    persisted = log_path.read_text()
    for text in (message, persisted):
        assert "wandb-secret" not in text
        assert "sandbox-secret" not in text
        assert "[redacted]" in text


def test_stop_stops_the_sandbox(tmp_path: Path) -> None:
    handle = FakeHandle(FakeProcess([], 0))
    runner = SandboxJobRunner(TrainingMetrics(), tmp_path / "log.jsonl", FakeClient(handle))
    request = TrainingRequest(dataset_repo_id="me/so101", policy_type="act", tracking="off")
    runner.start("job-3", request, str(tmp_path / "run"))
    runner.stop()
    assert handle.stopped is True


def test_stop_does_not_raise_when_the_sandbox_stop_fails(tmp_path: Path) -> None:
    handle = FailingStopHandle(FakeProcess([], 0))
    log_path = tmp_path / "log.jsonl"
    runner = SandboxJobRunner(TrainingMetrics(), log_path, FakeClient(handle))
    request = TrainingRequest(dataset_repo_id="me/so101", policy_type="act", tracking="off")
    runner.start("job-5", request, str(tmp_path / "run"))
    runner.stop()
    wait_until_finished(runner)
    persisted = log_path.read_text()
    assert "Could not stop the CoreWeave Sandbox:" in persisted
    assert "wandb-secret" not in persisted


def test_stop_failure_after_the_log_is_closed_is_still_written(tmp_path: Path) -> None:
    handle = FailingStopHandle(FakeProcess([], 0))
    log_path = tmp_path / "log.jsonl"
    runner = SandboxJobRunner(TrainingMetrics(), log_path, FakeClient(handle))
    request = TrainingRequest(dataset_repo_id="me/so101", policy_type="act", tracking="off")
    runner.start("job-7", request, str(tmp_path / "run"))
    wait_until_finished(runner)
    runner.stop()
    persisted = log_path.read_text()
    assert "Could not stop the CoreWeave Sandbox:" in persisted
    assert "[redacted]" in persisted
    assert "wandb-secret" not in persisted


def test_stream_error_is_redacted_everywhere(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    handle = FakeHandle(FailingStreamProcess())
    log_path = tmp_path / "log.jsonl"
    runner = SandboxJobRunner(TrainingMetrics(), log_path, FakeClient(handle))
    request = TrainingRequest(dataset_repo_id="me/so101", policy_type="act", tracking="wandb")
    runner.start("job-6", request, str(tmp_path / "run"))
    wait_until_finished(runner)
    assert runner.returncode() == 1
    message = runner.failure_message()
    assert message is not None
    assert "The CoreWeave Sandbox training failed:" in message
    assert "[redacted]" in message
    persisted = log_path.read_text()
    assert "[redacted]" in persisted
    for text in (message, persisted, caplog.text):
        assert "wandb-secret" not in text
