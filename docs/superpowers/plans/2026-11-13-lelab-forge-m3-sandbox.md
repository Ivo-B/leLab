# LeLab Forge M3 — Sandbox Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A training job started from the Train page can run on a CoreWeave GPU Sandbox, stream its logs into the existing job monitor, and delete that Sandbox on stop, shutdown, and the next server start.

**Architecture:** `SandboxJobRunner` implements the existing `JobRunner` protocol. It keeps a Sandbox alive with `sleep infinity`, writes the training argv to `/work/train_argv.json`, and runs `python -m lelab.forge.remote.train_entry` (the M2 wrapper) inside the Sandbox. The local watchdog already turns a finished runner into a job-card state, so the runner only has to stream stdout into the same log file and queue. Create is attempted once. Serverless placement with `gpu: 1` and no `type` is the RTX PRO 6000 Blackwell Server Edition, which is the only serverless GPU.

**Tech Stack:** Python 3.12, FastAPI, the existing job registry, `cwsandbox` 1.28.0 (`Sandbox.run`, `Sandbox.list`, `write_file`, `exec`, `process.stdout`, `stop`), React + Vite.

**Verified against cwsandbox 1.28.0 and the CoreWeave Sandbox docs (2026-11-13):**

- `AuthStrategy.COREWEAVE_API_KEY` reads `CWSANDBOX_API_KEY`. Do not send that key into the Sandbox.
- `cwsandbox.Secret` is `Secret(store, name, field="", env_var=None)`. It names a secret in a store. It cannot carry a raw key. `WANDB_API_KEY` therefore goes in `environment_variables` for the training process. Log lines and HTTP bodies pass through `redact_secrets` first.
- `max_lifetime_seconds` is fixed at create time. The platform maximum is 2,592,000 seconds (30 days). Unset means the runner policy or 10 minutes. This plan sets 14,400 seconds (4 hours) unless `CWSANDBOX_MAX_LIFETIME_SECONDS` overrides it.
- A scratch volume is `ScratchVolumeOptions(name, mount_path, size)`. Size is a request such as `"100Gi"`. The root filesystem is node disk and is not a quota.
- `Sandbox.list(tags=["lelab"]).result()` returns active sandboxes that have every listed tag. `sandbox.stop(missing_ok=True).result()` deletes one.
- `create_retry_budget_seconds` defaults to 0. Do not set it. A failed create is not retried.
- There is no published LeRobot CUDA image in those docs. The image is `CWSANDBOX_TRAIN_IMAGE`. It must already contain Python, CUDA, LeRobot, this `lelab` package, and `wandb`. The runner refuses to create a Sandbox when the image or the API key is missing.

## Global Constraints

- Do not fork LeRobot. The Sandbox runs the existing `lelab.forge.remote.train_entry` module.
- Do not start M4, M5, or M6. No judge, Weave, policy server, order agent, Automations, or ARIA.
- `WANDB_API_KEY` and `CWSANDBOX_API_KEY` never appear in a frontend response, a job log line, or a logger call.
- Weights & Biases plus `target.runner == "hf_cloud"` stays rejected with `Weights & Biases tracking runs on this machine`.
- With the Sandbox key or image unset, `POST /jobs/training` with `runner: "coreweave"` returns 400 and no Sandbox is created. Local and HF Jobs behavior stays as it is.
- One policy per job. The existing policy selector already chooses SmolVLA or one ACT skill. Do not add a batch launcher.
- Live Sandbox tests are marked `forge` and skipped unless `RUN_FORGE_LIVE=1`. Unit tests use a fake client and must not open a network connection.
- `JobRegistry()` construction must not list or create Sandboxes. The startup sweep is an explicit function called from the FastAPI startup hook.
- New Python modules start with the Apache-2.0 header in `tests/test_forge_config.py` lines 1–13.
- Pytest is not on `PATH`. Run `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest`. Install `cwsandbox` into that interpreter when a task imports it: `uv pip install --python /Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python 'cwsandbox>=1.14'`.
- Commit only the files the task lists. Push only to `origin` (`https://github.com/Ivo-B/leLab.git`). Do not push `upstream`. Do not remove the `github.repository == 'huggingface/leLab'` guard in `.github/workflows/sync_space.yml`.
- Work on the checked-out `forge/design` branch.

---

### Task 1: Sandbox settings

**Files:**
- Modify: `lelab/utils/forge_config.py`
- Modify: `lelab/server.py` (`forge_status`)
- Modify: `tests/test_forge_config.py`
- Test: `tests/test_forge_config.py`

**Interfaces:**
- Consumes: `ForgeConfig`, `load_forge_config`, `forge_configured` (unchanged meaning).
- Produces: `ForgeConfig.sandbox_api_key: str | None`, `sandbox_image: str | None`, `sandbox_lifetime_seconds: int`, `sandbox_cpu: str`, `sandbox_memory: str`, `sandbox_volume_size: str`, `hf_token: str | None`. `sandbox_configured(cfg: ForgeConfig) -> bool` is true only when both `sandbox_api_key` and `sandbox_image` are non-empty. `GET /forge-status` adds `sandbox_configured: bool` and still omits every secret.

- [ ] **Step 1: Write the failing test**

Add this test to `tests/test_forge_config.py` and extend `test_forge_status_hides_the_api_key` so the exact body includes `"sandbox_configured": False`. Also `monkeypatch.delenv` `CWSANDBOX_API_KEY`, `CWSANDBOX_TRAIN_IMAGE`, `CWSANDBOX_MAX_LIFETIME_SECONDS`, `HF_TOKEN`, and `HUGGING_FACE_HUB_TOKEN` in the existing missing-credentials test.

```python
def test_sandbox_settings_come_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("CWSANDBOX_API_KEY", "sandbox-test-key")
    monkeypatch.setenv("CWSANDBOX_TRAIN_IMAGE", "registry.example/lelab-train:cuda")
    monkeypatch.setenv("CWSANDBOX_MAX_LIFETIME_SECONDS", "7200")
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    cfg = load_forge_config()
    assert cfg.sandbox_api_key == "sandbox-test-key"
    assert cfg.sandbox_image == "registry.example/lelab-train:cuda"
    assert cfg.sandbox_lifetime_seconds == 7200
    assert cfg.sandbox_cpu == "8"
    assert cfg.sandbox_memory == "32Gi"
    assert cfg.sandbox_volume_size == "100Gi"
    assert cfg.hf_token == "hf-test-token"
    assert sandbox_configured(cfg) is True


def test_bad_lifetime_uses_four_hours(monkeypatch) -> None:
    monkeypatch.setenv("CWSANDBOX_MAX_LIFETIME_SECONDS", "0")
    assert load_forge_config().sandbox_lifetime_seconds == 14400


def test_sandbox_status_hides_both_keys(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from lelab.server import app

    monkeypatch.setenv("WANDB_API_KEY", "local-test-key")
    monkeypatch.setenv("WANDB_ENTITY", "coreweave-demo")
    monkeypatch.setenv("CWSANDBOX_API_KEY", "sandbox-test-key")
    monkeypatch.setenv("CWSANDBOX_TRAIN_IMAGE", "registry.example/lelab-train:cuda")
    client = TestClient(app)
    response = client.get("/forge-status")
    assert response.status_code == 200
    assert response.json()["sandbox_configured"] is True
    assert "sandbox-test-key" not in response.text
    assert "local-test-key" not in response.text
```

Import `sandbox_configured` in the test module.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_config.py -q`

Expected: FAIL because `sandbox_configured` and the new fields do not exist, and the status body has no `sandbox_configured`.

- [ ] **Step 3: Write the implementation**

Replace the dataclass and loaders in `lelab/utils/forge_config.py` with:

```python
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
    )


def forge_configured(cfg: ForgeConfig) -> bool:
    return cfg.api_key is not None and cfg.entity is not None


def sandbox_configured(cfg: ForgeConfig) -> bool:
    return cfg.sandbox_api_key is not None and cfg.sandbox_image is not None
```

In `forge_status`, import `sandbox_configured` and add `"sandbox_configured": sandbox_configured(cfg)` to the returned dict. Do not add the key, the image, or the HF token.

Update the exact-body assertion in `test_forge_status_hides_the_api_key` to include `"sandbox_configured": False`, and delete `CWSANDBOX_API_KEY` and `CWSANDBOX_TRAIN_IMAGE` in that test so a developer shell cannot flip it.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_config.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lelab/utils/forge_config.py lelab/server.py tests/test_forge_config.py docs/superpowers/plans/2026-11-13-lelab-forge-m3-sandbox.md
git commit -m "$(cat <<'EOF'
feat: read CoreWeave Sandbox settings from the environment

EOF
)"
```

---

### Task 2: Sandbox create request

**Files:**
- Create: `lelab/forge/sandbox.py`
- Create: `tests/test_forge_sandbox.py`
- Modify: `pyproject.toml` (add `cwsandbox>=1.14` to `dependencies`)

**Interfaces:**
- Consumes: `ForgeConfig` from Task 1.
- Produces:
  - `SANDBOX_TAG = "lelab"`
  - `REMOTE_OUTPUT_DIR = "/work/output"`
  - `sandbox_create_kwargs(cfg: ForgeConfig, skill: str | None) -> dict[str, object]`
  - `redact_secrets(text: str, cfg: ForgeConfig) -> str`
  - `training_command(config: TrainingRequest) -> list[str]`

`sandbox_create_kwargs` does not import `cwsandbox` and does not open a socket. The dict is plain data. The real client in Task 3 turns `volumes` into `ScratchVolumeOptions` and adds `auth=AuthStrategy.COREWEAVE_API_KEY`.

- [ ] **Step 1: Write the failing test**

```python
from lelab.forge.sandbox import REMOTE_OUTPUT_DIR, SANDBOX_TAG, redact_secrets, sandbox_create_kwargs, training_command
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_sandbox.py -q`

Expected: FAIL with `ModuleNotFoundError` or import error for `lelab.forge.sandbox`.

- [ ] **Step 3: Write the implementation**

`lelab/forge/sandbox.py`:

```python
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
    return build_training_command(config, REMOTE_OUTPUT_DIR, "python")


def dumps_command(command: list[str]) -> bytes:
    return json.dumps(command).encode()
```

Add `"cwsandbox>=1.14"` to `[project].dependencies` in `pyproject.toml`. Do not import it in this task.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_sandbox.py tests/test_train.py -q`

Expected: PASS. `test_train.py` stays green because `build_training_command` is unchanged.

- [ ] **Step 5: Commit**

```bash
git add lelab/forge/sandbox.py tests/test_forge_sandbox.py pyproject.toml
git commit -m "$(cat <<'EOF'
feat: describe the GPU Sandbox a training job will create

EOF
)"
```

---

### Task 3: SandboxJobRunner

**Files:**
- Create: `lelab/runners/sandbox.py`
- Create: `tests/test_forge_sandbox_runner.py`
- Modify: `pyproject.toml` only if Task 2 has not added `cwsandbox` yet

**Interfaces:**
- Consumes: `sandbox_create_kwargs`, `redact_secrets`, `training_command`, `dumps_command`, `REMOTE_OUTPUT_DIR` from Task 2. `JobRunner` methods: `start(job_id, config, output_dir)`, `stop()`, `is_running()`, `returncode()`, `stream_log_lines()`, `tracking_url()`.
- Produces: `SandboxJobRunner(metrics, log_file_path, client)`. `client.create(kwargs) -> handle`. `handle.write_file(path, contents)`, `handle.exec(command) -> process`, `handle.stop()`. `process.stdout` is an iterator of lines. `process.returncode` is an `int` after stdout is exhausted. `SandboxJobRunner.failure_message() -> str | None`.

The runner does not import `cwsandbox`. Task 4 supplies the real client.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from lelab.runners.sandbox import SandboxJobRunner
from lelab.train import TrainingRequest
from lelab.jobs import TrainingMetrics


class FakeProcess:
    def __init__(self, lines: list[str], code: int) -> None:
        self.stdout = iter(lines)
        self.returncode = code


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


def test_runner_streams_stdout_and_uploads_the_command(tmp_path: Path) -> None:
    handle = FakeHandle(FakeProcess(["Training:   1%| | 1/10 [00:01<00:09,  1.00step/s]\n"], 0))
    client = FakeClient(handle)
    log_path = tmp_path / "log.jsonl"
    runner = SandboxJobRunner(TrainingMetrics(), log_path, client)
    request = TrainingRequest(dataset_repo_id="me/so101", policy_type="act", tracking="wandb", skill="bolt")
    runner.start("job-1", request, str(tmp_path / "run"))
    deadline = 0
    while runner.is_running() and deadline < 50:
        import time
        time.sleep(0.02)
        deadline += 1
    assert runner.returncode() == 0
    assert handle.written[0][0] == "/work/train_argv.json"
    assert b"lelab.forge.remote.train_entry" in handle.written[0][1]
    assert handle.commands[0][0:3] == ["python", "-m", "lelab.forge.remote.train_entry"]
    assert client.created[0]["tags"] == ["lelab"]
    assert client.created[0]["resources"]["gpu"] == 1
    assert "log.jsonl" in log_path.read_text()
    lines = runner.stream_log_lines()
    assert any("Training:" in line.message for line in lines)


def test_create_failure_is_not_retried_and_offers_local(tmp_path: Path) -> None:
    client = FakeClient(error=RuntimeError("CWSANDBOX_RUNNER_UNAVAILABLE"))
    runner = SandboxJobRunner(TrainingMetrics(), tmp_path / "log.jsonl", client)
    request = TrainingRequest(dataset_repo_id="me/so101", policy_type="smolvla", tracking="wandb")
    runner.start("job-2", request, str(tmp_path / "run"))
    deadline = 0
    while runner.is_running() and deadline < 50:
        import time
        time.sleep(0.02)
        deadline += 1
    assert runner.returncode() == 1
    message = runner.failure_message()
    assert message is not None
    assert "CWSANDBOX_RUNNER_UNAVAILABLE" in message
    assert "Run this training on this machine instead." in message
    assert len(client.created) == 1


def test_stop_stops_the_sandbox(tmp_path: Path) -> None:
    handle = FakeHandle(FakeProcess([], 0))
    runner = SandboxJobRunner(TrainingMetrics(), tmp_path / "log.jsonl", FakeClient(handle))
    request = TrainingRequest(dataset_repo_id="me/so101", policy_type="act", tracking="off")
    runner.start("job-3", request, str(tmp_path / "run"))
    runner.stop()
    assert handle.stopped is True
```

The create-failure test needs the Forge env because `sandbox_create_kwargs` reads `load_forge_config()` inside the runner. Set `WANDB_API_KEY`, `WANDB_ENTITY`, `CWSANDBOX_API_KEY`, and `CWSANDBOX_TRAIN_IMAGE` with `monkeypatch` on every test in this file. Add a fourth test: the log file and `failure_message()` do not contain `wandb-secret` or `sandbox-secret` when `create` raises `RuntimeError("boom wandb-secret sandbox-secret")`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_sandbox_runner.py -q`

Expected: FAIL with import error for `lelab.runners.sandbox`.

- [ ] **Step 3: Write the implementation**

`lelab/runners/sandbox.py` starts a daemon thread from `start()` and returns immediately. The thread:

1. Calls `load_forge_config()` and `sandbox_create_kwargs(cfg, config.skill)`.
2. Calls `client.create(kwargs)` once. On any exception, sets `_returncode = 1` and `_failure` to `Could not start a CoreWeave Sandbox: {redact_secrets(str(exc), cfg)}. Run this training on this machine instead.` Writes that sentence as one log line through the same JSON-line shape as `SubprocessJobRunner._consume_lines` (reuse `_consume_lines` by subclassing `SubprocessJobRunner`, or append one `LogLine` and one JSON line). Does not call `create` again.
3. On success, `handle.write_file("/work/train_argv.json", dumps_command(cmd))`, then `process = handle.exec(cmd)`, then consumes `process.stdout` with the existing line pipeline (`parse_metrics_into`, tracking URL, queue, log file). After the iterator ends, `_returncode = process.returncode`.
4. `is_running()` is true while the thread is alive. `returncode()` is `None` until the thread finishes.
5. `stop()` sets the stop event, calls `handle.stop()` if a handle exists, and does not raise if `stop` itself raises. Log that failure with `redact_secrets`.
6. Never log `kwargs`, the environment, or the raw exception text. Log only `redact_secrets` output.

Subclassing `SubprocessJobRunner` is allowed. Do not call `_spawn` or `Popen`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_sandbox_runner.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lelab/runners/sandbox.py tests/test_forge_sandbox_runner.py
git commit -m "$(cat <<'EOF'
feat: stream a GPU Sandbox training into the job log

EOF
)"
```

---

### Task 4: Wire the runner, the sweep, and the guards

**Files:**
- Create: `lelab/runners/cwsandbox_client.py`
- Modify: `lelab/jobs.py` (`JobTarget.runner`, `JobRecord.runner`, `JobRegistry.start`, `_tick`, `_load_from_disk`)
- Modify: `lelab/server.py` startup and shutdown hooks
- Create: `tests/test_forge_sandbox_launch.py`
- Modify: `tests/test_lerobot_cli.py` (one argv parse test)

**Interfaces:**
- Consumes: `SandboxJobRunner`, `sandbox_create_kwargs`, `SANDBOX_TAG`, `sandbox_configured`.
- Produces: `CwsandboxClient.create(kwargs) -> handle` calling `Sandbox.run("sleep", "infinity", auth=AuthStrategy.COREWEAVE_API_KEY, container_image=..., placement_mode=..., tags=..., max_lifetime_seconds=..., resources=..., volumes=[ScratchVolumeOptions(...)], environment_variables=...).wait()`. `handle.write_file` calls `sandbox.write_file(path, contents).result()`. `handle.exec` returns the `Process` from `sandbox.exec(command)`. `handle.stop` calls `sandbox.stop(missing_ok=True).result()`. `sweep_lelab_sandboxes() -> int` lists `Sandbox.list(tags=["lelab"], auth=AuthStrategy.COREWEAVE_API_KEY).result()` and stops each. It returns the number stopped. It returns 0 without importing a failure when `sandbox_configured` is false. `JobTarget.runner` and `JobRecord.runner` accept `"coreweave"`.

- [ ] **Step 1: Write the failing tests**

`tests/test_forge_sandbox_launch.py`:

```python
def test_coreweave_requires_sandbox_settings(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
    monkeypatch.delenv("CWSANDBOX_TRAIN_IMAGE", raising=False)
    registry = JobRegistry(tmp_path)
    request = TrainingRequest(dataset_repo_id="me/so101", policy_type="act", tracking="wandb")
    with pytest.raises(ValueError, match="CWSANDBOX_API_KEY and CWSANDBOX_TRAIN_IMAGE"):
        registry.start(request, JobTarget(runner="coreweave"))


def test_coreweave_job_uses_the_sandbox_runner(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "wandb-secret")
    monkeypatch.setenv("WANDB_ENTITY", "coreweave-demo")
    monkeypatch.setenv("CWSANDBOX_API_KEY", "sandbox-secret")
    monkeypatch.setenv("CWSANDBOX_TRAIN_IMAGE", "registry.example/lelab-train:cuda")
    registry = JobRegistry(tmp_path)
    created = {}

    class Client:
        def create(self, kwargs):
            created["kwargs"] = kwargs
            raise RuntimeError("capacity")

    monkeypatch.setattr("lelab.runners.sandbox.CwsandboxClient", Client)
    request = TrainingRequest(dataset_repo_id="me/so101", policy_type="smolvla", tracking="wandb")
    record = registry.start(request, JobTarget(runner="coreweave"))
    deadline = 0
    while record.state == "running" and deadline < 50:
        time.sleep(0.05)
        deadline += 1
        record = registry.get(record.id)
    assert record.state == "failed"
    assert "capacity" in (record.error_message or "")
    assert "Run this training on this machine instead." in (record.error_message or "")
    assert "wandb-secret" not in (record.error_message or "")
    assert "sandbox-secret" not in (record.error_message or "")
    assert created["kwargs"]["resources"]["gpu"] == 1
```

The patch target must match where `JobRegistry.start` imports the client. Import `SandboxJobRunner` with a `client=` argument from `start()`, and in the test patch `lelab.runners.cwsandbox_client.CwsandboxClient` if that is the symbol `start` looks up. Assert `record.runner == "coreweave"`.

Add a test that `tracking="wandb"` and `JobTarget(runner="hf_cloud", flavor="a10g-small")` still raises `Weights & Biases tracking runs on this machine`.

Add a test that `sweep_lelab_sandboxes()` returns 0 when the Sandbox key is unset and that `cwsandbox` was not required to do so (call it inside `monkeypatch.delenv`).

Add `test_wandb_sandbox_argv_still_parses` in `tests/test_lerobot_cli.py` that builds `training_command` for a wandb ACT request and parses `cmd[3:]` with LeRobot's parser, asserting `--output_dir` is `/work/output` and `--trackio.project` is `so101-kitting`. Follow the existing `test_wandb_tracking_argv_still_parses` in that file.

In `_tick`, when finalizing, if `runner.failure_message()` returns a string, store that as `error_message` instead of `Job exited with code {rc}`.

In `_load_from_disk`, a persisted `runner == "coreweave"` record that is still `running` becomes `interrupted` with `error_message` `The Sandbox was stopped because the server restarted.` Do not try to reattach it.

Startup hook calls `sweep_lelab_sandboxes()`. Shutdown hook calls `job_registry.stop` for every running `coreweave` record, then `sweep_lelab_sandboxes()`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_sandbox_launch.py -q`

Expected: FAIL because `coreweave` is not a valid runner.

- [ ] **Step 3: Write the implementation**

`CwsandboxClient.create` maps the plain dict:

```python
from cwsandbox import AuthStrategy, Sandbox, ScratchVolumeOptions

sandbox = Sandbox.run(
    *kwargs["args"],
    auth=AuthStrategy.COREWEAVE_API_KEY,
    container_image=kwargs["container_image"],
    placement_mode=kwargs["placement_mode"],
    tags=kwargs["tags"],
    max_lifetime_seconds=kwargs["max_lifetime_seconds"],
    resources=kwargs["resources"],
    volumes=[ScratchVolumeOptions(**volume) for volume in kwargs["volumes"]],
    environment_variables=kwargs["environment_variables"],
).wait()
```

Do not pass `create_retry_budget_seconds`.

`JobRegistry.start` before the lock:

```python
if target.runner == "coreweave" and not sandbox_configured(load_forge_config()):
    raise ValueError(
        "CoreWeave Sandbox training needs CWSANDBOX_API_KEY and CWSANDBOX_TRAIN_IMAGE"
    )
```

Inside the lock, `runner == "coreweave"` constructs `SandboxJobRunner(record.metrics, log_path, CwsandboxClient())`. `wandb` plus `hf_cloud` is still rejected. `wandb` plus `coreweave` is allowed when both Forge and the Sandbox are configured. A missing Forge config with `tracking="wandb"` still raises `Weights & Biases tracking needs WANDB_API_KEY and WANDB_ENTITY` before the Sandbox check.

`sweep_lelab_sandboxes` logs each stop failure with `redact_secrets` and continues. It never logs the environment.

Install the SDK into the test interpreter before the green run:

`uv pip install --python /Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python 'cwsandbox>=1.14'`

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_sandbox_launch.py tests/test_forge_launch.py tests/test_jobs.py tests/test_lerobot_cli.py -q`

Expected: PASS. The HF-cloud rejection and the local wandb launch tests still pass.

- [ ] **Step 5: Commit**

```bash
git add lelab/runners/cwsandbox_client.py lelab/runners/sandbox.py lelab/jobs.py lelab/server.py tests/test_forge_sandbox_launch.py tests/test_lerobot_cli.py
git commit -m "$(cat <<'EOF'
feat: run a training job on a CoreWeave GPU Sandbox

EOF
)"
```

---

### Task 5: Train page target and Setup status

**Files:**
- Modify: `frontend/src/components/training/types.ts`
- Modify: `frontend/src/lib/jobsApi.ts`
- Modify: `frontend/src/components/training/config/TargetCard.tsx`
- Modify: `frontend/src/pages/Training.tsx`
- Modify: `frontend/src/pages/Setup.tsx`
- Modify: `frontend/src/components/jobs/JobCard.tsx` only if a runner badge there would otherwise say Local for a CoreWeave job
- Test: `frontend/src/components/training/target.test.ts`

**Interfaces:**
- Consumes: `GET /forge-status` now returns `sandbox_configured`. `POST /jobs/training` accepts `target.runner = "coreweave"`.
- Produces: a Compute target option `CoreWeave Sandbox`. It is disabled while `sandbox_configured` is false. Selecting it sets `{ runner: "coreweave" }` and skips the local policy-extra preflight, the same way `hf_cloud` already does. The job page badge reads `CoreWeave Sandbox` instead of `Local`.

- [ ] **Step 1: Write the failing test**

`frontend/src/components/training/target.ts`:

```ts
export function targetFromSelection(value: string): { runner: "local" | "hf_cloud" | "coreweave"; flavor?: string } {
  if (value === "local") return { runner: "local" };
  if (value === "coreweave") return { runner: "coreweave" };
  return { runner: "hf_cloud", flavor: value.slice("hf:".length) };
}

export function setupSandboxDetail(forgeConfigured: boolean, sandboxConfigured: boolean, entity: string): string {
  if (!forgeConfigured) {
    return "Add WANDB_API_KEY and WANDB_ENTITY to the backend environment.";
  }
  if (!sandboxConfigured) {
    return `Weights & Biases is connected as ${entity}. Add CWSANDBOX_API_KEY and CWSANDBOX_TRAIN_IMAGE to run training on a GPU Sandbox.`;
  }
  return `Weights & Biases is connected as ${entity}. GPU Sandbox training is available.`;
}
```

`frontend/src/components/training/target.test.ts` asserts `targetFromSelection("coreweave")` is `{ runner: "coreweave" }`, `targetFromSelection("local")` is `{ runner: "local" }`, and `targetFromSelection("hf:a10g-small")` is `{ runner: "hf_cloud", flavor: "a10g-small" }`. Assert the three `setupSandboxDetail` sentences above, including that none of them contain `API key values` or the words `local-test-key`.

- [ ] **Step 2: Run the test to verify it fails**

Run from `frontend/`: `npx vitest run src/components/training/target.test.ts`

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Write the implementation**

Use `targetFromSelection` in `TargetCard`. Add a `sandboxConfigured: boolean` prop. Render:

```tsx
<SelectItem value="coreweave" disabled={!sandboxConfigured}>
  CoreWeave Sandbox
</SelectItem>
```

When the option is disabled, the existing hint under the select says `Add CWSANDBOX_API_KEY and CWSANDBOX_TRAIN_IMAGE to the backend environment.` When the selected runner is `coreweave`, the hint says `RTX PRO 6000 on CoreWeave. The Sandbox is deleted when the job stops.`

`Training.tsx` loads `/forge-status` once and passes `sandboxConfigured`. The policy-extra preflight stays limited to `runner === "local"`. The job badge treats `coreweave` as `CoreWeave Sandbox`.

`Setup.tsx` uses `setupSandboxDetail`. Remove the sentences `Sandbox training arrives in a later milestone.` and `This arrives in the next milestone.`

Widen the TypeScript unions in `types.ts` and `jobsApi.ts` from `"local" | "hf_cloud"` to include `"coreweave"`.

- [ ] **Step 4: Run the tests to verify they pass**

Run from `frontend/`: `npx vitest run src/components/training/target.test.ts` then `npm test` and `npm run build`.

Expected: the new test passes, the full frontend suite passes, and the build exits 0.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/training/target.ts frontend/src/components/training/target.test.ts frontend/src/components/training/types.ts frontend/src/components/training/config/TargetCard.tsx frontend/src/lib/jobsApi.ts frontend/src/pages/Training.tsx frontend/src/pages/Setup.tsx frontend/src/components/jobs/JobCard.tsx
git commit -m "$(cat <<'EOF'
feat: start GPU Sandbox training from the Train page

EOF
)"
```

---

## Self-review

- Spec section 4.2 `SandboxJobRunner` and `train_entry.py`: Tasks 3 and 4.
- Spec section 5 step 3, one SmolVLA and one ACT per skill: the existing policy and skill fields, one job each. No batch launcher.
- Spec section 8.1 create failure and orphaned Sandboxes: Task 3 failure message, Task 4 single create, stop, startup sweep, shutdown sweep, reload marks the job interrupted.
- Spec section 8.2 argv parsed by LeRobot, live tests marked `forge` and skipped by default: Task 4 parse test. No live test is required to pass in CI. Do not add a test that creates a real Sandbox.
- Spec section 10 M3 and section 11 image, lifetime, volume: recorded in the header from the 2026-11-13 SDK check. Image is an explicit env var because the docs do not publish a LeRobot CUDA image.
- Secrets stay off the frontend and out of logs: Tasks 1, 3, and 4.
- Forge off still behaves like upstream: the new target is rejected without the two Sandbox env vars, and local / HF Jobs paths are unchanged.
