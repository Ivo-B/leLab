# LeLab Forge M2 — local W&B tracking, dataset Artifacts, Registry linking

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local training run and a finished recording show up in Forge as a W&B run, a dataset Artifact, and a Registry collection alias, while Forge switched off still trains and records exactly as upstream leLab.

**Architecture:** `ForgeConfig` reads the environment. Recording calls a dataset logger after `finalize()`. Local training with Weights & Biases selected launches `lelab.forge.remote.train_entry`, which replaces LeRobot's `TrackioLogger` with `WandbLogger` before `lerobot_train.main()`. LeRobot only constructs that logger when `--trackio.enable=true` and `--trackio.project` are set, so a Forge run passes those two flags and no Space id. The logger speaks W&B, not Trackio. `finish()` links the last model Artifact into the Registry with alias `candidate`.

**Tech Stack:** Python 3.12, FastAPI, the installed LeRobot v0.7 (`TrackioLogger` in `lerobot.common.trackio_utils`), `wandb>=0.19`, pytest, the existing React train form.

**Spec:** `docs/superpowers/specs/2026-11-13-lelab-forge-so101-design.md` sections 4.1, 4.4, 5 (steps 1 and 4), and milestone M2 in section 10.

## Global Constraints

- With Forge switched off, training argv and recording behavior stay as they are today. Hugging Face Hub push of datasets stays.
- Do not fork LeRobot. Do not add the CoreWeave Sandbox SDK. Sandbox training is M3.
- New Python files start with the Apache-2.0 header copied from `tests/test_train.py`.
- Product names, when they appear: Weights & Biases Models, CoreWeave Registry. New UI copy is sentence case.
- Dataset Artifact name is `so101-kitting`, type `dataset`. W&B assigns `:vN`. Do not put the version in the name.
- Registry path is `{entity}/wandb-registry-model/{collection}`. Collections are `so101-kitting-smolvla` and `so101-kitting-act-<skill>`. The alias this milestone writes is `candidate`. Do not move `production`.
- `WandbLogger` exposes `log_dict`, `log_policy`, `log_video`, and `finish` with the same parameter lists as `TrackioLogger`. A failure to log or link is logged and swallowed, matching Trackio's best-effort artifact upload.
- Tests must not contact the network. Monkeypatch `wandb`. Run `pytest tests/test_forge_config.py tests/test_forge_registry.py tests/test_forge_tracking.py tests/test_train.py tests/test_lerobot_cli.py -q` from the repo root for the tasks that name those files. Frontend tests: `npm test` from `frontend/`.
- New UI uses theme tokens (`text-foreground`, `text-muted-foreground`, `border-border`). Do not add `text-white` on a theme background.

---

### Task 1: Forge configuration and status

**Files:**

- Create: `lelab/utils/forge_config.py`
- Create: `tests/test_forge_config.py`
- Modify: `lelab/server.py` (add `GET /forge-status` near the other status routes)
- Modify: `pyproject.toml` dependencies (add `wandb`)

**Interfaces:**

- Consumes: environment variables `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`
- Produces: `ForgeConfig` with fields `api_key: str | None`, `entity: str | None`, `project: str`, and `def forge_configured(cfg: ForgeConfig) -> bool`. `project` defaults to `"so101-kitting"`. Configured means both `api_key` and `entity` are non-empty.

- [ ] **Step 1: Write the failing test**

```python
from lelab.utils.forge_config import ForgeConfig, forge_configured, load_forge_config


def test_missing_credentials_are_not_configured(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    cfg = load_forge_config()
    assert cfg == ForgeConfig(api_key=None, entity=None, project="so101-kitting")
    assert forge_configured(cfg) is False


def test_key_and_entity_configure_forge(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "local-test-key")
    monkeypatch.setenv("WANDB_ENTITY", "coreweave-demo")
    monkeypatch.setenv("WANDB_PROJECT", "so101")
    cfg = load_forge_config()
    assert cfg.api_key == "local-test-key"
    assert cfg.entity == "coreweave-demo"
    assert cfg.project == "so101"
    assert forge_configured(cfg) is True


def test_blank_entity_is_not_configured(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "local-test-key")
    monkeypatch.setenv("WANDB_ENTITY", "  ")
    assert forge_configured(load_forge_config()) is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_forge_config.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'lelab.utils.forge_config'`

- [ ] **Step 3: Implement config, the status route, and the dependency**

`lelab/utils/forge_config.py`:

```python
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
```

In `pyproject.toml` `dependencies`, add `"wandb>=0.19"` after the `lerobot` line.

In `lelab/server.py`, add this route next to the other `GET` status routes:

```python
@app.get("/forge-status")
def forge_status():
    from .utils.forge_config import forge_configured, load_forge_config

    cfg = load_forge_config()
    return {
        "forge_configured": forge_configured(cfg),
        "entity": cfg.entity,
        "project": cfg.project,
    }
```

Do not return `api_key`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_forge_config.py -q`
Expected: PASS, 3 tests.

Add this test to the same file and run it too:

```python
def test_forge_status_hides_the_api_key(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from lelab.server import app

    monkeypatch.setenv("WANDB_API_KEY", "local-test-key")
    monkeypatch.setenv("WANDB_ENTITY", "coreweave-demo")
    client = TestClient(app)
    response = client.get("/forge-status")
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "forge_configured": True,
        "entity": "coreweave-demo",
        "project": "so101-kitting",
    }
    assert "local-test-key" not in response.text
```

Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml lelab/utils/forge_config.py lelab/server.py tests/test_forge_config.py docs/superpowers/plans/2026-11-13-lelab-forge-m2-tracking.md
git commit -m "feat: read Forge tracking settings from the environment"
```

---

### Task 2: Registry collection names and episode counts

**Files:**

- Create: `lelab/forge/__init__.py` (empty aside from the license header)
- Create: `lelab/forge/registry.py`
- Create: `tests/test_forge_registry.py`

**Interfaces:**

- Consumes: `read_episode_index` from `lelab.episode_media` (`EpisodeIndexRow.tasks`)
- Produces:
  - `collection_name(policy_type: str, skill: str | None) -> str`
  - `registry_path(entity: str, collection: str) -> str`
  - `episode_counts(dataset_dir: Path) -> dict[str, int]`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_forge_registry.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'lelab.forge'`

- [ ] **Step 3: Implement**

`lelab/forge/registry.py`:

```python
import re
from pathlib import Path

_SKILL_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def collection_name(policy_type: str, skill: str | None) -> str:
    if policy_type == "smolvla":
        return "so101-kitting-smolvla"
    if policy_type == "act":
        if skill is None or _SKILL_SLUG.fullmatch(skill) is None:
            raise ValueError(
                "ACT registry linking needs a skill slug such as bolt, bracket, or gear"
            )
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_forge_registry.py -q`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add lelab/forge/__init__.py lelab/forge/registry.py tests/test_forge_registry.py
git commit -m "feat: name the Registry collections for SmolVLA and ACT"
```

---

### Task 3: Dataset Artifact after recording

**Files:**

- Create: `lelab/forge/tracking.py` (dataset function only in this task; Task 4 adds `WandbLogger` to the same file)
- Create: `tests/test_forge_tracking.py`
- Modify: `lelab/record.py` after the Hub push block that ends near line 1119

**Interfaces:**

- Consumes: `load_forge_config`, `forge_configured`, `episode_counts`
- Produces: `log_dataset_if_configured(dataset_dir: Path, repo_id: str) -> str | None`. Returns the run URL when a run is opened, otherwise `None`. Never raises.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from lelab.forge.tracking import log_dataset_if_configured


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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_forge_tracking.py -q`
Expected: FAIL with `ImportError` for `log_dataset_if_configured`

- [ ] **Step 3: Implement and call it from recording**

Add to `lelab/forge/tracking.py`:

```python
import logging
from pathlib import Path

from lelab.forge.registry import episode_counts
from lelab.utils.forge_config import forge_configured, load_forge_config

logger = logging.getLogger(__name__)


def log_dataset_if_configured(dataset_dir: Path, repo_id: str) -> str | None:
    cfg = load_forge_config()
    if not forge_configured(cfg):
        return None
    try:
        import wandb

        run = wandb.init(
            entity=cfg.entity,
            project=cfg.project,
            job_type="record",
            name=f"dataset-{repo_id.replace('/', '-')}",
        )
        artifact = wandb.Artifact(
            "so101-kitting",
            type="dataset",
            metadata={"repo_id": repo_id, "episode_counts": episode_counts(dataset_dir)},
        )
        artifact.add_dir(str(dataset_dir))
        run.log_artifact(artifact)
        run.finish()
        return run.url
    except Exception:
        logger.exception("Dataset artifact was not logged for %s", repo_id)
        return None
```

In `lelab/record.py`, immediately after the `if cfg.dataset.push_to_hub:` block (after the `else` that logs "No episodes saved"), and before `log_say("Exiting"...`:

```python
    if dataset.num_episodes > 0:
        from lelab.forge.tracking import log_dataset_if_configured

        log_dataset_if_configured(Path(dataset.root), cfg.dataset.repo_id)
```

`Path` is already imported in `record.py`. If it is not, import it from `pathlib`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_forge_tracking.py -q`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add lelab/forge/tracking.py lelab/record.py tests/test_forge_tracking.py
git commit -m "feat: log a dataset Artifact when a recording finishes"
```

---

### Task 4: WandbLogger with the Trackio surface

**Files:**

- Modify: `lelab/forge/tracking.py`
- Modify: `tests/test_forge_tracking.py`

**Interfaces:**

- Consumes: `TrainPipelineConfig` (only as the object LeRobot passes in; tests use a stub with `.trackio.project`, `.policy.type`, and `.to_dict()`), `collection_name`, `registry_path`, `load_forge_config`
- Produces: `class WandbLogger` constructed as `WandbLogger(cfg)`. Reads the skill slug from `FORGE_SKILL`. `log_policy` records the last logged model Artifact. `finish` links that artifact at the registry path with `aliases=["candidate"]` when `collection_name` succeeds, then calls `wandb.finish()`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_forge_tracking.py`:

```python
import inspect
from pathlib import Path

from lerobot.common.trackio_utils import TrackioLogger


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
    from lelab.forge.tracking import WandbLogger
    import wandb

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
```

The third test reaches `logger._run`. Keep that attribute name.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_forge_tracking.py::test_wandb_logger_matches_trackio_parameters -q`
Expected: FAIL with `ImportError` for `WandbLogger`

- [ ] **Step 3: Implement WandbLogger**

Add this class to `lelab/forge/tracking.py`. Match `TrackioLogger` parameter lists exactly, including `custom_step_key` on `log_dict`.

```python
import os

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE

from lelab.forge.registry import collection_name, registry_path
from lerobot.utils.constants import PRETRAINED_MODEL_DIR


class WandbLogger:
    def __init__(self, cfg) -> None:
        import wandb

        self._wandb = wandb
        self._cfg = cfg
        self._run = wandb.init(
            entity=load_forge_config().entity,
            project=cfg.trackio.project,
            job_type="train",
            config=cfg.to_dict(),
        )
        self._last_artifact = None
        logger.info("Track this run --> %s", self._run.url)

    def log_policy(self, checkpoint_dir) -> None:
        try:
            weights_dir = Path(checkpoint_dir) / PRETRAINED_MODEL_DIR
            artifact = self._wandb.Artifact(f"policy-{Path(checkpoint_dir).name}", type="model")
            adapter = weights_dir / "adapter_model.safetensors"
            standard = weights_dir / SAFETENSORS_SINGLE_FILE
            if adapter.exists():
                artifact.add_file(adapter)
            elif standard.exists():
                artifact.add_file(standard)
            else:
                logger.warning("No model weights in %s", weights_dir)
                return
            self._run.log_artifact(artifact)
            self._last_artifact = artifact
        except Exception:
            logger.exception("Model artifact was not logged for %s", checkpoint_dir)

    def log_dict(self, d, step=None, mode="train", custom_step_key=None) -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")
        payload = {}
        for key, value in d.items():
            if isinstance(value, (int, float, str)):
                payload[f"{mode}/{key}"] = value
        if not payload:
            return
        if custom_step_key is not None:
            self._run.log(payload)
        else:
            self._run.log(payload, step=step)

    def log_video(self, video_path, step, mode="train") -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        video = self._wandb.Video(video_path, caption=f"{mode} step {step}")
        self._run.log({f"{mode}/video": video}, step=step)

    def finish(self) -> None:
        try:
            self._link_last_checkpoint()
        except Exception:
            logger.exception("Checkpoint was not linked into the Registry")
        finally:
            self._run.finish()

    def _link_last_checkpoint(self) -> None:
        if self._last_artifact is None:
            return
        cfg = load_forge_config()
        if not forge_configured(cfg):
            return
        skill = os.environ.get("FORGE_SKILL")
        skill = skill.strip() if skill else None
        collection = collection_name(self._cfg.policy.type, skill or None)
        self._run.link_artifact(
            self._last_artifact,
            registry_path(cfg.entity, collection),
            aliases=["candidate"],
        )
```

`collection_name` raises `ValueError` for ACT without a slug. `_link_last_checkpoint` lets that propagate to `finish`, which logs it and still calls `wandb.finish()`. The skip-link test expects no raise and `linked is None`. Catch `ValueError` inside `_link_last_checkpoint`, log it with `logger.warning`, and return:

```python
        try:
            collection = collection_name(self._cfg.policy.type, skill or None)
        except ValueError as exc:
            logger.warning("%s", exc)
            return
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_forge_tracking.py -q`
Expected: PASS. The Trackio signature test fails if the parameter lists differ. Fix the class until the lists match; do not change the test to be weaker.

- [ ] **Step 5: Commit**

```bash
git add lelab/forge/tracking.py tests/test_forge_tracking.py
git commit -m "feat: log local training to Weights & Biases with a candidate alias"
```

---

### Task 5: Launch the logger wrapper for a local Forge run

**Files:**

- Create: `lelab/forge/remote/__init__.py`
- Create: `lelab/forge/remote/train_entry.py`
- Modify: `lelab/train.py` (`TrainingRequest.tracking`, `TrainingRequest.skill`, `build_training_command`)
- Modify: `lelab/jobs.py` (`_spawn` extra env, `JobRegistry.start` guard, `LocalJobRunner.start`)
- Modify: `tests/test_train.py`
- Modify: `tests/test_lerobot_cli.py` (one new test; do not change the existing Trackio parse test)

**Interfaces:**

- Consumes: `WandbLogger`, `forge_configured`, `load_forge_config`
- Produces:
  - `TrainingRequest.tracking: Literal["off", "trackio", "wandb"] = "off"`
  - `TrainingRequest.skill: str | None = None`
  - When `tracking == "wandb"`, argv starts with `[python, "-m", "lelab.forge.remote.train_entry"]`, includes `--trackio.enable true` and `--trackio.project` equal to `load_forge_config().project`, and omits `--trackio.space_id`.
  - `train_entry.main()` assigns `lerobot.scripts.lerobot_train.TrackioLogger = WandbLogger` and then calls that module's `main()`.
  - `JobRegistry.start` raises `ValueError` with message `Weights & Biases tracking needs WANDB_API_KEY and WANDB_ENTITY` when `tracking == "wandb"` and Forge is not configured.
  - `JobRegistry.start` raises `ValueError` with message `Weights & Biases tracking runs on this machine` when `tracking == "wandb"` and `target.runner == "hf_cloud"`.
  - `LocalJobRunner` puts `FORGE_SKILL` in the child environment when `skill` is set, and does not set it when `skill` is `None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_train.py`:

```python
def test_wandb_tracking_uses_the_wrapper_and_no_space(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_PROJECT", "so101-kitting")
    from lelab.train import TrainingRequest, build_training_command

    cmd = build_training_command(
        TrainingRequest(dataset_repo_id="me/so101", tracking="wandb", skill="bolt", policy_type="act"),
        "/tmp/out",
    )
    assert cmd[:3] == ["python", "-m", "lelab.forge.remote.train_entry"]
    assert _arg_value(cmd, "--trackio.enable") == "true"
    assert _arg_value(cmd, "--trackio.project") == "so101-kitting"
    assert "--trackio.space_id" not in cmd


def test_trackio_argv_is_unchanged_when_forge_tracking_is_off() -> None:
    from lelab.train import TrainingRequest, build_training_command

    cmd = build_training_command(TrainingRequest(dataset_repo_id="x"), "/tmp/out")
    assert cmd[:3] == ["python", "-m", "lerobot.scripts.lerobot_train"]
    assert _arg_value(cmd, "--trackio.enable") == "false"
```

Append to `tests/test_lerobot_cli.py`:

```python
def test_wandb_tracking_argv_still_parses(monkeypatch) -> None:
    from lelab.train import TrainingRequest, build_training_command

    monkeypatch.setenv("WANDB_PROJECT", "so101-kitting")
    cmd = build_training_command(
        TrainingRequest(dataset_repo_id="me/dataset", policy_type="act", policy_device="cpu", tracking="wandb"),
        "/tmp/out",
    )
    monkeypatch.setattr(sys, "argv", ["lerobot-train", *cmd[3:]])
    cfg = _parse_training()
    assert cfg.trackio.enable is True
    assert cfg.trackio.project == "so101-kitting"
    assert cfg.trackio.space_id is None
```

Add `tests/test_forge_launch.py`:

```python
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
    runner.start("job", TrainingRequest(dataset_repo_id="me/so101", tracking="wandb", skill="gear"), str(tmp_path))
    assert seen["cmd"][2] == "lelab.forge.remote.train_entry"
    assert seen["env"]["FORGE_SKILL"] == "gear"
    assert "FORGE_SKILL" not in os.environ
```

The last test constructs `LocalJobRunner` without `__init__`. If `start` reads attributes set in `__init__`, call the real constructor instead. Do not weaken the assertions.

Add `tests/test_train_entry.py`:

```python
def test_train_entry_swaps_the_logger_before_main(monkeypatch) -> None:
    import lerobot.scripts.lerobot_train as train_mod

    from lelab.forge.remote import train_entry
    from lelab.forge.tracking import WandbLogger

    called = {}

    def fake_main() -> None:
        called["logger"] = train_mod.TrackioLogger

    original = train_mod.TrackioLogger
    monkeypatch.setattr(train_mod, "main", fake_main)
    try:
        train_entry.main()
        assert called["logger"] is WandbLogger
    finally:
        train_mod.TrackioLogger = original
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_train.py::test_wandb_tracking_uses_the_wrapper_and_no_space tests/test_train_entry.py -q`
Expected: FAIL because `tracking` is not a field and `train_entry` does not exist.

- [ ] **Step 3: Implement**

`lelab/forge/remote/train_entry.py`:

```python
def main() -> None:
    import lerobot.scripts.lerobot_train as train_mod

    from lelab.forge.tracking import WandbLogger

    train_mod.TrackioLogger = WandbLogger
    train_mod.main()


if __name__ == "__main__":
    main()
```

In `TrainingRequest`, add:

```python
tracking: Literal["off", "trackio", "wandb"] = "off"
skill: str | None = None
```

Import `Literal` from `typing`.

In `build_training_command`, set the module from the request:

```python
module = "lelab.forge.remote.train_entry" if request.tracking == "wandb" else "lerobot.scripts.lerobot_train"
cmd = [python_executable, "-m", module]
```

Replace the Trackio block with:

```python
    if request.tracking == "wandb":
        from lelab.utils.forge_config import load_forge_config

        cmd.extend(["--trackio.enable", "true", "--trackio.project", load_forge_config().project])
    else:
        cmd.extend(["--trackio.enable", "true" if request.trackio_enable else "false"])
        if request.trackio_enable:
            if request.trackio_project:
                cmd.extend(["--trackio.project", request.trackio_project])
            if request.trackio_space_id:
                cmd.extend(["--trackio.space_id", request.trackio_space_id, "--trackio.private", "true"])
```

Change `_spawn` to `def _spawn(self, cmd: list[str], thread_name: str, extra_env: dict[str, str] | None = None) -> None` and after copying `os.environ`:

```python
        if extra_env:
            child_env.update(extra_env)
```

In `LocalJobRunner.start`:

```python
        extra_env = {"FORGE_SKILL": config.skill} if config.skill else None
        self._spawn(cmd, thread_name=f"job-{job_id}-stdout", extra_env=extra_env)
```

In `JobRegistry.start`, before creating the job id:

```python
        if config.tracking == "wandb":
            from lelab.utils.forge_config import forge_configured, load_forge_config

            if target.runner == "hf_cloud":
                raise ValueError("Weights & Biases tracking runs on this machine")
            if not forge_configured(load_forge_config()):
                raise ValueError("Weights & Biases tracking needs WANDB_API_KEY and WANDB_ENTITY")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_train.py tests/test_lerobot_cli.py tests/test_forge_launch.py tests/test_train_entry.py tests/test_jobs.py -q`
Expected: PASS, including the existing Trackio argv test and `test_training_argv_parses`.

- [ ] **Step 5: Commit**

```bash
git add lelab/forge/remote lelab/train.py lelab/jobs.py tests/test_train.py tests/test_lerobot_cli.py tests/test_forge_launch.py tests/test_train_entry.py
git commit -m "feat: run local Forge training through the W&B logger"
```

---

### Task 6: Setup status and the train-form switch

**Files:**

- Modify: `frontend/src/pages/Setup.tsx`
- Modify: `frontend/src/components/training/types.ts`
- Modify: `frontend/src/lib/jobsApi.ts` (`TrainingRequest` fields)
- Modify: `frontend/src/pages/Training.tsx` (`configToRequest` and the initial config)
- Modify: `frontend/src/components/training/config/EssentialsCard.tsx`
- Create: `frontend/src/components/training/tracking.ts`
- Create: `frontend/src/components/training/tracking.test.ts`

**Interfaces:**

- Consumes: `GET /forge-status` body `{ forge_configured, entity, project }` and `useApi().baseUrl`
- Produces: `trackingPayload(config)` returns `{ tracking: "wandb", skill }` or `{ tracking: "off" }` or `{ tracking: "trackio" }`. `configToRequest` spreads that object and still sends the existing `trackio_*` fields. The skill input is a slug: lowercase letters, digits, and single hyphens.

- [ ] **Step 1: Write the failing test**

`frontend/src/components/training/tracking.ts` does not exist yet. `frontend/src/components/training/tracking.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import { trackingPayload } from "./tracking";

const base = {
  trackio_enable: false,
  wandb_enable: false,
  skill: "",
};

describe("trackingPayload", () => {
  it("leaves tracking off by default", () => {
    expect(trackingPayload(base)).toEqual({ tracking: "off" });
  });

  it("sends the skill slug with Weights & Biases", () => {
    expect(
      trackingPayload({ ...base, wandb_enable: true, skill: "bracket" }),
    ).toEqual({
      tracking: "wandb",
      skill: "bracket",
    });
  });

  it("prefers Weights & Biases when both switches are on", () => {
    expect(
      trackingPayload({
        ...base,
        trackio_enable: true,
        wandb_enable: true,
        skill: "bolt",
      }).tracking,
    ).toBe("wandb");
  });

  it("omits a blank skill", () => {
    expect(
      trackingPayload({ ...base, wandb_enable: true, skill: "  " }),
    ).toEqual({ tracking: "wandb" });
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run from `frontend/`: `npx vitest run src/components/training/tracking.test.ts`
Expected: FAIL because `./tracking` cannot be resolved.

- [ ] **Step 3: Implement the helper and the two screens**

`frontend/src/components/training/tracking.ts`:

```typescript
export function trackingPayload(config: {
  trackio_enable: boolean;
  wandb_enable: boolean;
  skill?: string;
}): { tracking: "off" | "trackio" | "wandb"; skill?: string } {
  if (config.wandb_enable) {
    const skill = config.skill?.trim();
    return skill ? { tracking: "wandb", skill } : { tracking: "wandb" };
  }
  if (config.trackio_enable) return { tracking: "trackio" };
  return { tracking: "off" };
}
```

Add `wandb_enable: boolean` and `skill?: string` to `TrainingConfig` in `types.ts`. Add `tracking?: "off" | "trackio" | "wandb"` and `skill?: string` to the request type in `jobsApi.ts`.

In `configToRequest`, spread `...trackingPayload(c)` beside the existing `trackio_*` fields. Initial state sets `wandb_enable: false`.

In `EssentialsCard.tsx`, under the Trackio switch, add a second switch:

```tsx
<div className="flex items-center space-x-3 pt-6">
  <Switch
    id="wandb_enable"
    checked={config.wandb_enable}
    onCheckedChange={(checked) => updateConfig("wandb_enable", checked)}
  />
  <Label htmlFor="wandb_enable">Track with Weights & Biases</Label>
</div>
```

When `config.wandb_enable && config.policy_type === "act"`, show an input labeled "Skill" with placeholder `bolt`, bound to `config.skill`. Its detail text is `Used for the CoreWeave Registry collection.`

In `Setup.tsx`, fetch `${baseUrl}/forge-status` once with the existing `useApi` hook. While the request is in flight, keep the current Connect Forge sentence. On success with `forge_configured`, the detail is `Weights & Biases is connected as {entity}. Sandbox training arrives in a later milestone.` On failure or `forge_configured: false`, the detail is `Add WANDB_API_KEY and WANDB_ENTITY to the backend environment. Sandbox training arrives in a later milestone.` Remove the sentence `This arrives in the next milestone.`

- [ ] **Step 4: Run the tests and the build**

Run from `frontend/`: `npm test` and `npm run build`
Expected: the tracking tests pass, the existing suite stays green, and the build exits 0.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/Setup.tsx frontend/src/pages/Training.tsx frontend/src/components/training frontend/src/lib/jobsApi.ts
git commit -m "feat: choose Weights & Biases from the train form"
```

---

## Self-review

Spec coverage:

- Section 5 step 1 (dataset Artifact with per-skill counts, Hub push unchanged): Task 3.
- Section 5 step 4 for a local run (model Artifact, `candidate` alias, one collection per model): Tasks 2 and 4. `production` is intentionally not moved.
- Section 4.4 (four methods, no LeRobot fork): Task 4 contract test and Task 5 wrapper.
- Section 4.1 upstream default: Tasks 1 and 5. `tracking="off"` keeps today's argv.
- Milestone M2 demo (lineage in Forge): Tasks 3, 4, 5, and 6. A real run still needs `WANDB_API_KEY` and `WANDB_ENTITY` in the environment; the tests never call W&B.
- Out of scope, confirmed absent: `SandboxJobRunner`, Weave, the order agent, ARIA, curation notebook.

Placeholder scan: no TBD, no "write tests for the above" without the test body, no "similar to Task N".

Type consistency: `tracking` is `"off" | "trackio" | "wandb"` in Python and TypeScript. `collection_name`, `registry_path`, `log_dataset_if_configured`, and `WandbLogger` are named the same in the tests and the implementations. `FORGE_SKILL` is the only environment channel for the skill slug.
