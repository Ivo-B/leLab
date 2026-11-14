# LeLab Forge M6 — Improve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Closing an evaluation has a written ARIA automation, the curation rules live in tested code and a marimo notebook, and the Improve page plus the README tell a person how to run that loop.

**Architecture:** `aria_prompt()` returns the automation name, the event, and a prompt under 4,000 characters. `flag_episode` marks idle starts, dropped frames, and outlier lengths. The notebook downloads the `so101-kitting:latest` artifact and calls `flag_episode`. The Improve page shows that text when Forge is on. The README carries the same steps and the W&B Report outline. Nothing in this plan calls W&B, ARIA, or the arm.

**Tech Stack:** Python 3.12, FastAPI already serving `/forge-status`, React + Vitest, marimo source that CoreWeave Notebooks can open.

## Global Constraints

- Work on branch `forge/m6-improve`. Push only to `origin` (`https://github.com/Ivo-B/leLab.git`). Do not push `upstream`.
- Do not redo M5. No planner transcript changes, no operator-question UI, no rollout from `pick`, no camera capture.
- Do not start the arm. Do not call `wandb`, ARIA, or the network from tests.
- Do not invent a W&B or ARIA URL. The page and README name the project Automations tab and the action **Trigger ARIA**.
- The prompt is at most 4,000 characters. Checked against the CoreWeave docs on 2026-11-14: an ARIA automation is a project automation whose action is Trigger ARIA, available on W&B Multi-tenant Cloud when Smart features are on. A run event can be "a run reaches FINISHED". Notebooks can download an artifact with `run.use_artifact` and `artifact.download()`. Sandbox files are temporary.
- New Python modules start with the Apache-2.0 header in `tests/test_forge_config.py` lines 1–13, then a blank line.
- Pytest is not on `PATH`. Run `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest`.
- Commit only the files the task lists. Do not commit `graphify-out/` or `.env`. Do not skip hooks.

---

### Task 1: Write the ARIA automation

**Files:**

- Create: `lelab/improve/__init__.py`
- Create: `lelab/improve/aria.py`
- Create: `tests/test_improve_aria.py`
- Modify: `docs/superpowers/plans/2026-11-14-lelab-forge-m6-improve.md` if it is still untracked

**Interfaces:**

- Consumes: nothing.
- Produces: `AUTOMATION_NAME = "so101-kitting evaluation"`, `AUTOMATION_EVENT = "A run reaches FINISHED"`, `aria_prompt() -> str`. The prompt names the stages `started`, `reached`, `grasped`, and `delivered`, asks for the failing skill and stage, asks for a next-experiment proposal, and says not to change an alias or launch a job. `len(aria_prompt()) <= 4000`.

- [x] **Step 1: Write the failing test**

```python
from lelab.improve.aria import AUTOMATION_EVENT, AUTOMATION_NAME, aria_prompt


def test_prompt_fits_the_aria_limit_and_names_the_failure() -> None:
    prompt = aria_prompt()
    assert AUTOMATION_NAME == "so101-kitting evaluation"
    assert AUTOMATION_EVENT == "A run reaches FINISHED"
    assert len(prompt) <= 4000
    assert "grasped" in prompt
    assert "do not change an alias" in prompt.lower()
    assert "http" not in prompt.lower()
```

- [x] **Step 2: Run the test to verify it fails**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_improve_aria.py -q`

Expected: FAIL with `ModuleNotFoundError`

- [x] **Step 3: Write the minimal implementation**

`lelab/improve/__init__.py` is only the Apache-2.0 header.

```python
AUTOMATION_NAME = "so101-kitting evaluation"
AUTOMATION_EVENT = "A run reaches FINISHED"


def aria_prompt() -> str:
    return (
        "Look at the Weave evaluation that just finished in this project. "
        "Name the skill and the stage where trials failed. "
        "The stages are started, reached, grasped, and delivered. "
        "A trial with no stage needs review and counts as a failure. "
        "Propose the next experiment: which skill needs more demonstrations, "
        "and whether to retrain SmolVLA or the ACT policy for that skill. "
        "Do not change an alias and do not launch a job."
    )
```

- [x] **Step 4: Run the test to verify it passes**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_improve_aria.py -q`

Expected: all passed.

- [x] **Step 5: Commit**

```bash
git add lelab/improve/__init__.py lelab/improve/aria.py tests/test_improve_aria.py docs/superpowers/plans/2026-11-14-lelab-forge-m6-improve.md
git commit -m "$(cat <<'EOF'
feat: describe the evaluation ARIA automation

EOF
)"
```

---

### Task 2: Flag episodes worth curating

**Files:**

- Create: `lelab/improve/curate.py`
- Create: `tests/test_improve_curate.py`

**Interfaces:**

- Consumes: nothing from Task 1.
- Produces: `IDLE_START_S = 1.0` and `flag_episode(start_idle_s: float, frame_gaps_s: list[float], duration_s: float, peer_durations: list[float]) -> list[str]`. Reasons, in this order, are `idle_start`, `dropped_frames`, and `outlier_length`. Idle start is `start_idle_s > 1.0`. A dropped frame is a positive gap greater than twice the median of the positive gaps. An outlier length needs at least four peer durations and a duration below half the median or above one and a half times the median. Peers include this episode's duration.

- [ ] **Step 1: Write the failing test**

```python
from lelab.improve.curate import flag_episode


def test_flags_are_idle_start_dropped_frames_and_outlier_length() -> None:
    peers = [10.0, 10.0, 10.0, 30.0]
    assert flag_episode(1.5, [0.1, 0.1, 0.5], 30.0, peers) == [
        "idle_start",
        "dropped_frames",
        "outlier_length",
    ]
    assert flag_episode(1.0, [0.1, 0.1], 10.0, [10.0, 10.0, 10.0, 10.0]) == []
    assert flag_episode(0.0, [], 99.0, [99.0]) == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_improve_curate.py -q`

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the minimal implementation**

```python
from statistics import median

IDLE_START_S = 1.0


def flag_episode(
    start_idle_s: float,
    frame_gaps_s: list[float],
    duration_s: float,
    peer_durations: list[float],
) -> list[str]:
    reasons: list[str] = []
    if start_idle_s > IDLE_START_S:
        reasons.append("idle_start")
    positive = [gap for gap in frame_gaps_s if gap > 0]
    if positive and any(gap > 2 * median(positive) for gap in positive):
        reasons.append("dropped_frames")
    if len(peer_durations) >= 4:
        middle = median(peer_durations)
        if duration_s < 0.5 * middle or duration_s > 1.5 * middle:
            reasons.append("outlier_length")
    return reasons
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_improve_curate.py -q`

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add lelab/improve/curate.py tests/test_improve_curate.py
git commit -m "$(cat <<'EOF'
feat: flag idle, dropped, and outlier episodes

EOF
)"
```

---

### Task 3: Add the curation notebook

**Files:**

- Create: `notebooks/curate_dataset.py`
- Create: `tests/test_curate_notebook.py`

**Interfaces:**

- Consumes: `flag_episode` from `lelab.improve.curate`.
- Produces: a marimo script. It imports `flag_episode`. It shows `run.use_artifact("so101-kitting:latest")` and `artifact.download()`. It states that notebook sandbox files are temporary. It does not call `wandb.init` at import time.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_notebook_downloads_the_dataset_artifact_and_flags_episodes() -> None:
    text = Path("notebooks/curate_dataset.py").read_text()
    assert "flag_episode" in text
    assert 'use_artifact("so101-kitting:latest")' in text
    assert "artifact.download()" in text
    assert "temporary" in text.lower()
    assert "wandb.init(" not in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_curate_notebook.py -q`

Expected: FAIL with `FileNotFoundError`

- [ ] **Step 3: Write the minimal implementation**

The file starts with `import marimo` and defines `app = marimo.App()`. One cell imports `flag_episode`. One cell contains the two artifact lines inside a function named `download_dataset` that is not called at import time. A markdown cell says the notebook sandbox files are temporary. End with:

```python
if __name__ == "__main__":
    app.run()
```

Do not add `marimo` to `pyproject.toml`. CoreWeave Notebooks provides it.

- [ ] **Step 4: Run the test to verify it passes**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_curate_notebook.py tests/test_improve_curate.py -q`

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add notebooks/curate_dataset.py tests/test_curate_notebook.py
git commit -m "$(cat <<'EOF'
feat: add the dataset curation notebook

EOF
)"
```

---

### Task 4: Show the automation on the Improve page

**Files:**

- Create: `frontend/src/pages/Improve.tsx`
- Create: `frontend/src/pages/Improve.test.tsx`
- Modify: `frontend/src/App.tsx`

**Interfaces:**

- Consumes: `GET /forge-status` with the same strict `forge_configured === true` check as `Orders.tsx`. `AUTOMATION_NAME`, `AUTOMATION_EVENT`, and the prompt text from Task 1. Duplicate those three strings in the page. Do not import Python.
- Produces: Forge off renders `CoreWeave ARIA reports arrive in a later milestone.` and no automation name. Forge on renders the heading Improve, the name `so101-kitting evaluation`, the event `A run reaches FINISHED`, the prompt sentence that contains `Do not change an alias`, and the empty state `Close an evaluation, then open the project Automations tab and choose Trigger ARIA.` There is no anchor whose href contains `wandb.ai` or `aria`.

- [ ] **Step 1: Write the failing test**

Follow `frontend/src/pages/Orders.test.tsx` for `stubFetch`, the local `userEvent` helper, `ApiProvider`, and `MemoryRouter`. Cover forge off, forge on with `forge_configured: "yes"` still off, and forge on with the four strings above. Assert `queryByRole("link")` inside the page section does not include an href containing `wandb.ai`.

- [ ] **Step 2: Run the test to verify it fails**

Run from `frontend/`: `npm test -- src/pages/Improve.test.tsx`

Expected: FAIL because the page module does not exist.

- [ ] **Step 3: Write the minimal implementation**

Replace the `/improve` placeholder in `App.tsx` with `<Improve />`. Use the existing `PlaceholderPage` for the forge-off branch. Do not add a font, a teal color, or a status that is only a color. Do not poll and do not request camera bytes.

- [ ] **Step 4: Run the test to verify it passes**

Run from `frontend/`: `npm test -- src/pages/Improve.test.tsx`

Expected: the file passes.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/Improve.tsx frontend/src/pages/Improve.test.tsx frontend/src/App.tsx
git commit -m "$(cat <<'EOF'
feat: show the ARIA automation on Improve

EOF
)"
```

---

### Task 5: Document the storyline and the report

**Files:**

- Modify: `README.md`
- Create: `tests/test_forge_readme.py`

**Interfaces:**

- Consumes: `AUTOMATION_NAME` and `aria_prompt()` so the README can be compared with the code. The test imports them. The README contains the same name and the same prompt text.
- Produces: a `## CoreWeave Forge` section before `## Contribute`. It lists the eight storyline steps from the design: setup, record, curate, train, evaluate, record more and promote, order, and repeat on the CoreWeave policy host. It names the notebook `notebooks/curate_dataset.py`, the automation event `A run reaches FINISHED`, and the action Trigger ARIA. It includes the prompt from `aria_prompt()` as the W&B Report brief ARIA should write. It does not contain `https://wandb.ai`.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from lelab.improve.aria import AUTOMATION_NAME, aria_prompt


def test_readme_matches_the_automation_and_lists_the_storyline() -> None:
    text = Path("README.md").read_text()
    assert AUTOMATION_NAME in text
    assert aria_prompt() in text
    assert "notebooks/curate_dataset.py" in text
    assert "A run reaches FINISHED" in text
    assert "Trigger ARIA" in text
    assert "https://wandb.ai" not in text
    for step in ("Setup", "Record", "Curate", "Train", "Evaluate", "Promote", "Order"):
        assert step in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_readme.py -q`

Expected: FAIL because the section is absent.

- [ ] **Step 3: Write the minimal implementation**

Insert the section before `## Contribute`. Keep the existing upstream sections. The eight steps are short sentences, not a transcript of a live run.

- [ ] **Step 4: Run the test to verify it passes**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_readme.py tests/test_improve_aria.py -q`

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_forge_readme.py
git commit -m "$(cat <<'EOF'
docs: describe the Forge storyline and ARIA report

EOF
)"
```
