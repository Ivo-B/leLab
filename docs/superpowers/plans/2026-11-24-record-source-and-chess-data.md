# Record Source, Chess Dataset Check and M3 Transfer Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Record page says which kind of data it makes, sim datasets get a chosen name, training warns when a dataset is not chess transfer data, and real recordings become chess transfer data (M3 Tasks 1–3) so sim and real can be co-trained.

**Architecture:** Task 1 adds a `RecordSourceBanner` component and an optional `name` to `POST /sim-generate`. Task 2 adds `lelab/dataset_check.py` (reads `meta/info.json`, the episode index and the transfer sidecar; no torch, no LeRobot dataset objects), a `GET /datasets/chess-check` route and a warning under the Train page's dataset picker. Tasks 3–5 execute M3 Tasks 1–3 from `docs/superpowers/plans/2026-11-23-garage-chess-m3-real-board.md` with the amendments listed in each task here.

**Tech Stack:** Python 3.12, FastAPI, pydantic, pyarrow (through `lelab.episode_media`), React + Vite + vitest.

Decision record: a four-voice council (2026-11-24) rejected a "Source: Teleoperation / Simulation" switch on the Record page until real recording makes chess transfer data. The two entry points stay (configure a real recording on Home, or Home → "Generate in simulation"); the banner makes the hidden rule visible.

## Global Constraints

- Work on branch `forge/chess`. Push only to `origin` (`https://github.com/Ivo-B/leLab.git`), and only when asked. Never `upstream`.
- Do not start M4 (no voice, player, Stockfish or game loop).
- Do not start the robot arm in a unit test. `lelab/chess/` never imports `lelab.sim`, `lerobot.robots`, `lelab.record`, `lelab.forge` or `openai`.
- The task string (`CHESS_TASK`), the markers and the sidecar format (`meta/garage_transfers.json`, entries `{"episode_index", "source", "target"}`) are M1's; do not change them.
- Dataset artifact names: `so101-chess-sim` for sim, `so101-chess-real` for everything else (`lelab/forge/tracking.py`). Do not rename.
- An existing local dataset is never overwritten by sim generation (the M1 dataset `local/so101-chess-sim` included).
- The lens and colour corrections (`lelab/lens.py`, `lelab/color_gain.py`, frontend `cameraLens.ts`, `cameraGain.ts`) are applied before anything reads pixels for chess: board calibration clicks, the reference frame, board-state reads and the overlay all work on corrected frames. Do not reorder `color_gain.install` / `lens.install` in `lelab/record.py`.
- `WANDB_API_KEY` and `CWSANDBOX_API_KEY` never appear in a response, a log line or a logger call. Do not touch `.env`.
- New Python modules start with the Apache-2.0 header (`tests/test_forge_config.py` lines 1–13), then a blank line.
- Python tests: `uv run pytest` outside the sandbox. Frontend: `npx vitest run` and `npx tsc --noEmit -p .` in `frontend/`. Lint: `uv run ruff check lelab tests`, `uv run ruff format --check lelab tests`.
- Stage files by name. Never `git add -A`, `git add .`, `git commit -a`. Do not skip hooks. `MUJOCO_LOG.TXT` and `lerobot.code-workspace` are never committed.
- After each task's commit run `graphify update .` (graphify-out is untracked; do not commit it).

---

### Task 1: Record mode banner and sim dataset name

**Files:**
- Create: `frontend/src/components/recording/RecordSourceBanner.tsx`, `frontend/src/components/recording/RecordSourceBanner.test.tsx`
- Modify: `frontend/src/pages/Recording.tsx` (the no-config branch near line 600 and the session header near line 714)
- Modify: `frontend/src/components/recording/SimGenerateControl.tsx`
- Modify: `frontend/src/pages/Recording.test.tsx`
- Modify: `lelab/server.py` (`SimGenerateBody`, `_sim_generate_state`, `_sim_repo_id`, `_run_sim_generate`, `start_sim_generate`, near lines 560–650)
- Modify: `tests/test_sim_jobs.py`

**Interfaces:**
- Produces: `POST /sim-generate` body `{episodes, seed, randomize, name?: str}`; `name` matches `^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$` (422 otherwise). Response `{"started": True, "repo_id": "local/<name or default>"}`. 409 with detail `"Dataset local/<x> already exists; pick another name"` when the directory exists in the LeRobot cache (`episode_media.lerobot_cache_root()`).
- Produces: `GET /sim-generate` gains `"repo_id": str | None` (idle default `None`).
- Produces: `RecordSourceBanner({ source }: { source: "sim" | "teleop" })`. Task 5 extends it with `chessTransfers?: boolean`.

- [ ] **Step 1: Backend failing tests** (append to `tests/test_sim_jobs.py`; update the `idle_state` fixture and the existing exact-dict assertions to include `"repo_id"`, and make the `idle_state` fixture also `monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))` so the existence check never sees the developer's real cache)

```python
def test_generate_uses_the_chosen_name(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    release, calls = threading.Event(), []
    monkeypatch.setattr(server, "_load_generate_dataset", lambda: blocking_generate(release, calls))
    response = client.post("/sim-generate", json={"episodes": 5, "seed": 0, "randomize": True, "name": "chess-sim-a"})
    assert response.status_code == 200
    assert response.json() == {"started": True, "repo_id": "local/chess-sim-a"}
    assert client.get("/sim-generate").json()["repo_id"] == "local/chess-sim-a"
    release.set()
    wait_until_idle(client, "/sim-generate")
    assert calls[0][0] == "local/chess-sim-a"


def test_generate_refuses_an_existing_dataset(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    (tmp_path / "local" / "so101-chess-sim").mkdir(parents=True)
    calls: list[tuple] = []
    monkeypatch.setattr(server, "_load_generate_dataset", lambda: blocking_generate(threading.Event(), calls))
    response = client.post("/sim-generate", json={"episodes": 5, "name": "so101-chess-sim"})
    assert response.status_code == 409
    assert response.json()["detail"] == "Dataset local/so101-chess-sim already exists; pick another name"
    assert calls == []
    assert client.get("/sim-generate").json()["running"] is False


@pytest.mark.parametrize("name", ["", "../escape", "a/b", "-lead", "x" * 97, "spa ce"])
def test_generate_rejects_bad_names(client: TestClient, name: str) -> None:
    assert client.post("/sim-generate", json={"episodes": 5, "name": name}).status_code == 422
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest -q tests/test_sim_jobs.py`
Expected: the three new tests fail (`repo_id` missing from the response, no 409, names accepted).

- [ ] **Step 3: Backend implementation** (`lelab/server.py`)

```python
SIM_DATASET_NAME = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"


class SimGenerateBody(BaseModel):
    episodes: int = Field(default=30, ge=1)
    seed: int = 0
    randomize: bool = True
    name: str | None = Field(default=None, pattern=SIM_DATASET_NAME)


_sim_generate_state: dict[str, Any] = {"running": False, "saved": 0, "episodes": 0, "error": None, "repo_id": None}


def _sim_repo_id(body: SimGenerateBody) -> str:
    if body.name:
        return f"local/{body.name}"
    suffix = "" if body.randomize else "-norand"
    return f"local/so101-chess-sim-{body.episodes}ep-s{body.seed}{suffix}"
```

`_run_sim_generate(body, repo_id)` passes `repo_id` to `generate_dataset` instead of calling `_sim_repo_id` itself. `start_sim_generate`:

```python
@app.post("/sim-generate")
def start_sim_generate(body: SimGenerateBody):
    """Generate a chess dataset in simulation on a background thread."""
    repo_id = _sim_repo_id(body)
    if (episode_media.lerobot_cache_root() / repo_id).exists():
        raise HTTPException(status_code=409, detail=f"Dataset {repo_id} already exists; pick another name")
    with _sim_lock:
        if _sim_job_running():
            raise HTTPException(status_code=409, detail="A simulation job is already running")
        _sim_generate_state.update(running=True, saved=0, episodes=body.episodes, error=None, repo_id=repo_id)
    threading.Thread(target=_run_sim_generate, args=(body, repo_id), daemon=True, name="sim-generate").start()
    return {"started": True, "repo_id": repo_id}
```

Import `episode_media` the way `server.py` already imports sibling modules (check the existing import block; add `from . import episode_media` only if it is not imported yet).

- [ ] **Step 4: Run the backend tests**

Run: `uv run pytest -q tests/test_sim_jobs.py`
Expected: all pass.

- [ ] **Step 5: Frontend failing tests**

`frontend/src/components/recording/RecordSourceBanner.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import RecordSourceBanner from "./RecordSourceBanner";

describe("RecordSourceBanner", () => {
  it("says simulation never moves the arm and names the sim artifact", () => {
    render(<RecordSourceBanner source="sim" />);
    const banner = screen.getByRole("note");
    expect(banner).toHaveTextContent("Simulation");
    expect(banner).toHaveTextContent("The arm does not move.");
    expect(banner).toHaveTextContent("so101-chess-sim");
  });

  it("says teleoperation moves the real arm and is not co-training data yet", () => {
    render(<RecordSourceBanner source="teleop" />);
    const banner = screen.getByRole("note");
    expect(banner).toHaveTextContent("Teleoperation");
    expect(banner).toHaveTextContent("The real arm moves.");
    expect(banner).toHaveTextContent("so101-chess-real");
    expect(banner).toHaveTextContent("no red/green markers or transfer sidecar");
  });
});
```

In `frontend/src/pages/Recording.test.tsx`, inside `describe("Recording: generate in simulation")`, add:

```tsx
  it("shows the simulation banner and posts the chosen dataset name", async () => {
    const fetchMock = stubFetch(() => respond({ started: true, repo_id: "local/chess-sim-a" }));
    renderRecording();
    expect(await screen.findByRole("note")).toHaveTextContent("Simulation");
    fireEvent.change(screen.getByLabelText("Dataset name (optional)"), { target: { value: "chess-sim-a" } });
    fireEvent.click(screen.getByRole("button", { name: "Generate in simulation" }));
    await waitFor(() => expect(lastPostBody(fetchMock)).toMatchObject({ name: "chess-sim-a" }));
  });

  it("disables the button for a dataset name with a slash", async () => {
    stubFetch(() => respond({ running: false, saved: 0, episodes: 0, error: null, repo_id: null }));
    renderRecording();
    fireEvent.change(await screen.findByLabelText("Dataset name (optional)"), { target: { value: "a/b" } });
    expect(screen.getByRole("button", { name: "Generate in simulation" })).toBeDisabled();
  });
```

Use the file's existing helpers (`stubFetch`, `respond`, the render helper at line ~40, and however the existing "posts the episodes, seed and randomization" test reads the POST body); rename `renderRecording` / `lastPostBody` to whatever those helpers are called there. Leave the name empty in the existing tests: an empty name is not sent.

- [ ] **Step 6: Run them and see them fail**

Run: `cd frontend && npx vitest run src/components/recording/RecordSourceBanner.test.tsx src/pages/Recording.test.tsx`
Expected: cannot resolve `./RecordSourceBanner`; no "Dataset name (optional)" field.

- [ ] **Step 7: Frontend implementation**

`frontend/src/components/recording/RecordSourceBanner.tsx`:

```tsx
interface RecordSourceBannerProps {
  source: "sim" | "teleop";
}

const COPY = {
  sim: {
    title: "Simulation",
    body: "The arm does not move. Episodes are generated in MuJoCo with the red/green markers and the transfer sidecar, and logged to W&B as so101-chess-sim when Forge is connected.",
  },
  teleop: {
    title: "Teleoperation",
    body: "The real arm moves. With Forge connected the dataset is logged to W&B as so101-chess-real. It has no red/green markers or transfer sidecar, so it is not chess co-training data yet.",
  },
} as const;

export default function RecordSourceBanner({ source }: RecordSourceBannerProps) {
  const { title, body } = COPY[source];
  return (
    <div role="note" className="rounded-lg border border-border bg-muted/40 px-4 py-2 text-sm">
      <span className="font-bold">{title}.</span> {body}
    </div>
  );
}
```

`Recording.tsx`: in the no-config branch render `<RecordSourceBanner source="sim" />` directly under the `<h1>`; in the session view render `<RecordSourceBanner source="teleop" />` between the Back button block and the `bg-gray-900` panel (inside the `mb-3 flex-shrink-0` wrapper, below the button, with `mt-3`).

`SimGenerateControl.tsx`: add `const [name, setName] = useState("")`, a third input in the grid:

```tsx
        <div className="flex flex-col gap-1.5 sm:col-span-2">
          <label htmlFor="sim-name" className="text-sm font-bold">
            Dataset name (optional)
          </label>
          <Input
            id="sim-name"
            value={name}
            placeholder="so101-chess-sim-30ep-s0"
            disabled={running}
            onChange={(event) => setName(event.target.value)}
          />
        </div>
```

with `const NAME_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/;`, `const nameValid = name.trim() === "" || NAME_PATTERN.test(name.trim());`, `canStart` also requires `nameValid`, and the request adds `...(name.trim() ? { name: name.trim() } : {})`. Add `repo_id: string | null` to `GenerateStatus`; when the job finished without error, the status text becomes `Saved ${saved} simulated episodes to ${repo_id}.` (fall back to the current text when `repo_id` is null).

- [ ] **Step 8: Run the frontend checks**

Run: `cd frontend && npx vitest run && npx tsc --noEmit -p .`
Expected: all pass, no type errors.

- [ ] **Step 9: Commit**

```bash
git add lelab/server.py tests/test_sim_jobs.py frontend/src/components/recording/RecordSourceBanner.tsx frontend/src/components/recording/RecordSourceBanner.test.tsx frontend/src/components/recording/SimGenerateControl.tsx frontend/src/pages/Recording.tsx frontend/src/pages/Recording.test.tsx
git commit -m "feat: say which data the Record page makes and name sim datasets"
```

---

### Task 2: Chess dataset check and the Train page warning

**Files:**
- Create: `lelab/dataset_check.py`, `tests/test_dataset_check.py`
- Modify: `lelab/server.py` (one route next to the other `/datasets/*` routes)
- Create: `frontend/src/components/training/config/ChessDatasetCheck.tsx`, `frontend/src/components/training/config/ChessDatasetCheck.test.tsx`
- Modify: `frontend/src/components/training/config/EssentialsCard.tsx` (render the check under the dataset picker)

**Interfaces:**
- Consumes: `episode_media.resolve_dataset_dir(repo_id) -> Path` (raises `DatasetNotFoundError`), `episode_media.read_info(dir) -> dict`, `episode_media.read_episode_index(dir) -> list[EpisodeIndexRow]` (`row.tasks`), `lelab.chess.transfers.CHESS_TASK`, `read_transfer_sidecar(dir) -> list[dict]`.
- Produces: `lelab/dataset_check.py`: `CHESS_JOINTS: tuple[str, ...]`, `POLICY_CAMERAS: tuple[str, ...] = ("top", "wrist")`, `chess_problems(dataset_dir: Path) -> list[str]`, `mismatches(a: Path, b: Path) -> list[str]`, `handle_chess_check(repo_id: str, against: str | None) -> dict`.
- Produces: `GET /datasets/chess-check?repo_id=<id>[&against=<id>]` → `{"checked": bool, "repo_id": str, "problems": list[str], "mismatches": list[str], "message": str | None}`. `checked` is false (with `message`) when a dataset is not in the local cache; it is never a 4xx for a Hub-only dataset. M3 Task 5 (co-training) calls `mismatches` before mixing two datasets.

- [ ] **Step 1: Failing tests** (`tests/test_dataset_check.py`, Apache header first)

```python
"""chess_problems and mismatches over hand-written LeRobot v3.0 metadata. No torch, no videos."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from lelab import dataset_check, server
from lelab.chess.transfers import CHESS_TASK, write_transfer_sidecar

JOINTS = list(dataset_check.CHESS_JOINTS)


def make_dataset(root: Path, *, episodes: int = 2, task: str = CHESS_TASK, sidecar: bool = True,
                 cameras: tuple[str, ...] = ("top", "wrist", "side"), joints: list[str] = JOINTS,
                 size: tuple[int, int] = (480, 640), fps: int = 30) -> Path:
    meta = root / "meta"
    (meta / "episodes" / "chunk-000").mkdir(parents=True)
    vector = {"dtype": "float32", "shape": [len(joints)], "names": joints}
    features = {"action": vector, "observation.state": vector}
    for cam in cameras:
        features[f"observation.images.{cam}"] = {"dtype": "video", "shape": [*size, 3], "names": ["height", "width", "channels"]}
    (meta / "info.json").write_text(json.dumps({"fps": fps, "total_episodes": episodes, "features": features}))
    pd.DataFrame({
        "episode_index": list(range(episodes)),
        "length": [10] * episodes,
        "tasks": [[task]] * episodes,
    }).to_parquet(meta / "episodes" / "chunk-000" / "file-000.parquet")
    if sidecar:
        for i in range(episodes):
            write_transfer_sidecar(root, i, ("e2", "e4"))
    return root


def test_a_chess_transfer_dataset_has_no_problems(tmp_path: Path) -> None:
    assert dataset_check.chess_problems(make_dataset(tmp_path / "d")) == []


def test_a_plain_teleop_dataset_lists_every_gap(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "d", task="pick the bolt", sidecar=False, cameras=("wrist",))
    problems = dataset_check.chess_problems(root)
    assert any("observation.images.top" in p for p in problems)
    assert any("pick the bolt" in p for p in problems)
    assert any("2 of 2 episodes have no transfer" in p for p in problems)


def test_partial_sidecar_and_wrong_joints(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "d", episodes=3, sidecar=False, joints=["a", "b"])
    write_transfer_sidecar(root, 0, ("e2", "e4"))
    problems = dataset_check.chess_problems(root)
    assert any("2 of 3 episodes have no transfer" in p for p in problems)
    assert any(p.startswith("action joints") for p in problems)


def test_unreadable_sidecar_is_a_problem_not_a_crash(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "d", sidecar=False)
    (root / "meta" / "garage_transfers.json").write_text("{not json")
    assert any("garage_transfers.json" in p for p in dataset_check.chess_problems(root))


def test_mismatches_compare_fps_cameras_and_shapes(tmp_path: Path) -> None:
    sim = make_dataset(tmp_path / "sim")
    real = make_dataset(tmp_path / "real", cameras=("top", "wrist"), size=(1080, 1920), fps=15)
    found = dataset_check.mismatches(sim, real)
    assert any(p.startswith("fps") for p in found)
    assert any("observation.images.side" in p for p in found)
    assert any("observation.images.top" in p and "shape" in p for p in found)
    assert dataset_check.mismatches(sim, make_dataset(tmp_path / "same")) == []


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    return tmp_path


def test_route_checks_a_local_dataset_against_another(cache: Path) -> None:
    make_dataset(cache / "local" / "sim")
    make_dataset(cache / "local" / "real", sidecar=False)
    body = TestClient(server.app).get("/datasets/chess-check", params={"repo_id": "local/real", "against": "local/sim"}).json()
    assert body["checked"] is True
    assert body["repo_id"] == "local/real"
    assert any("no transfer" in p for p in body["problems"])
    assert body["mismatches"] == []


def test_route_does_not_fail_for_a_hub_only_dataset(cache: Path) -> None:
    response = TestClient(server.app).get("/datasets/chess-check", params={"repo_id": "someone/remote"})
    assert response.status_code == 200
    assert response.json()["checked"] is False
    assert response.json()["message"]
```

If `read_episode_index` needs columns the fixture lacks (check `EpisodeIndexRow` and `read_episode_index` in `lelab/episode_media.py`, and the fixtures in `tests/test_episode_media.py`), add them to `make_dataset` the way `tests/test_episode_media.py` builds its episodes parquet; do not change `episode_media`.

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest -q tests/test_dataset_check.py`
Expected: `ImportError: cannot import name 'dataset_check'`.

- [ ] **Step 3: Implementation** (`lelab/dataset_check.py`, Apache header first)

```python
"""Is a local dataset chess transfer data, and can two datasets be trained together?

Reads only metadata (`meta/info.json`, the episode index and `meta/garage_transfers.json`), so it is cheap enough
to run when a dataset is picked. The red/green markers are not detected in pixels: an episode has them exactly when
its transfer is in the sidecar, because the overlay and the sidecar come from the same recording step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import episode_media
from .chess.transfers import CHESS_TASK, SIDECAR_NAME, read_transfer_sidecar

CHESS_JOINTS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
POLICY_CAMERAS = ("top", "wrist")


def chess_problems(dataset_dir: Path) -> list[str]:
    info = episode_media.read_info(dataset_dir)
    features = info.get("features") or {}
    problems: list[str] = []
    for key in ("action", "observation.state"):
        names = list((features.get(key) or {}).get("names") or [])
        if names != list(CHESS_JOINTS):
            problems.append(f"{key} joints are {names}, expected {list(CHESS_JOINTS)}")
    for camera in POLICY_CAMERAS:
        key = f"{episode_media.VIDEO_KEY_PREFIX}{camera}"
        if (features.get(key) or {}).get("dtype") != "video":
            problems.append(f"no {key} video")
    other_tasks = sorted({t for row in episode_media.read_episode_index(dataset_dir) for t in row.tasks} - {CHESS_TASK})
    if other_tasks:
        problems.append(f"episodes with a task other than the chess task: {other_tasks}")
    total = int(info.get("total_episodes") or 0)
    try:
        covered = {entry.get("episode_index") for entry in read_transfer_sidecar(dataset_dir)}
    except (OSError, json.JSONDecodeError, AttributeError):
        problems.append(f"meta/{SIDECAR_NAME} is unreadable")
        return problems
    missing = sum(1 for index in range(total) if index not in covered)
    if missing:
        problems.append(
            f"{missing} of {total} episodes have no transfer in meta/{SIDECAR_NAME} (recorded without the red/green markers)"
        )
    return problems


def _feature_signature(feature: dict[str, Any]) -> tuple[Any, Any, Any]:
    return feature.get("dtype"), list(feature.get("shape") or []), feature.get("names")


def mismatches(a: Path, b: Path) -> list[str]:
    info_a, info_b = episode_media.read_info(a), episode_media.read_info(b)
    found: list[str] = []
    if info_a.get("fps") != info_b.get("fps"):
        found.append(f"fps {info_a.get('fps')} vs {info_b.get('fps')}")
    features_a, features_b = info_a.get("features") or {}, info_b.get("features") or {}
    for key in sorted(set(features_a) | set(features_b)):
        if key not in features_a or key not in features_b:
            found.append(f"{key} only in {'the first' if key in features_a else 'the second'} dataset")
            continue
        sig_a, sig_b = _feature_signature(features_a[key]), _feature_signature(features_b[key])
        for label, value_a, value_b in zip(("dtype", "shape", "names"), sig_a, sig_b, strict=True):
            if value_a != value_b:
                found.append(f"{key} {label} {value_a} vs {value_b}")
    return found


def handle_chess_check(repo_id: str, against: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"checked": False, "repo_id": repo_id, "problems": [], "mismatches": [], "message": None}
    try:
        dataset_dir = episode_media.resolve_dataset_dir(repo_id)
        other_dir = episode_media.resolve_dataset_dir(against) if against else None
    except episode_media.DatasetNotFoundError as exc:
        result["message"] = f"{exc}. Only local datasets can be checked."
        return result
    result["checked"] = True
    result["problems"] = chess_problems(dataset_dir)
    if other_dir is not None:
        result["mismatches"] = mismatches(dataset_dir, other_dir)
    return result
```

Route in `lelab/server.py`, next to the other `/datasets/*` routes:

```python
@app.get("/datasets/chess-check")
def dataset_chess_check(repo_id: str, against: str | None = None):
    """Whether a local dataset is chess transfer data, and how it differs from `against`."""
    return dataset_check.handle_chess_check(repo_id, against)
```

with `from . import dataset_check` in the import block. A route declared after a catch-all `/datasets/{...}` path would be shadowed: place it before any parametrised `/datasets/` route.

- [ ] **Step 4: Run the backend tests**

Run: `uv run pytest -q tests/test_dataset_check.py`
Expected: all pass.

- [ ] **Step 5: Frontend failing test** (`ChessDatasetCheck.test.tsx`)

```tsx
import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import ChessDatasetCheck from "./ChessDatasetCheck";

const fetcher = vi.fn();
vi.mock("@/contexts/ApiContext", () => ({
  useApi: () => ({ baseUrl: "http://api", fetchWithHeaders: fetcher }),
}));

function respond(body: unknown) {
  fetcher.mockResolvedValueOnce({ ok: true, status: 200, json: () => Promise.resolve(body) } as Response);
}

describe("ChessDatasetCheck", () => {
  it("says a clean dataset is chess transfer data", async () => {
    respond({ checked: true, repo_id: "local/sim", problems: [], mismatches: [], message: null });
    render(<ChessDatasetCheck repoId="local/sim" />);
    expect(await screen.findByText("Chess transfer data: markers, sidecar and task string present.")).toBeInTheDocument();
    expect(fetcher).toHaveBeenCalledWith("http://api/datasets/chess-check?repo_id=local%2Fsim");
  });

  it("lists the problems of a dataset that is not chess transfer data", async () => {
    respond({ checked: true, repo_id: "local/real", problems: ["2 of 2 episodes have no transfer"], mismatches: [], message: null });
    render(<ChessDatasetCheck repoId="local/real" />);
    expect(await screen.findByRole("alert")).toHaveTextContent("2 of 2 episodes have no transfer");
  });

  it("shows nothing for a dataset that is not local", async () => {
    respond({ checked: false, repo_id: "a/b", problems: [], mismatches: [], message: "not local" });
    const { container } = render(<ChessDatasetCheck repoId="a/b" />);
    await waitFor(() => expect(fetcher).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
  });
});
```

Reset `fetcher` in a `beforeEach(() => fetcher.mockReset())`.

- [ ] **Step 6: Run it and see it fail**

Run: `cd frontend && npx vitest run src/components/training/config/ChessDatasetCheck.test.tsx`
Expected: cannot resolve `./ChessDatasetCheck`.

- [ ] **Step 7: Implementation**

`ChessDatasetCheck.tsx`:

```tsx
import { useEffect, useState } from "react";
import { useApi } from "@/contexts/ApiContext";

interface ChessCheck {
  checked: boolean;
  problems: string[];
}

// Warns, never blocks: a dataset that fails the check can still be trained, it just is not chess transfer data.
export default function ChessDatasetCheck({ repoId }: { repoId: string }) {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [check, setCheck] = useState<ChessCheck | null>(null);

  useEffect(() => {
    let cancelled = false;
    setCheck(null);
    if (!repoId) return;
    fetchWithHeaders(`${baseUrl}/datasets/chess-check?repo_id=${encodeURIComponent(repoId)}`)
      .then((response) => (response.ok ? response.json() : null))
      .then((body) => {
        if (!cancelled) setCheck(body as ChessCheck | null);
      })
      .catch(() => {
        if (!cancelled) setCheck(null);
      });
    return () => {
      cancelled = true;
    };
  }, [repoId, baseUrl, fetchWithHeaders]);

  if (!check?.checked) return null;
  if (check.problems.length === 0) {
    return <p className="mt-2 text-sm text-muted-foreground">Chess transfer data: markers, sidecar and task string present.</p>;
  }
  return (
    <div role="alert" className="mt-2 text-sm text-amber-400">
      <p className="font-bold">Not chess transfer data:</p>
      <ul className="list-disc pl-5">
        {check.problems.map((problem) => (
          <li key={problem}>{problem}</li>
        ))}
      </ul>
    </div>
  );
}
```

`EssentialsCard.tsx`: render `<ChessDatasetCheck repoId={config.dataset_repo_id} />` right after the dataset picker. In `EssentialsCard.test.tsx`, mock `./ChessDatasetCheck` to `() => null` if the existing tests do not provide an `ApiContext`.

- [ ] **Step 8: Run all checks**

Run: `uv run pytest -q && uv run ruff check lelab tests && uv run ruff format --check lelab tests && cd frontend && npx vitest run && npx tsc --noEmit -p .`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add lelab/dataset_check.py tests/test_dataset_check.py lelab/server.py frontend/src/components/training/config/ChessDatasetCheck.tsx frontend/src/components/training/config/ChessDatasetCheck.test.tsx frontend/src/components/training/config/EssentialsCard.tsx frontend/src/components/training/config/EssentialsCard.test.tsx
git commit -m "feat: check whether a dataset is chess transfer data before training"
```

---

### Task 3: M3 Task 1 — calibration and board-state API

Execute **Task 1** of `docs/superpowers/plans/2026-11-23-garage-chess-m3-real-board.md` (`lelab/board_state.py`, the three `/chess/*` routes, `tests/test_chess_api.py`) exactly as written there, with these amendments:

- `lelab/server.py` changed after that plan was written (camera controls, lens and gain routes). Put the routes next to the current camera-reference routes; match the file as it is now, not the line numbers in the M3 plan.
- The `top` reference the board state compares against is now saved lens- and colour-corrected (`frontend/src/hooks/useCameraReference.ts`). `read_board_state` does not correct frames itself: the caller sends a corrected PNG (Task 5 amendment). Add one sentence saying so to the `read_board_state` docstring.
- When done, update the M3 plan's **Status** line: "Task 1 implemented on 2026-11-24 (commit <sha>)."

Commit as the M3 plan says, plus the M3 plan file.

---

### Task 4: M3 Task 2 — transfer sampling and the recording hooks

Execute **Task 2** of the M3 plan (`TransferSampler` and preconditions in `lelab/chess/transfers.py`, the `lelab/record.py` hooks: fixed task, overlay step, `/camera-feed/top` overlay, `transfer` in the status, sidecar after save) exactly as written there, with these amendments:

- `lelab/record.py` now calls `color_gain.install(...)` and `lens.install(...)` just before `record_with_web_events(...)` in `handle_start_recording`. Those patch `OpenCVCamera._postprocess_image`, which runs before `robot_observation_processor`, so the overlay step draws the markers on the corrected frame, the same geometry the board calibration was clicked on. Keep that order. Add a test that `handle_start_recording` with `chess_transfers=True` still installs the gain and lens before the record loop starts (patch `color_gain.install`, `lens.install` and `record_with_web_events` and assert the call order).
- Check that `/camera-feed/top` frames come from the same corrected camera read; if the feed reads frames through another path, apply `lens.apply_lens` and `color_gain.apply_gains` there before the overlay (lens first, then colour, as `captureVideoFrame` does) and test it.
- `meta/garage_transfers.json` written by this task must make `dataset_check.chess_problems` (Task 2 of this plan) return `[]` for a dataset recorded in transfer mode. Add a test that writes sidecar entries through the record hook's code path for two saved episodes and asserts `chess_problems` lists no sidecar problem.
- Update the M3 plan's **Status** line with Task 2.

Commit as the M3 plan says (`git add -p` for `lelab/record.py` / `lelab/server.py` only if they carry unrelated hunks), plus the M3 plan file.

---

### Task 5: M3 Task 3 — board calibration on Setup, transfer mode on Record

Execute **Task 3** of the M3 plan (`boardCalibrationApi.ts`, `BoardCalibrationCard`, Setup, `useRecording`, `RecordingModal`, `Recording.tsx`) exactly as written there, with these amendments:

- **Check position must send a corrected frame.** The M3 plan's `checkPosition` calls `captureVideoFrame(video)` on a raw `getUserMedia` stream, but the `top` reference is saved with the lens and colour corrections, so the board-state diff would compare a bent frame with a straight one. Move the correction lookup out of `useCameraReference.save` into `frontend/src/lib/cameraReferenceApi.ts`:

  ```ts
  /** A frame from `video` with the saved lens and colour corrections of its camera, as the recorder stores it. */
  export async function captureCorrectedFrame(
    video: HTMLVideoElement,
    baseUrl: string,
    fetcher: Fetcher,
    { gain = true, lens = true }: { gain?: boolean; lens?: boolean } = {},
  ): Promise<Blob | null> {
    const label = (video.srcObject as MediaStream | null)?.getVideoTracks()[0]?.label ?? "";
    const gains = gain ? gainForCamera(await getCameraGains(baseUrl, fetcher), label) : null;
    const lensModel = lens ? lensForCamera(await getCameraLenses(baseUrl, fetcher), label) : null;
    return captureVideoFrame(video, { gains, lens: lensModel });
  }
  ```

  `useCameraReference.save` calls it (behaviour unchanged; its tests must still pass). `BoardCalibrationCard.checkPosition` calls `captureCorrectedFrame(video, baseUrl, fetchWithHeaders, corrections)` where `corrections` are the `top` camera's saved Colour/Lens correction switches, read the same way the Calibration page's camera card reads them (`useSavedCameraSetting` / `useCameraGain` / `useCameraLens`; check those hooks for the exact API). Test: mock `captureCorrectedFrame` and assert `checkPosition` calls it and posts its blob to `/chess/board-state`.
- **The click canvas is the corrected reference.** The 16 calibration clicks go on the saved reference image (already corrected), never on a live raw stream. Keep the M3 plan's design; add a one-line note under the card's image ("Click on the straightened reference; recapture it after changing the lens") so a lens change is followed by a new reference and new clicks.
- **The Record banner learns transfer mode.** Extend `RecordSourceBanner` (Task 1) with `chessTransfers?: boolean`. With `source="teleop"` and `chessTransfers`, the copy is: title "Chess transfer mode", body "The real arm moves. The top camera shows the red/green markers, each episode's transfer is saved in the sidecar, and the dataset is logged to W&B as so101-chess-real when Forge is connected. This is chess co-training data." `Recording.tsx` passes `chessTransfers={recordingConfig.chess_transfers === true}` (the field the M3 plan adds). Add the test case to `RecordSourceBanner.test.tsx`.
- Update the M3 plan's **Status** line with Task 3, and add to its "Manual hardware checklist" step 5: "Then open Train, pick the new dataset: the chess check shows no problems."

Commit as the M3 plan says, plus `cameraReferenceApi.ts`, `useCameraReference.ts`, the banner files and the M3 plan file.

---

## Operator steps (not automated)

1. Before Task 3's hardware use: the M3 start checks in the M3 plan (camera poses into `sim_cell.json`, cell built, joint zero matches the sim render).
2. After Task 5: calibrate the `top` lens and colour first, capture the empty-board reference, then the 16 board clicks (order matters: a lens change invalidates the reference and the clicks).
3. Run the M3 manual hardware checklist, including the new Train-page chess check.
4. Until Task 5 is done, don't record real episodes intended for co-training: they would have no markers or sidecar.

## After this plan

M3 Tasks 4–6 (real evaluation with the overlay in the rollout, correction recording and co-training, the gate) get full TDD detail next, from the outline in the M3 plan. Co-training (M3 Task 5) calls `dataset_check.mismatches` on the sim and real datasets before it mixes them.
