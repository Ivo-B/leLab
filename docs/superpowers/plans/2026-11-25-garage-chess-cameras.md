# Garage Chess — three cameras, one contract, sim-real parity (up to M3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Status:** drafted 2026-11-25. Runs after M3 Tasks 1–3 (done) and before the M3 hardware checklist and Task 4. Nothing here starts M4.

**Goal:** `top`, `wrist` and `side` are all attached, calibrated, recorded to one fixed contract, and measured into `sim_cell.json`, so a sim-trained policy sees the same pictures on the real arm and every camera mismatch is caught before it costs real episodes. The tools to compare camera sets exist, but the experiment itself is not run here.

**Architecture:** One camera contract (names, size, fps) lives in `lelab/chess/transfers.py` and is read by record-start checks, the dataset check and the sim generator. A pure-numpy pose fit in `lelab/chess/camera_pose.py` turns four clicked board corners into a camera pose for `sim_cell.json`; `lelab/sim/camera_fit.py` applies it to the cell; Setup gets one card per camera. A parity sheet renders sim and real side by side. Training and evaluation learn which cameras a policy reads and record it, so "does `side` help" becomes a later A/B on identical data.

**Tech Stack:** Python 3.12, FastAPI, OpenCV, numpy, MuJoCo (existing `lelab[sim]`), React + Vite + vitest.

Spec: `docs/superpowers/specs/2026-11-23-garage-chess-voice-design.md` (sections 2, 5.1, 5.5, 14, 15). Related: `docs/superpowers/plans/2026-11-25-camera-identity-and-exposure.md` (Task 0 below), `docs/superpowers/plans/2026-11-23-garage-chess-m3-real-board.md`.

## North star

A human plays a full chess game against the SO-101. The gate on the way there is M3: **real ≥ 80 % on the transfer plan with fewer than 100 real episodes**. Every real episode we need because sim and real cameras disagree is a failure of this plan. So each camera decision is judged by one question: *does it make the sim-trained policy transfer with fewer real episodes?* Anything that does not answer that is out.

## Debate and decisions

| # | Question | Options | Decision | Why |
|---|---|---|---|---|
| D1 | Which cameras go into real datasets? | (a) `top`+`wrist`, `side` kept separate; (b) all three always, pick at train time | **(b)** | A dataset can drop a camera later but never gain one. The side-camera A/B then trains on identical episodes. Cost: one more 640×480 stream. Sim datasets already hold all three. |
| D2 | When is the camera setup checked? | (a) after recording, in the dataset check; (b) at record start | **(b), plus the dataset check as a second net** | A wrong size or fps found after 30 episodes costs 30 episodes. Refuse before the first one. |
| D3 | What is the contract? | 1080p (what the guide uses for the lens tool) or the sim's 640×480 @ 30 | **640×480 @ 30, 4:3** | Mixed shapes cannot be co-trained. The sim is cheaper to regenerate than real data, so if a real camera cannot deliver 4:3 with the same field of view, change the sim once (Task 1 fallback), never the other way round per dataset. |
| D4 | Which rule decides compatibility of two datasets? | strict on every feature; or only the cameras the policy reads | **Only the cameras the policy reads, plus state, action and fps.** Extra cameras are ignored; a camera the policy needs and a dataset lacks is a mismatch | A legacy `top`+`wrist` real dataset still trains the two-camera policy; a three-camera policy refuses it. |
| D5 | How deep is the `side` calibration? | (i) reference + lens + colour only; (ii) + pose fit from the 4 board corners; (iii) + the 16-point polygons | **(ii)** | The sim-to-real gap for `side` is its pose and field of view (spec 15, M3 check). The only consumer of polygons is `top` (overlay and board reading); the VLM judge that used `side` was removed. (iii) is YAGNI. |
| D6 | Fit pose from clicks, or measure with a tape? | tape (±1 cm); clicks + `solvePnP`; ChArUco board | **Clicks + `solvePnP`, tape as the cross-check** | Prototype (below) shows clicks resolve `side` to about 10 mm at 1 px noise, as good as a tape, in one minute, with no new hardware. ChArUco is better but needs printing and a dependency; revisit only if the parity sheet shows more than 3 px error. |
| D7 | Does `side` get the red/green markers? | yes / no | **No** | Spec 5.1: the overlay edits `top` only. A policy that reads `side` still gets the goal from `top`. |
| D8 | Run the side experiment in M3? | now / after the gate | **After the gate**, with the knobs built now | M3's gate is about sim→real. The A/B needs the gate's training and evaluation plumbing, so build only the switches and the logging. |

**Prototype result behind D6** (2026-11-25, `SimCell` default poses, four board corners, 640×480, 1 px click noise, 30 draws):

| Camera | Field of view | Position error |
|---|---|---|
| `side` (45° oblique) | recovered to ±1° (curvature 2.4 px over ±5°) | median 10 mm |
| `top` (looks straight down) | **not recoverable** (curvature 0.02 px); a 3° error in the assumed field of view moves the camera 38 mm | median 11 mm with the right field of view, worst 35 mm |

So: fit pose and field of view on `side`; for `top`, fix the field of view (the same lens model as `side` after the same lens correction, or measured) and fit the pose only. The residual (about 1 cm, about 1°) is the width of the domain randomization (spec 5.3), which is what it is for.

## Findings (what is wrong with cameras today)

1. **Two identical cameras collide by name.** `top` and `side` are the same model. Gains, lens and USB controls are keyed by name or `vid:pid`. The fix exists in the working tree, uncommitted (`2026-11-25-camera-identity-and-exposure.md`).
2. **No camera contract.** Sim datasets are `top, wrist, side` at 640×480 @ 30. Real recording takes whatever width, height and fps the camera form holds; the guide even tells the operator to calibrate the lens at `1920x1080`. `chess_mode_problem` checks only `top` and its rotation. A real dataset at another size or with a missing camera is found only by `dataset_check.mismatches` after recording.
3. **`dataset_check.mismatches` compares every feature.** A legacy two-camera real dataset is flagged against the three-camera sim data, and nothing says which cameras a policy actually reads.
4. **No camera pose for `side`.** `sim_cell.json` holds the sim's guess for `top_camera` and `side_camera`; the real cell never writes back (M3 start check, spec 15). Pose and field of view cannot be measured with the tools we have.
5. **No way to see sim next to real.** Wrist orientation and field of view, and the `side` pose, can only be judged by recording and watching.
6. **Training cannot choose cameras.** `lerobot-train` infers `input_features` from every camera in the dataset (`lerobot/policies/factory.py`, `if not cfg.input_features`). `TrainingRequest` has no camera field, so "with and without `side`" cannot be run from LeLab.
7. **Evaluation does not record the camera set.** `EvalReport` and the real `TrialRow` carry no `cameras`, so a Weave table cannot split by camera set.

## Global constraints

- Work on `forge/chess`. Push only to `origin`. Stage files by name; the working tree holds unrelated uncommitted edits (camera identity work, `lerobot.code-workspace`, `MUJOCO_LOG.TXT`). Never `git add -A`.
- `lelab/chess/` imports no `lelab.sim`, `lerobot.robots`, `lelab.record`, `lelab.forge` or `openai`. `lelab/camera_config.py` is allowed.
- Task string, markers and sidecar stay M1's. Sim datasets are not changed except through the shared constants in Task 2.
- Camera names are exactly `top`, `wrist`, `side`. Their order in every config and dataset is `top`, `wrist`, `side`.
- Run Python tests with `uv run pytest`, frontend with `npm test -- --run` in `frontend/`, lint with `uv run ruff check lelab tests` and `uv run ruff format --check lelab tests`. Rendering tests run outside the Cursor sandbox.
- New Python modules start with the Apache-2.0 header in `tests/test_forge_config.py` lines 1–13.
- Out of scope: M4 and later; running the side-camera experiment; a `side` overlay or polygons; wrist pose fitting; ChArUco.

## File structure

- Create `lelab/chess/camera_pose.py`: `project`, `fit_camera`, `CameraFit`, `CameraPoseError`.
- Create `lelab/sim/camera_fit.py`: `world_corners`, `apply_fit`, `handle_camera_pose`.
- Create `lelab/sim/parity.py`: `blend`, `sheet`, `run_parity`; modify `lelab/sim/__main__.py` (`parity` command).
- Modify `lelab/chess/transfers.py`: contract constants and `camera_contract_problems`; `chess_mode_calibration` calls it.
- Modify `lelab/sim/generate.py`, `lelab/sim/env.py`: take camera names, size and fps from the contract.
- Modify `lelab/dataset_check.py`, `lelab/server.py`: `cameras` rule, `GET /datasets/chess-check?cameras=`, `POST /chess/camera-pose`.
- Modify `lelab/train.py`: `policy_cameras`. Modify `lelab/sim/evaluate.py`, `lelab/forge/weave_eval.py`: log `cameras`.
- Create `frontend/src/lib/cameraPoseApi.ts`, `frontend/src/components/setup/CameraPoseCard.tsx` (+ tests); modify `frontend/src/pages/Setup.tsx`.
- Modify `CHESS_DEMO_GUIDE.md`, `README.md` (after Task 0 lands them).
- Tests: `tests/test_chess_camera_pose.py`, `tests/test_sim_camera_fit.py`, `tests/test_sim_parity.py`, additions to `tests/test_chess_transfers.py`, `tests/test_record.py`, `tests/test_dataset_check.py`, `tests/test_train.py`, `tests/test_sim_evaluate.py`, `tests/test_sim_world.py`, `tests/test_sim_generate.py`, `tests/chess_render.py`.

---

### Task 0: Land the camera-identity and exposure work

**Why first:** `top` and `side` are the same model. Without port-based identity the second calibration overwrites the first, and the USB sweep can drive the wrong camera.

- [ ] **Step 1:** Run the plan `docs/superpowers/plans/2026-11-25-camera-identity-and-exposure.md`, Tasks 0–6. Its backend (identity keys, `location`, exposure/brightness in `Calibrator`) is already in the working tree; the browser part (its Task 5) and the demo-Mac verification (its Task 6, Step 2) are open.
- [ ] **Step 2:** Acceptance on the demo Mac with all three cameras attached: `uv run lelab-camera list` prints three different ids; calibrate `top` and `side` in `uv run lelab-camera-calibrate` (they appear with their ports); `~/.cache/huggingface/lerobot/` holds two separate gain entries and two separate lens entries; recording frames from each shows its own correction.
- [ ] **Step 3:** `uv run pytest -q && uv run ruff check lelab tests`. Commit the files that plan lists, by name, with its message.

### Task 1: Spike on the demo Mac — what can the cameras deliver at the contract

No commit except the guide note. Decides the numbers Task 2 writes down.

- [ ] **Step 1:** Run this once per camera index (`uv run lelab-camera list` gives the indices) with the other two unplugged from the USB hub if they share one:

```python
# uv run python - <<'EOF'
import cv2, time, sys
index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
for width, height in ((640, 480), (1920, 1080)):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height); cap.set(cv2.CAP_PROP_FPS, 30)
    for _ in range(10): cap.read()
    start, frames = time.time(), 0
    while time.time() - start < 3:
        ok, frame = cap.read(); frames += ok
    print(index, (width, height), "delivers", frame.shape[1::-1], f"{frames / 3:.1f} fps")
    cv2.imwrite(f"/tmp/cam{index}_{width}.png", frame); cap.release()
# EOF
```

- [ ] **Step 2:** Open `/tmp/cam<i>_640.png` and `/tmp/cam<i>_1920.png` side by side. The same scene edge-to-edge in both means 640×480 is scaled (same field of view): good. A tighter picture at 640 means it is a crop: bad for `top` and `side`.
- [ ] **Step 3:** With all three cameras attached, record a 10 s test through the Record page at 640×480 @ 30. Read the log: no "frames arriving too slowly".
- [ ] **Step 4 — decide and write down in `CHESS_DEMO_GUIDE.md` (one line under "Calibrate arms and cameras"):**

| Result | Decision |
|---|---|
| All three deliver 4:3 at 640×480 with the same field of view as at 1080p and hold ≥ 29 fps together | Contract stays 640×480 @ 30. Calibrate the lens at the recording size: `uv run lelab-camera-lens --size 640x480`. Replace the two `1920x1080` mentions in the guide. |
| A camera only offers 16:9 (e.g. 848×480) | Change the sim **once**: `DATASET_SIZE = (848, 480)` in Task 2; regenerate the sim dataset. Do not mix. |
| Three cameras cannot hold 30 fps together at 640×480 | Stop. Put the third camera on its own USB controller or hub port; do not lower the fps (sim is 30). |

### Task 2: The camera contract

**Files:**

- Modify: `lelab/chess/transfers.py`, `lelab/sim/generate.py`, `lelab/sim/env.py`
- Modify: `tests/chess_render.py`, `tests/test_chess_transfers.py`, `tests/test_record.py`, `tests/test_sim_generate.py`

**Interfaces:**

- `transfers.py` gains `DATASET_CAMERAS = ("top", "wrist", "side")`, `POLICY_CAMERAS = ("top", "wrist")`, `DATASET_SIZE = (640, 480)` (width, height), `DATASET_FPS = 30`, and `camera_contract_problems(cameras: dict[str, dict]) -> list[str]`.
- `chess_mode_problem` (via `chess_mode_calibration`) refuses to start unless the problems list is empty.
- `tests/chess_render.py` gains `contract_cameras(**overrides) -> dict[str, dict]`.

- [ ] **Step 1: Failing tests.** Append to `tests/chess_render.py`:

```python
def contract_cameras(**overrides: dict) -> dict[str, dict]:
    """The three recording cameras at the dataset contract; `overrides` patch one camera by name."""
    cameras = {
        name: {"type": "opencv", "camera_index": i, "width": 640, "height": 480, "fps": 30, "rotation": 0}
        for i, name in enumerate(("top", "wrist", "side"))
    }
    for name, patch in overrides.items():
        cameras[name] = {**cameras[name], **patch}
    return cameras
```

Append to `tests/test_chess_transfers.py` (merge imports):

```python
from lelab.chess.transfers import camera_contract_problems
from tests.chess_render import contract_cameras


def test_the_contract_cameras_have_no_problems() -> None:
    assert camera_contract_problems(contract_cameras()) == []


def test_a_missing_camera_is_named() -> None:
    cameras = contract_cameras()
    del cameras["side"]
    assert any("'side'" in p for p in camera_contract_problems(cameras))


def test_size_and_fps_must_match_the_sim_datasets() -> None:
    found = camera_contract_problems(contract_cameras(wrist={"width": 1920, "height": 1080}, side={"fps": 15}))
    assert any("'wrist'" in p and "1920x1080" in p for p in found)
    assert any("'side'" in p and "15" in p for p in found)


def test_unset_size_is_a_problem_because_the_camera_would_pick_its_own() -> None:
    cameras = contract_cameras()
    del cameras["top"]["width"]
    assert any("'top'" in p and "width" in p for p in camera_contract_problems(cameras))


def test_a_quarter_turn_swaps_the_delivered_size() -> None:
    # 480x640 native, rotated 90 degrees, delivers 640x480
    assert camera_contract_problems(contract_cameras(wrist={"width": 480, "height": 640, "rotation": 90})) == []


def test_chess_mode_refuses_a_camera_off_contract(tmp_lerobot_home) -> None:
    save_calibration(calibration())
    assert chess_mode_problem(contract_cameras()) is None
    assert "640x480" in chess_mode_problem(contract_cameras(top={"width": 1920, "height": 1080}))
```

Edit `test_chess_mode_problem_checks_top_camera_rotation_and_calibration` (the contract check now runs before the calibration check, so the "Calibrate" case needs contract cameras): the case `"Calibrate" in chess_mode_problem({"top": {"rotation": 0}})` becomes `"Calibrate" in chess_mode_problem(contract_cameras())`; the two cases that expect `None` (`{"top": {"rotation": 0}, "wrist": {"rotation": 180}}` and `{"top": {}}`) become one: `chess_mode_problem(contract_cameras(wrist={"rotation": 180})) is None`. The `"top" in …` and `"rotation" in …` cases stay. Then run `grep -n "chess_transfers=True" tests/test_record.py` and replace each fixture's `cameras={"top": {...}}` that is meant to pass the start check with `contract_cameras()`; fixtures that test the refusal keep their narrow dict.

Append to `tests/test_sim_generate.py`:

```python
def test_the_generator_uses_the_camera_contract() -> None:
    import inspect

    from lelab.chess.transfers import DATASET_CAMERAS, DATASET_FPS, DATASET_SIZE
    from lelab.sim import generate

    assert generate.CAMERAS == DATASET_CAMERAS
    assert generate.FPS == DATASET_FPS
    params = inspect.signature(generate.generate_dataset).parameters
    assert (params["width"].default, params["height"].default) == DATASET_SIZE
```

- [ ] **Step 2: Run:** `uv run pytest tests/test_chess_transfers.py tests/test_record.py tests/test_sim_generate.py -q`. Expected: `ImportError` for `camera_contract_problems`, plus failures where old fixtures lack cameras.
- [ ] **Step 3: Implement** in `lelab/chess/transfers.py`, below `TOP_CAMERA = "top"` (add `from lelab.camera_config import lerobot_camera_settings` to the imports):

```python
# The camera contract, shared by real recording, the dataset check and the sim generator.
# Sim and real datasets must have these cameras, this size and this fps to be mixed in one run.
DATASET_CAMERAS = ("top", "wrist", "side")
POLICY_CAMERAS = ("top", "wrist")  # what a policy reads by default; `side` is recorded for experiments
DATASET_SIZE = (640, 480)  # width, height as delivered, after rotation
DATASET_FPS = 30


def camera_contract_problems(cameras: dict[str, dict]) -> list[str]:
    """What stops these camera settings from producing a chess dataset the sim datasets can be mixed with."""
    problems: list[str] = []
    missing = [name for name in DATASET_CAMERAS if name not in cameras]
    if missing:
        quoted = ", ".join(f"'{name}'" for name in missing)
        problems.append(
            f"Chess transfer mode records {', '.join(DATASET_CAMERAS)}; add the camera named {quoted}."
        )
    width, height = DATASET_SIZE
    for name in DATASET_CAMERAS:
        camera = cameras.get(name)
        if camera is None or camera.get("type", "opencv") != "opencv":
            continue
        unset = [key for key in ("width", "height", "fps") if camera.get(key) is None]
        if unset:
            problems.append(f"Set the {'/'.join(unset)} of the '{name}' camera ({width}x{height} at {DATASET_FPS} fps).")
            continue
        settings = lerobot_camera_settings(camera)
        delivered = (settings["width"], settings["height"])
        if delivered != DATASET_SIZE:
            problems.append(
                f"'{name}' delivers {delivered[0]}x{delivered[1]}; the sim datasets are {width}x{height}."
            )
        if camera["fps"] != DATASET_FPS:
            problems.append(f"'{name}' runs at {camera['fps']} fps; the sim datasets are {DATASET_FPS} fps.")
    return problems
```

In `chess_mode_calibration`, after the rotation check and before loading the calibration, add:

```python
    problems = camera_contract_problems(cameras)
    if problems:
        raise ChessModeError(" ".join(problems))
```

In `lelab/sim/generate.py` replace the literal `CAMERAS = ("top", "wrist", "side")` and `FPS = 30` with `from lelab.chess.transfers import DATASET_CAMERAS, DATASET_FPS, DATASET_SIZE`, `CAMERAS = DATASET_CAMERAS`, `FPS = DATASET_FPS`, and make the `width: int = 640, height: int = 480` defaults in `settled_world` and `generate_dataset` read `DATASET_SIZE[0]` and `DATASET_SIZE[1]`. In `lelab/sim/env.py` do the same for `width`, `height`, `fps` and the `cameras=("top", "wrist", "side")` default. In `lelab/dataset_check.py` delete the local `POLICY_CAMERAS = ("top", "wrist")` line and add `POLICY_CAMERAS` to the existing `from .chess.transfers import CHESS_TASK, SIDECAR_NAME, read_transfer_sidecar` (so `dataset_check.POLICY_CAMERAS` stays importable).

- [ ] **Step 4: Run:** `uv run pytest tests/test_chess_transfers.py tests/test_record.py tests/test_sim_generate.py tests/test_sim_env.py tests/test_dataset_check.py -q`. Expected: pass.
- [ ] **Step 5: Commit:**

```bash
git add lelab/chess/transfers.py lelab/sim/generate.py lelab/sim/env.py lelab/dataset_check.py tests/chess_render.py tests/test_chess_transfers.py tests/test_record.py tests/test_sim_generate.py
git commit -m "feat: one camera contract for chess recording, datasets and the sim generator"
```

### Task 3: The dataset rule — compare the cameras the policy reads

**Files:** Modify `lelab/dataset_check.py`, `lelab/server.py` (the `/datasets/chess-check` route), `tests/test_dataset_check.py`.

**Interfaces:**

- `mismatches(a: Path, b: Path, cameras: Sequence[str] | None = None) -> list[str]`: with `cameras=None` it compares everything as today; with a list it compares fps, non-image features and only `observation.images.<c>` for `c` in `cameras`. A listed camera missing from either dataset is a mismatch.
- `dataset_cameras(dataset_dir: Path) -> list[str]`: the dataset's video cameras, contract order first.
- `GET /datasets/chess-check?repo_id=&against=&cameras=top,wrist` → the existing body plus `"cameras": [...]` and, when `against` is given and readable, `"shared_cameras": [...]`.

- [ ] **Step 1: Failing tests.** Append to `tests/test_dataset_check.py`:

```python
def test_extra_cameras_do_not_block_a_policy_that_ignores_them(tmp_path: Path) -> None:
    sim = make_dataset(tmp_path / "sim")
    real = make_dataset(tmp_path / "real", cameras=("top", "wrist"))
    assert any("observation.images.side" in p for p in dataset_check.mismatches(sim, real))
    assert dataset_check.mismatches(sim, real, cameras=("top", "wrist")) == []


def test_a_camera_the_policy_reads_must_exist_in_both(tmp_path: Path) -> None:
    sim = make_dataset(tmp_path / "sim")
    real = make_dataset(tmp_path / "real", cameras=("top", "wrist"))
    found = dataset_check.mismatches(sim, real, cameras=("top", "wrist", "side"))
    assert any("observation.images.side only in the first dataset" in p for p in found)


def test_selected_cameras_still_compare_shape_and_fps(tmp_path: Path) -> None:
    sim = make_dataset(tmp_path / "sim")
    real = make_dataset(tmp_path / "real", size=(1080, 1920), fps=15)
    found = dataset_check.mismatches(sim, real, cameras=("top",))
    assert any(p.startswith("fps") for p in found)
    assert any("observation.images.top" in p and "shape" in p for p in found)
    assert not any("observation.images.wrist" in p for p in found)


def test_dataset_cameras_come_in_contract_order(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "d", cameras=("side", "top", "wrist"))
    assert dataset_check.dataset_cameras(root) == ["top", "wrist", "side"]


def test_route_reports_cameras_and_applies_the_selection(cache: Path) -> None:
    make_dataset(cache / "local" / "sim")
    make_dataset(cache / "local" / "real", cameras=("top", "wrist"))
    body = (
        TestClient(server.app)
        .get(
            "/datasets/chess-check",
            params={"repo_id": "local/real", "against": "local/sim", "cameras": "top,wrist"},
        )
        .json()
    )
    assert body["cameras"] == ["top", "wrist"]
    assert body["shared_cameras"] == ["top", "wrist"]
    assert body["mismatches"] == []
```

- [ ] **Step 2: Run:** `uv run pytest tests/test_dataset_check.py -q`. Expected: `dataset_cameras` missing, `cameras` keyword rejected.
- [ ] **Step 3: Implement** in `lelab/dataset_check.py` (add `from collections.abc import Sequence` and `from .chess.transfers import DATASET_CAMERAS`):

```python
def dataset_cameras(dataset_dir: Path) -> list[str]:
    """The dataset's video cameras: the contract's cameras first, in contract order, then any others."""
    features = episode_media.read_info(dataset_dir).get("features") or {}
    prefix = episode_media.VIDEO_KEY_PREFIX
    found = [k[len(prefix) :] for k, v in features.items() if k.startswith(prefix) and v.get("dtype") == "video"]
    ordered = [c for c in DATASET_CAMERAS if c in found]
    return ordered + sorted(set(found) - set(ordered))


def _compared_keys(keys: set[str], cameras: Sequence[str] | None) -> list[str]:
    if cameras is None:
        return sorted(keys)
    prefix = episode_media.VIDEO_KEY_PREFIX
    wanted = {f"{prefix}{camera}" for camera in cameras}
    return sorted(k for k in keys if not k.startswith(prefix) or k in wanted) + sorted(
        wanted - keys
    )
```

Change `mismatches` to `def mismatches(a: Path, b: Path, cameras: Sequence[str] | None = None)` and its loop to `for key in _compared_keys(set(features_a) | set(features_b), cameras):`. A wanted camera that is in neither dataset also needs a message, so keep the existing `key not in features_a or key not in features_b` branch; in that branch, for a key in neither, the message must read `only in the first dataset` only when it is in `features_a`. Replace the branch body with:

```python
        if key not in features_a or key not in features_b:
            where = "the first" if key in features_a else "the second" if key in features_b else "neither"
            found.append(
                f"{key} is in neither dataset" if where == "neither" else f"{key} only in {where} dataset"
            )
            continue
```

Change `handle_chess_check(repo_id, against=None, cameras: Sequence[str] | None = None)`: after `result["checked"] = True` add `result["cameras"] = dataset_cameras(dataset_dir)`; in the `against` branch compute `other = episode_media.resolve_dataset_dir(against)`, `result["mismatches"] = mismatches(dataset_dir, other, cameras)` and `result["shared_cameras"] = [c for c in result["cameras"] if c in dataset_cameras(other)]`. Initialise `result["cameras"] = []` in the dict at the top so the unchecked body keeps its shape. In `lelab/server.py` change the route to

```python
@app.get("/datasets/chess-check")
def get_chess_check(repo_id: str, against: str | None = None, cameras: str | None = None):
    selected = [c for c in (cameras or "").split(",") if c] or None
    return dataset_check.handle_chess_check(repo_id, against, selected)
```

(keep the existing function name and docstring; only add the parameter and the split).

- [ ] **Step 4: Run:** `uv run pytest tests/test_dataset_check.py tests/test_record.py -q`. Expected: pass; the existing strict-comparison tests still pass because `cameras=None` is unchanged.
- [ ] **Step 5: Commit:** `git add lelab/dataset_check.py tests/test_dataset_check.py && git add -p lelab/server.py` then `git commit -m "feat: compare datasets on the cameras the policy reads"`.

### Task 4: Camera pose from four board corners

**Files:**

- Create: `lelab/chess/camera_pose.py`, `lelab/sim/camera_fit.py`
- Create: `tests/test_chess_camera_pose.py`, `tests/test_sim_camera_fit.py`; modify `tests/test_sim_world.py`
- Modify: `lelab/server.py` (one route)

**Interfaces:**

- `camera_pose.project(pos, xyaxes, fovy, size, points) -> np.ndarray (N, 2)`: the same pinhole model as `ChessWorld.project` (MuJoCo camera: looks along −z, square pixels, principal point at the centre, `fovy` vertical).
- `camera_pose.fit_camera(points3d, points2d, size, fovy=None, fallback_fovy=45.0) -> CameraFit`; `CameraFit(pos, xyaxes, fovy, rms_px, fov_fitted)`. With `fovy` given it is fixed. Without it the field of view is searched 20°–100° and kept only if the reprojection error rises by at least `FOV_FLAT_PX = 1.0` within ±5° of the best value; otherwise `fallback_fovy` is used and `fov_fitted` is `False`.
- `sim/camera_fit.py`: `world_corners(cell) -> np.ndarray (4, 3)` (a1, h1, h8, a8 at the board surface), `apply_fit(cell, camera: str, fit) -> SimCell`, `handle_camera_pose(body: dict) -> dict`, `CameraFitError`.
- `POST /chess/camera-pose` body `{"camera": "top"|"side", "image_size": [w, h], "board_corners": [[x, y] × 4], "fovy": number|null, "apply": bool}` → `{"camera", "pos", "xyaxes", "fovy", "rms_px", "fov_fitted", "warnings": [...], "applied": bool}`; 400 `{"status": "error", "message"}` on `CameraFitError`.

- [ ] **Step 1: Failing tests.** `tests/test_chess_camera_pose.py`:

```python
"""Pose fit from four board corners, on synthetic projections of the default sim cell."""

from __future__ import annotations

import numpy as np
import pytest

from lelab.chess.camera_pose import CameraPoseError, fit_camera, project
from lelab.sim.cell import SimCell

SIZE = (640, 480)
CELL = SimCell()
Z = CELL.board_thickness
X0, X1, H = CELL.board_x0, CELL.board_x0 + 8 * CELL.square_size, CELL.half
CORNERS = np.array([(X0, H, Z), (X0, -H, Z), (X1, -H, Z), (X1, H, Z)])  # a1, h1, h8, a8


def seen(pose, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return project(pose.pos, pose.xyaxes, pose.fovy, SIZE, CORNERS) + rng.normal(0, noise, (4, 2))


def test_the_oblique_side_camera_is_recovered_with_its_field_of_view() -> None:
    fit = fit_camera(CORNERS, seen(CELL.side_camera), SIZE)
    assert fit.fov_fitted
    assert fit.fovy == pytest.approx(CELL.side_camera.fovy, abs=0.6)
    assert np.linalg.norm(np.array(fit.pos) - CELL.side_camera.pos) < 0.003
    assert fit.rms_px < 0.1


def test_noisy_clicks_still_land_within_the_randomization_width() -> None:
    fit = fit_camera(CORNERS, seen(CELL.side_camera, noise=1.0), SIZE)
    assert fit.fovy == pytest.approx(CELL.side_camera.fovy, abs=3.0)
    assert np.linalg.norm(np.array(fit.pos) - CELL.side_camera.pos) < 0.03


def test_a_camera_looking_straight_down_cannot_tell_its_field_of_view() -> None:
    fit = fit_camera(CORNERS, seen(CELL.top_camera), SIZE, fallback_fovy=39.0)
    assert not fit.fov_fitted
    assert fit.fovy == 39.0


def test_a_given_field_of_view_is_kept_and_the_pose_follows() -> None:
    fit = fit_camera(CORNERS, seen(CELL.top_camera), SIZE, fovy=CELL.top_camera.fovy)
    assert not fit.fov_fitted
    assert fit.fovy == CELL.top_camera.fovy
    assert np.linalg.norm(np.array(fit.pos) - CELL.top_camera.pos) < 0.003


def test_the_fitted_pose_reprojects_onto_the_clicks() -> None:
    clicks = seen(CELL.side_camera, noise=0.5)
    fit = fit_camera(CORNERS, clicks, SIZE)
    again = project(fit.pos, fit.xyaxes, fit.fovy, SIZE, CORNERS)
    assert np.sqrt(np.mean(np.sum((again - clicks) ** 2, axis=1))) == pytest.approx(fit.rms_px)


@pytest.mark.parametrize("points", [np.zeros((3, 2)), np.full((4, 2), np.nan), np.zeros((4, 3))])
def test_unusable_clicks_raise(points: np.ndarray) -> None:
    with pytest.raises(CameraPoseError):
        fit_camera(CORNERS, points, SIZE)
```

Append to `tests/test_sim_world.py` (it already has the `world` fixture and its skips):

```python
def test_numpy_camera_model_matches_the_world_projection(world) -> None:
    import numpy as np

    from lelab.chess.camera_pose import project

    points = np.array([[0.2, 0.0, 0.0], [0.3, 0.05, 0.03], [0.16, -0.1, 0.004]])
    for name, pose in (("top", world.cell.top_camera), ("side", world.cell.side_camera)):
        got = project(pose.pos, pose.xyaxes, pose.fovy, (world.width, world.height), points)
        np.testing.assert_allclose(got, world.project(name, points), atol=1e-3)
```

`tests/test_sim_camera_fit.py`:

```python
"""Applying a camera fit to the cell, and the route that does it."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from lelab.chess.camera_pose import project
from lelab.server import app
from lelab.sim.camera_fit import CameraFitError, handle_camera_pose, world_corners
from lelab.sim.cell import CameraPose, SimCell, load_cell, save_cell

SIZE = (640, 480)


def side_clicks(cell: SimCell) -> list[list[float]]:
    pose = cell.side_camera
    return project(pose.pos, pose.xyaxes, pose.fovy, SIZE, world_corners(cell)).tolist()


def body(cell: SimCell, **extra) -> dict:
    return {"camera": "side", "image_size": list(SIZE), "board_corners": side_clicks(cell), "fovy": None, **extra}


def test_world_corners_are_a1_h1_h8_a8_on_the_board_surface() -> None:
    cell = SimCell()
    corners = world_corners(cell)
    assert corners.shape == (4, 3)
    assert corners[0][:2] == pytest.approx((cell.board_x0, cell.half))  # a1: rank 1 edge, file a on the left
    assert corners[2][:2] == pytest.approx((cell.board_x0 + 8 * cell.square_size, -cell.half))
    assert set(corners[:, 2]) == {cell.board_thickness}


def test_fit_without_apply_leaves_the_cell_alone(tmp_lerobot_home) -> None:
    out = handle_camera_pose(body(SimCell()))
    assert out["applied"] is False
    assert out["rms_px"] < 0.5
    assert load_cell() == SimCell()


def test_apply_writes_the_fitted_pose_into_the_cell(tmp_lerobot_home) -> None:
    moved_pose = CameraPose((0.12, -0.30, 0.32), (0.9, -0.42, 0, 0.3, 0.64, 0.71), 47.0)
    moved = SimCell(side_camera=moved_pose)
    save_cell(SimCell())
    out = handle_camera_pose(body(moved, apply=True))
    assert out["applied"] is True
    written = load_cell().side_camera
    assert np.linalg.norm(np.array(written.pos) - moved.side_camera.pos) < 0.004
    assert written.fovy == pytest.approx(47.0, abs=0.6)
    assert load_cell().top_camera == SimCell().top_camera  # the other camera is untouched


def test_a_flat_field_of_view_keeps_the_cell_value_and_warns(tmp_lerobot_home) -> None:
    cell = SimCell()
    pose = cell.top_camera
    clicks = project(pose.pos, pose.xyaxes, pose.fovy, SIZE, world_corners(cell)).tolist()
    out = handle_camera_pose({"camera": "top", "image_size": list(SIZE), "board_corners": clicks, "fovy": None})
    assert out["fov_fitted"] is False
    assert out["fovy"] == cell.top_camera.fovy
    assert any("field of view" in w for w in out["warnings"])


@pytest.mark.parametrize(
    "patch",
    [{"camera": "wrist"}, {"image_size": [1920, 1080]}, {"board_corners": [[0, 0]] * 3}, {"image_size": [0, 480]}],
)
def test_bad_requests_are_refused(tmp_lerobot_home, patch: dict) -> None:
    with pytest.raises(CameraFitError):
        handle_camera_pose({**body(SimCell()), **patch})


def test_route_returns_400_with_the_reason(tmp_lerobot_home) -> None:
    response = TestClient(app).post("/chess/camera-pose", json={**body(SimCell()), "camera": "wrist"})
    assert response.status_code == 400
    assert "camera" in response.json()["message"]
```

- [ ] **Step 2: Run:** `uv run pytest tests/test_chess_camera_pose.py tests/test_sim_camera_fit.py -q`. Expected: cannot import `lelab.chess.camera_pose`.
- [ ] **Step 3: Implement** `lelab/chess/camera_pose.py`:

```python
"""A camera's pose in the arm frame from clicked board corners.

The model is the one MuJoCo renders with and `ChessWorld.project` uses: a pinhole looking along the
camera's -z axis, square pixels, the principal point at the centre, `fovy` the vertical field of view.
The clicks must be on a straightened picture (the lens correction applied), or the model does not hold.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

FOV_GRID = np.arange(20.0, 100.01, 0.5)
FOV_FLAT_PX = 1.0  # the error must rise this much within FOV_PROBE_DEG of the best field of view
FOV_PROBE_DEG = 5.0


class CameraPoseError(ValueError):
    """The clicks or the points cannot give a pose."""


@dataclass(frozen=True)
class CameraFit:
    pos: tuple[float, float, float]
    xyaxes: tuple[float, float, float, float, float, float]
    fovy: float
    rms_px: float
    fov_fitted: bool


def _focal(fovy: float, height: int) -> float:
    return 0.5 * height / math.tan(math.radians(fovy) / 2)


def project(pos, xyaxes, fovy: float, size: tuple[int, int], points) -> np.ndarray:
    """World points (N, 3) -> pixels (N, 2) in a `size` = (width, height) picture."""
    width, height = size
    x = np.asarray(xyaxes[:3], dtype=np.float64)
    y = np.asarray(xyaxes[3:], dtype=np.float64)
    x = x / np.linalg.norm(x)
    y = y - x * (x @ y)  # MuJoCo orthogonalises xyaxes the same way
    y = y / np.linalg.norm(y)
    rot = np.stack([x, y, np.cross(x, y)], axis=1)
    local = (np.asarray(points, dtype=np.float64) - np.asarray(pos, dtype=np.float64)) @ rot
    focal = _focal(fovy, height)
    return np.stack(
        [width / 2 + focal * local[:, 0] / -local[:, 2], height / 2 - focal * local[:, 1] / -local[:, 2]],
        axis=1,
    )


def _solve(points3d: np.ndarray, points2d: np.ndarray, fovy: float, size: tuple[int, int]) -> CameraFit:
    width, height = size
    focal = _focal(fovy, height)
    k = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(points3d, points2d, k, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise CameraPoseError("The corners do not give a camera pose; click them again.")
    rvec, tvec = cv2.solvePnPRefineLM(points3d, points2d, k, None, rvec, tvec)
    rotation, _ = cv2.Rodrigues(rvec)
    to_world = rotation.T  # columns: the OpenCV camera's x (right), y (down), z (forward) in the world
    pos = tuple(float(v) for v in (-to_world @ tvec).ravel())
    xyaxes = tuple(float(v) for v in (*to_world[:, 0], *(-to_world[:, 1])))  # MuJoCo's y is up
    error = project(pos, xyaxes, fovy, size, points3d) - points2d
    rms = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
    return CameraFit(pos, xyaxes, float(fovy), rms, False)  # type: ignore[arg-type]


def fit_camera(
    points3d: Sequence[Sequence[float]],
    points2d: Sequence[Sequence[float]],
    size: tuple[int, int],
    fovy: float | None = None,
    fallback_fovy: float = 45.0,
) -> CameraFit:
    p3 = np.asarray(points3d, dtype=np.float64)
    p2 = np.asarray(points2d, dtype=np.float64)
    if p3.ndim != 2 or p3.shape[1] != 3 or p2.shape != (len(p3), 2) or len(p3) < 4:
        raise CameraPoseError("Need at least four corners, as [x, y] pixel pairs.")
    if not (np.isfinite(p3).all() and np.isfinite(p2).all()):
        raise CameraPoseError("The corners must be finite numbers.")
    if fovy is not None:
        return _solve(p3, p2, float(fovy), size)
    errors = np.array([_solve(p3, p2, float(f), size).rms_px for f in FOV_GRID])
    best = int(np.argmin(errors))
    probe = int(round(FOV_PROBE_DEG / (FOV_GRID[1] - FOV_GRID[0])))
    lo, hi = max(best - probe, 0), min(best + probe, len(FOV_GRID) - 1)
    if max(errors[lo], errors[hi]) - errors[best] < FOV_FLAT_PX:
        return _solve(p3, p2, float(fallback_fovy), size)
    fit = _solve(p3, p2, float(FOV_GRID[best]), size)
    return CameraFit(fit.pos, fit.xyaxes, fit.fovy, fit.rms_px, True)
```

`lelab/sim/camera_fit.py`:

```python
"""Write a fitted camera pose into the sim cell, from the Setup page's four clicked board corners."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from lelab.chess.camera_pose import CameraFit, CameraPoseError, fit_camera
from lelab.chess.transfers import DATASET_SIZE

from .cell import CameraPose, SimCell, load_cell, save_cell

CAMERA_FIELDS = {"top": "top_camera", "side": "side_camera"}
MAX_RMS_PX = 3.0


class CameraFitError(ValueError):
    """The request cannot be fitted."""


def world_corners(cell: SimCell) -> np.ndarray:
    """a1, h1, h8, a8 on the board surface, in the arm frame (as `ChessWorld.board_calibration` projects them)."""
    z = cell.board_thickness
    x0, x1, h = cell.board_x0, cell.board_x0 + 8 * cell.square_size, cell.half
    return np.array([(x0, h, z), (x0, -h, z), (x1, -h, z), (x1, h, z)], dtype=np.float64)


def apply_fit(cell: SimCell, camera: str, fit: CameraFit) -> SimCell:
    pose = CameraPose(fit.pos, fit.xyaxes, fit.fovy)
    return replace(cell, **{CAMERA_FIELDS[camera]: pose})


def _size(value: Any) -> tuple[int, int]:
    try:
        width, height = (int(v) for v in value)
    except (TypeError, ValueError) as e:
        raise CameraFitError("image_size must be [width, height]") from e
    if width <= 0 or height <= 0:
        raise CameraFitError("image_size must be positive")
    if abs(width / height - DATASET_SIZE[0] / DATASET_SIZE[1]) > 0.01:
        raise CameraFitError(
            f"The reference is {width}x{height}; the sim renders {DATASET_SIZE[0]}x{DATASET_SIZE[1]} (4:3). "
            "Set the camera to a 4:3 mode and capture the reference again."
        )
    return width, height


def handle_camera_pose(body: dict[str, Any]) -> dict[str, Any]:
    camera = body.get("camera")
    if camera not in CAMERA_FIELDS:
        raise CameraFitError(f"camera must be one of {sorted(CAMERA_FIELDS)}")
    size = _size(body.get("image_size"))
    corners = body.get("board_corners")
    if not isinstance(corners, list) or len(corners) != 4:
        raise CameraFitError("board_corners needs exactly four [x, y] points: a1, h1, h8, a8")
    cell = load_cell()
    current: CameraPose = getattr(cell, CAMERA_FIELDS[camera])
    fovy = body.get("fovy")
    try:
        fit = fit_camera(world_corners(cell), corners, size, fovy=fovy, fallback_fovy=current.fovy)
    except CameraPoseError as e:
        raise CameraFitError(str(e)) from e
    warnings: list[str] = []
    if fovy is None and not fit.fov_fitted:
        warnings.append(
            f"This view cannot tell its field of view from four corners; kept {fit.fovy:g}°. "
            "Pass the field of view of the straightened picture if it is not that."
        )
    if fit.rms_px > MAX_RMS_PX:
        warnings.append(
            f"The corners are {fit.rms_px:.1f} px from the fitted board; check the click order (a1, h1, h8, a8)."
        )
    applied = body.get("apply") is True
    if applied:
        save_cell(apply_fit(cell, camera, fit))
    return {
        "camera": camera,
        "pos": list(fit.pos),
        "xyaxes": list(fit.xyaxes),
        "fovy": fit.fovy,
        "rms_px": fit.rms_px,
        "fov_fitted": fit.fov_fitted,
        "warnings": warnings,
        "applied": applied,
    }
```

In `lelab/server.py`, next to the other `/chess/*` routes add (and `from .sim.camera_fit import CameraFitError, handle_camera_pose` to the imports; `lelab/sim/cell.py` does not import MuJoCo):

```python
@app.post("/chess/camera-pose")
async def post_camera_pose(request: Request):
    """Fit `top` or `side` from the four board corners clicked on its reference; `apply` writes sim_cell.json."""
    try:
        return handle_camera_pose(await request.json())
    except (CameraFitError, ValueError) as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})
```

- [ ] **Step 4: Run:** `uv run pytest tests/test_chess_camera_pose.py tests/test_sim_camera_fit.py tests/test_sim_world.py -q` (the last outside the sandbox). Expected: pass. If `test_noisy_clicks…` is flaky for some seed, raise `abs=3.0` no further than 4.0 and report it; the prototype's median at 1 px noise was 0.9° and 10 mm.
- [ ] **Step 5: Commit:** `git add lelab/chess/camera_pose.py lelab/sim/camera_fit.py tests/test_chess_camera_pose.py tests/test_sim_camera_fit.py tests/test_sim_world.py && git add -p lelab/server.py` then `git commit -m "feat: fit a camera pose from four board corners and write it to the sim cell"`.

### Task 5: Setup — one pose card per camera

**Files:**

- Create: `frontend/src/lib/cameraPoseApi.ts`, `frontend/src/lib/cameraPoseApi.test.ts`
- Create: `frontend/src/components/setup/CameraPoseCard.tsx`, `frontend/src/components/setup/CameraPoseCard.test.tsx`
- Modify: `frontend/src/pages/Setup.tsx`

**Interfaces:**

- `cameraPoseApi.ts`: `CameraName = "top" | "side"`, `PoseFit { camera, pos, xyaxes, fovy, rms_px, fov_fitted, warnings, applied }`, `fitCameraPose(baseUrl, fetcher, body: { camera; image_size: [number, number]; board_corners: Point[]; fovy: number | null; apply: boolean }): Promise<PoseFit>` (throws `ApiError` with the server message).
- `CameraPoseCard({ robot, camera })`: renders the camera's saved reference (`useCameraReference(robot, camera)`), collects four clicks (a1, h1, h8, a8), **Fit** (`apply: false`) shows position in mm, field of view with "fitted" or "kept", the pixel error and the warnings; **Write to sim cell** (`apply: true`, enabled only after a fit with `rms_px` ≤ 3); **Reset points**. For `top` an extra button **Use saved board corners** fills the four points from `GET /chess/board-calibration` (`board_corners`, scaled by `image_size` of the reference), so `top` needs no second round of clicks.

- [ ] **Step 1: Failing tests.** `frontend/src/lib/cameraPoseApi.test.ts`:

```ts
import { describe, expect, it, vi } from "vitest";
import { fitCameraPose } from "./cameraPoseApi";

const BASE = "http://localhost:8000";
const body = {
  camera: "side" as const,
  image_size: [640, 480] as [number, number],
  board_corners: [[1, 2], [3, 4], [5, 6], [7, 8]] as Array<[number, number]>,
  fovy: null,
  apply: false,
};

describe("fitCameraPose", () => {
  it("POSTs JSON and returns the fit", async () => {
    const fit = { camera: "side", pos: [0.1, -0.3, 0.3], xyaxes: [1, 0, 0, 0, 1, 0], fovy: 45, rms_px: 0.4, fov_fitted: true, warnings: [], applied: false };
    const fetcher = vi.fn().mockResolvedValue({ ok: true, status: 200, json: () => Promise.resolve(fit) } as Response);
    expect(await fitCameraPose(BASE, fetcher, body)).toEqual(fit);
    expect(fetcher).toHaveBeenCalledWith(
      `${BASE}/chess/camera-pose`,
      expect.objectContaining({ method: "POST", body: JSON.stringify(body) }),
    );
  });

  it("throws the server's reason", async () => {
    const fetcher = vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: () => Promise.resolve({ message: "The reference is 1920x1080" }),
    } as Response);
    await expect(fitCameraPose(BASE, fetcher, body)).rejects.toThrow("The reference is 1920x1080");
  });
});
```

`CameraPoseCard.test.tsx` mirrors `BoardCalibrationCard.test.tsx` (same `stubFetch`/`mountImage` helpers, serving `/robots/arm/camera-reference/side` and `/chess/camera-pose`). Cases: (1) four clicks on `side` enable **Fit**; Fit posts `board_corners` equal to the four clicked points and `apply: false`, and the page shows `fitted` and the pixel error; (2) **Write to sim cell** is disabled until a fit exists and disabled again when `rms_px` is 3.5, and clicking it posts `apply: true`; (3) with `camera="top"`, **Use saved board corners** fills four points from a stubbed `/chess/board-calibration` and **Fit** is enabled without any click; (4) a missing reference shows "Capture the side camera reference on the Calibration page".

- [ ] **Step 2: Run:** `cd frontend && npm test -- --run src/lib/cameraPoseApi.test.ts src/components/setup/CameraPoseCard.test.tsx`. Expected: cannot resolve the modules.
- [ ] **Step 3: Implement.** `cameraPoseApi.ts` follows `boardCalibrationApi.ts` (`ApiError`, the same `fail()` shape; copy it, do not import a private helper):

```ts
import { ApiError, Fetcher } from "./apiClient";
import type { Point } from "./boardCalibrationApi";

export type CameraName = "top" | "side";

export interface PoseFit {
  camera: CameraName;
  pos: [number, number, number];
  xyaxes: number[];
  fovy: number;
  rms_px: number;
  fov_fitted: boolean;
  warnings: string[];
  applied: boolean;
}

export interface PoseRequest {
  camera: CameraName;
  image_size: [number, number];
  board_corners: Point[];
  fovy: number | null;
  apply: boolean;
}

export async function fitCameraPose(baseUrl: string, fetcher: Fetcher, request: PoseRequest): Promise<PoseFit> {
  const r = await fetcher(`${baseUrl}/chess/camera-pose`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!r.ok) {
    let message = `Fit camera pose failed: ${r.status}`;
    try {
      const err = (await r.json()) as { message?: string };
      if (err?.message) message = err.message;
    } catch {
      /* no JSON body */
    }
    throw new ApiError(message, r.status, null);
  }
  return (await r.json()) as PoseFit;
}
```

`CameraPoseCard.tsx` reuses the click-and-marker pattern of `BoardCalibrationCard` (same `imagePoint`, same `errorText`, markers `#f59e0b`) with one group of four hints `["a1", "h1", "h8", "a8"]`; it keeps `fit: PoseFit | null` in state, clears it when points change, and renders `pos` as `x y z` in millimetres (`pos * 1000`, rounded), `fovy` to one decimal with "(fitted)" or "(kept)", and `rms_px` to one decimal. In `Setup.tsx` add a second list item after Board calibration:

```tsx
        <li className="rounded-md border border-border p-4">
          <span className="font-bold">Camera positions for the simulation</span>
          <span className="mt-1 mb-3 block text-sm text-muted-foreground">
            Click the four board corners on each camera's straightened reference. The fit is written to the sim
            cell so simulated top and side pictures match the real ones.
          </span>
          <div className="flex flex-col gap-6">
            <CameraPoseCard robot={robotName ?? undefined} camera="side" />
            <CameraPoseCard robot={robotName ?? undefined} camera="top" />
          </div>
        </li>
```

- [ ] **Step 4: Run:** `cd frontend && npm test -- --run && npx tsc --noEmit -p tsconfig.app.json`. Expected: pass.
- [ ] **Step 5: Commit:** `git add frontend/src/lib/cameraPoseApi.ts frontend/src/lib/cameraPoseApi.test.ts frontend/src/components/setup/CameraPoseCard.tsx frontend/src/components/setup/CameraPoseCard.test.tsx frontend/src/pages/Setup.tsx && git commit -m "feat: fit and write the real top and side camera poses on Setup"`.

### Task 6: Parity sheet — sim next to real

**Files:** Create `lelab/sim/parity.py`, `tests/test_sim_parity.py`; modify `lelab/sim/__main__.py`.

**Interfaces:**

- `parity.blend(real, sim, alpha=0.5) -> np.ndarray` (both `uint8` HxWx3 of the same shape).
- `parity.sheet(rows: list[tuple[str, np.ndarray, np.ndarray]]) -> np.ndarray`: one row per camera, `real | sim | blend`, the camera name written at the top-left.
- `parity.run_parity(robot: str, out: Path) -> list[str]`: loads `camera_reference` PNGs for `top`, `side`, `wrist`, renders the nominal sim start position (no overlay, no randomization) with `ChessWorld(load_cell(), SimPosition(chess.Board()))`, writes the sheet, returns the cameras it found; a camera without a reference is skipped and listed in the printed message. `python -m lelab.sim parity --robot arm --out parity.png`.

- [ ] **Step 1: Failing tests:**

```python
"""Pure parts of the parity sheet; rendering is exercised by `python -m lelab.sim parity`."""

from __future__ import annotations

import numpy as np
import pytest

from lelab.sim.parity import blend, sheet

BLACK = np.zeros((480, 640, 3), np.uint8)
WHITE = np.full((480, 640, 3), 255, np.uint8)


def test_blend_is_the_mean_at_half_alpha() -> None:
    assert blend(BLACK, WHITE).mean() == pytest.approx(127.5, abs=1)


def test_blend_refuses_different_shapes() -> None:
    with pytest.raises(ValueError):
        blend(BLACK, WHITE[:100])


def test_sheet_has_three_columns_and_a_row_per_camera() -> None:
    out = sheet([("top", BLACK, WHITE), ("side", WHITE, BLACK)])
    assert out.shape == (2 * 480, 3 * 640, 3)
    assert out.dtype == np.uint8
    assert out[:, :640].mean() > 0  # the camera name is drawn on the first real picture
    assert (out[0:480, 640:1280] == 255).all()  # the sim column is untouched
```

- [ ] **Step 2: Run:** `uv run pytest tests/test_sim_parity.py -q`. Expected: cannot import `lelab.sim.parity`.
- [ ] **Step 3: Implement** `lelab/sim/parity.py`:

```python
"""Real references next to the simulated cell, to judge camera pose, field of view and wrist orientation."""

from __future__ import annotations

from pathlib import Path

import chess
import cv2
import numpy as np

from lelab import camera_reference
from lelab.chess.transfers import DATASET_SIZE

CAMERAS = ("top", "side", "wrist")


def blend(real: np.ndarray, sim: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if real.shape != sim.shape:
        raise ValueError(f"shapes differ: {real.shape} vs {sim.shape}")
    return cv2.addWeighted(real, 1 - alpha, sim, alpha, 0)


def sheet(rows: list[tuple[str, np.ndarray, np.ndarray]]) -> np.ndarray:
    lines = []
    for name, real, sim in rows:
        labelled = real.copy()
        cv2.putText(labelled, name, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        lines.append(np.concatenate([labelled, sim, blend(real, sim)], axis=1))
    return np.concatenate(lines, axis=0)


def _reference(robot: str, camera: str) -> np.ndarray | None:
    png = camera_reference.load_reference(robot, camera)
    if png is None:
        return None
    bgr = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    width, height = DATASET_SIZE
    if abs(rgb.shape[1] / rgb.shape[0] - width / height) > 0.01:
        raise ValueError(f"The {camera} reference is {rgb.shape[1]}x{rgb.shape[0]}, not 4:3; capture it again.")
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def run_parity(robot: str, out: Path) -> list[str]:
    from .cell import load_cell
    from .scene import SimPosition
    from .world import ChessWorld

    width, height = DATASET_SIZE
    world = ChessWorld(load_cell(), SimPosition(chess.Board()), width=width, height=height)
    try:
        rows, found = [], []
        for camera in CAMERAS:
            real = _reference(robot, camera)
            if real is not None:
                rows.append((camera, real, world.render(camera)))
                found.append(camera)
    finally:
        world.close()
    if rows:
        cv2.imwrite(str(out), cv2.cvtColor(sheet(rows), cv2.COLOR_RGB2BGR))
    return found
```

In `lelab/sim/__main__.py` add

```python
def _parity(robot: str, out: str) -> int:
    from .parity import CAMERAS, run_parity

    found = run_parity(robot, Path(out))
    missing = [c for c in CAMERAS if c not in found]
    if not found:
        print(f"No camera references for {robot}; capture them on the Calibration page.")
        return 1
    print(f"{out}: {', '.join(found)}" + (f" (no reference for {', '.join(missing)})" if missing else ""))
    return 0
```

a subparser `parity = sub.add_parser("parity", help="real references | sim render | blend, per camera")` with `--robot` (required) and `--out` (default `Path(tempfile.gettempdir()) / "garage_parity.png"`), and dispatch it next to the other commands. Update the module docstring to list `parity`.

- [ ] **Step 4: Run:** `uv run pytest tests/test_sim_parity.py -q` (pass) and, outside the sandbox with references captured, `uv run python -m lelab.sim parity --robot <robot> --out /tmp/parity.png`; open the PNG. Judge: board edges in the blend column lie on top of each other within a few pixels for `top` and `side`; the wrist picture is the same way up and shows the gripper jaws in the same place.
- [ ] **Step 5: Commit:** `git add lelab/sim/parity.py lelab/sim/__main__.py tests/test_sim_parity.py && git commit -m "feat: parity sheet of real references against the simulated cell"`.

### Task 7: The switches for the side-camera experiment (no experiment run)

**Files:** Modify `lelab/train.py`, `lelab/sim/evaluate.py`, `lelab/forge/weave_eval.py`, `tests/test_train.py`, `tests/test_sim_evaluate.py`, `tests/test_forge_weave_eval.py`; Training page checkboxes in `frontend/src/pages/Training.tsx` (small).

**Interfaces:**

- `TrainingRequest.policy_cameras: list[str] | None = None`. With a list, `build_training_command` appends `--policy.input_features=<JSON>` holding `observation.state` and one `observation.images.<c>` per camera, with the contract's shapes; unset keeps LeRobot's inference from the dataset. Only valid for chess datasets (the shapes are the contract's), so a list outside `DATASET_CAMERAS` raises `ValueError`.
- `EvalReport.cameras: tuple[str, ...] = ()`, written to `eval_report.json` and to Weave (`inputs["cameras"]` per trial, `cameras` in the summary). `TrialRow.cameras: tuple[str, ...] = ()` logs the same for real trials.

- [ ] **Step 1: Spike (10 minutes, outside the sandbox):** confirm LeRobot v0.7.0 trains on a camera subset. Generate 2 sim episodes (`uv run python -m lelab.sim generate --repo-id local/cam-spike --episodes 2 --no-artifact`), then

```bash
uv run python -m lerobot.scripts.lerobot_train --dataset.repo_id local/cam-spike --policy.type act --steps 2 --batch_size 2 --policy.device cpu --policy.push_to_hub false \
  --policy.input_features='{"observation.state":{"type":"STATE","shape":[6]},"observation.images.top":{"type":"VISUAL","shape":[3,480,640]},"observation.images.wrist":{"type":"VISUAL","shape":[3,480,640]}}'
```

Expected: trains two steps; the saved `config.json` lists only `top` and `wrist`. If draccus rejects the JSON, use `lelab.forge.remote.train_entry`-style wrapping (set `cfg.policy.input_features` in a small entry module) and keep the same `policy_cameras` interface; record the finding at the top of this task. If the extra `side` video is still decoded, note the cost (it is decoded, never used).

- [ ] **Step 2: Failing tests.** `tests/test_train.py`:

```python
def test_policy_cameras_become_explicit_input_features() -> None:
    import json

    from lelab.train import TrainingRequest, build_training_command

    request = TrainingRequest(dataset_repo_id="local/chess", policy_cameras=["top", "wrist"])
    cmd = build_training_command(request, "out")
    flag = next(a for a in cmd if a.startswith("--policy.input_features="))
    features = json.loads(flag.split("=", 1)[1])
    assert set(features) == {"observation.state", "observation.images.top", "observation.images.wrist"}
    assert features["observation.images.top"] == {"type": "VISUAL", "shape": [3, 480, 640]}


def test_without_policy_cameras_lerobot_infers_them() -> None:
    from lelab.train import TrainingRequest, build_training_command

    cmd = build_training_command(TrainingRequest(dataset_repo_id="local/chess"), "out")
    assert not any(a.startswith("--policy.input_features") for a in cmd)


def test_an_unknown_camera_is_refused() -> None:
    import pytest

    from lelab.train import TrainingRequest, build_training_command

    with pytest.raises(ValueError, match="overhead"):
        build_training_command(TrainingRequest(dataset_repo_id="local/chess", policy_cameras=["overhead"]), "out")
```

`tests/test_sim_evaluate.py` (reuse the file's existing fake-policy helpers; the new test builds a report with `cameras=("top", "wrist")` and checks `to_dict()["cameras"] == ["top", "wrist"]` and that `log_to_weave` passes `"cameras": "top,wrist"` in each prediction's `inputs` and in the summary using the file's existing fake `eval_logger`). `tests/test_forge_weave_eval.py`: a `TrialRow(..., cameras=("top", "wrist", "side"))` logs `inputs["cameras"] == "top,wrist,side"`; the default row logs `""`.

- [ ] **Step 3: Implement.** `lelab/train.py`: add the field after `dataset_image_transforms_enable`, and in `build_training_command` after the `--policy.type` line:

```python
    if request.policy_cameras is not None:
        unknown = [c for c in request.policy_cameras if c not in DATASET_CAMERAS]
        if unknown:
            raise ValueError(f"Unknown camera {unknown[0]!r}; chess datasets have {', '.join(DATASET_CAMERAS)}")
        width, height = DATASET_SIZE
        features = {"observation.state": {"type": "STATE", "shape": [len(CHESS_JOINTS)]}}
        for camera in request.policy_cameras:
            features[f"observation.images.{camera}"] = {"type": "VISUAL", "shape": [3, height, width]}
        cmd.append(f"--policy.input_features={json.dumps(features)}")
```

with `from lelab.chess.transfers import DATASET_CAMERAS, DATASET_SIZE` and `from lelab.dataset_check import CHESS_JOINTS` at the top (check `lelab.train` does not already import something that makes this circular; `dataset_check` imports only `episode_media` and `chess.transfers`). In `lelab/sim/evaluate.py`: `EvalReport` gets `cameras: tuple[str, ...] = ()`, `to_dict` adds `"cameras": list(self.cameras)`, `log_to_weave` adds `"cameras": ",".join(report.cameras)` to each `inputs` and `"cameras": list(report.cameras)` to `log_summary`; in `evaluate_checkpoint` after `run_plan(...)`'s `finally` add `report = dataclasses.replace(report, cameras=tuple(loaded.cameras))`. In `lelab/forge/weave_eval.py`: `TrialRow.cameras: tuple[str, ...] = ()` and `"cameras": ",".join(row.cameras)` in `inputs`.

On the Training page add three checkboxes (`top`, `wrist` checked and `side`) shown only when a dataset passes the chess check; the selection is sent as `policy_cameras` (omitted when all three are checked and the dataset has all three, so the default stays LeRobot's inference). Warn with the chess check's `mismatches` for the selection, as the page already does for the comparison dataset.

- [ ] **Step 4: Run:** `uv run pytest tests/test_train.py tests/test_sim_evaluate.py tests/test_forge_weave_eval.py -q && cd frontend && npm test -- --run`. Expected: pass.
- [ ] **Step 5: Commit:** by name, `git commit -m "feat: choose the policy's cameras at training and log them with every evaluation"`.

### Task 8: Docs, hardware checklist and graph

- [ ] **Step 1:** `CHESS_DEMO_GUIDE.md` (after Task 0's edits are committed, so the two do not collide): in "Calibrate arms and cameras" state the contract (`top`, `wrist`, `side`, 640×480 at 30 fps, `top` rotation 0), the Task 1 decision, and the lens size; add the Setup step "Camera positions for the simulation"; add troubleshooting rows: "Record refuses to start: 'delivers 1920x1080'" → set the camera form's size and fps to the contract; "pose fit warns the field of view is kept" → expected for `top`; pass the value measured on `side`; "pose fit RMS above 3 px" → corners clicked out of order or the reference is not straightened.
- [ ] **Step 2: Operator checklist** (record the outcome in the guide's M3 section):
  1. Task 0 acceptance holds for three cameras.
  2. All three cameras at 640×480 @ 30, the same table edge-to-edge as at 1080p (Task 1).
  3. Empty board in view, references captured for `top`, `side`, `wrist` with the lens straightened and the colour calibrated.
  4. Setup → Board calibration (16 points on `top`).
  5. Setup → Camera positions: `side` fit with RMS ≤ 3 px and a field of view within 5° of the lens tool's value; Write. `top`: Use saved board corners, pass the field of view that `side` fitted if both cameras are the same model with the same lens settings, Write.
  6. `uv run python -m lelab.sim parity --robot <robot>`: board edges coincide to a few pixels in `top` and `side`; wrist upright with the jaws in the same place. If the wrist is upside down, change the wrist camera's rotation (180) and recapture its reference; do not touch the sim.
  7. Regenerate the sim dataset (the cell changed): `uv run python -m lelab.sim generate ...`.
  8. Record 3 chess-transfer episodes; open them in the dataset browser: three cameras, markers on `top` only, then the Train page's chess check against the sim dataset reports no problems for `top`+`wrist` and for all three.
- [ ] **Step 3:** `uv run pytest -q`, `uv run ruff check lelab tests`, `uv run ruff format --check lelab tests`, `cd frontend && npm test -- --run && npx tsc --noEmit -p tsconfig.app.json`, then `graphify update .`.
- [ ] **Step 4:** Commit the guide and README by name: `git commit -m "docs: three-camera contract, pose fit and parity sheet in the chess guide"`.

---

## The side-camera experiment (designed here, run after the M3 gate)

**Question:** does adding `side` to the policy's inputs raise real success, or fix a named failure class, enough to justify a third stream?

**Arms (identical data, identical everything else):** A = `top`+`wrist`; B = `top`+`wrist`+`side`. One sim+real dataset mix, one `policy_cameras` setting per run, ≥ 3 training seeds per arm, same steps and batch size. The knobs exist after Task 7; the run itself is not part of M3.

**Evaluation:**

| Where | Trials per arm | What it can detect |
|---|---|---|
| Sim (Sandbox, randomized) | 20 transfers × 20 seeds = 400 | A gap of about 8 points; per transfer class about 20 points |
| Real arm | 20 transfers × 5 repeats = 100, same placements for both arms | A gap of about 20 points overall; per class, only large effects (class counts are small) |

Rough power: two proportions near 70 %, α = 0.05, 80 % power need about 330 trials per arm for a 10-point gap and about 150 for 15 points, so a real-arm "no difference" below 20 points is not evidence of none. Report 95 % intervals, not point estimates.

**Decision rule (write it before running):** keep `side` in the policy if sim shows ≥ +5 points with non-overlapping intervals across the training seeds **and** real shows ≥ +10 points, or it removes a failure class (`topple`, `place`, `drop`) that the corrections would otherwise need real episodes for. Otherwise `side` stays recorded for review and the VLM-free operator judging, not read by the policy. Camera set is a column in the Weave tables (Task 7), so the comparison is one filter.

**Risks to name in the report:** the real `side` pose is only as good as the fit (about 1 cm, 1°); a `side` that helps in sim may not help on real if the fit is off — the parity sheet is the first thing to check before reading a negative real result.

## Self-review

- **Spec coverage:** spec 2 (three cameras, all recorded, poses measured into the cell file) → Tasks 2, 4, 5; spec 5.1 (identical feature names and shapes) → Tasks 2, 3; spec 5.5 and 15 (camera poses measured at M3) → Tasks 4–6; the north star's "fewer than 100 real episodes" → Tasks 2 and 6 prevent mismatched data and Task 7 keeps the real corrections the only variable. Findings 1–7 each map to a task (1→0, 2→1–2, 3→3, 4→4–5, 5→6, 6→7, 7→7).
- **Placeholders:** none; the two spikes (Task 1, Task 7 Step 1) have explicit outcomes and fallbacks.
- **Type names:** `DATASET_CAMERAS`, `POLICY_CAMERAS`, `DATASET_SIZE`, `DATASET_FPS`, `camera_contract_problems` (Task 2) are used unchanged in Tasks 3, 4, 6, 7; `CameraFit`, `fit_camera`, `project` (Task 4) are used in the same form by the route, the world test and the card.
- **Known limits:** the pose fit assumes a straightened picture with the principal point at the centre (the lens tool's output is); moving a camera to another USB port needs a new calibration (Task 0); `top` cannot be fitted for its field of view, by geometry.
