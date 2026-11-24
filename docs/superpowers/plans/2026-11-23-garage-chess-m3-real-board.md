# Garage Chess M3 — Real Board, Corrections and Co-training

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Status:** drafted on 2026-11-23 together with M1. Tasks 1–3 are fully specified and depend only on M1. Task 1 implemented on 2026-11-24 (commit 1336e73); Task 2 implemented on 2026-11-24 (commit a27deb2); Task 3 is not implemented yet. Tasks 4–6 (real evaluation of the sim policy, multi-dataset co-training, the gate) are outlined at the end and get full TDD detail at the start of M3, once M2's `sim/evaluate.py` and training interfaces exist.

**Goal:** The real cell is built to `sim_cell.json`, the operator calibrates the board on Setup, the app reads the real board, records real correction episodes in chess transfer mode with the same overlay and sidecar as the sim data, runs the sim-trained `sim-best` policy on the real transfer plan, and co-trains on sim + real until the real plan reaches ≥ 80 % with fewer than 100 real episodes.

**Architecture:** M1 provides `chess/game.py`, `calibration.py`, `overlay.py`, `vision.py`, `transfers.py` (task string, plan, sidecar) and `sim/`. M3 adds `lelab/board_state.py` and the `/chess/*` routes, extends `transfers.py` with the real recording sampler and preconditions, hooks `record.py` (fixed task, overlay in the robot observation pipeline and in `/camera-feed/top`, transfer in the status, sidecar after save), and adds the Setup board calibration card and the Record transfer mode.

**Tech Stack:** Python 3.12, FastAPI, python-chess, OpenCV, numpy, LeRobot v0.7.0 processors and datasets, React + Vite + vitest.

Spec: `docs/superpowers/specs/2026-11-23-garage-chess-voice-design.md` (sections 5.5, 5.6, 6, 11, milestone M3).

## Global Constraints

- Work on branch `forge/chess`. Push only to `origin` (`https://github.com/Ivo-B/leLab.git`). Do not push `upstream`.
- Do not start M4. No voice, player, Stockfish or game loop.
- Do not start the robot arm in a unit test. `lelab/chess/` never imports `lelab.sim`, `lerobot.robots`, `lelab.record`, `lelab.forge` or `openai`.
- Task string, markers and the sidecar format are M1's; do not change them, or sim and real datasets stop mixing.
- The camera named `top` must be configured with rotation `0` for chess transfer mode. Refuse to start otherwise.
- Board calibration lives at `~/.cache/huggingface/lerobot/garage/board_calibration.json`; read paths through `lelab.utils.config` at call time.
- `WANDB_API_KEY` and `CWSANDBOX_API_KEY` never appear in a response, a log line, or a logger call.
- New Python modules start with the Apache-2.0 header in `tests/test_forge_config.py` lines 1–13, then a blank line.
- Run Python tests with `uv run pytest`, frontend tests with `npm test -- --run` in `frontend/`, lint with `uv run ruff check` and `uv run ruff format --check`.
- Do not commit pre-existing uncommitted edits. Stage files by name; use `git add -p` for `lelab/server.py` and `lelab/record.py` when they carry unrelated hunks.
- Commit only the files the task lists. Do not skip hooks.

Checked on 2026-11-23: `record_loop` applies `robot_observation_processor` to every observation before `build_dataset_frame`, so a step appended there reaches the recorded video. `LeRobotDatasetMetadata.save_episode` increments `info.total_episodes`, so `dataset.num_episodes - 1` after `save_episode()` is the index just saved. `RobotProcessorPipeline`, `observation_to_transition`, `transition_to_observation` and `RobotObservation` import from `lerobot.processor`.

Before starting, run the spec's M3 start checks: measure the real `top` and `side` camera poses and field of view into `sim_cell.json`; build the board, bin and parking to the M1 cell; check that the follower's calibrated zero matches the sim joint zero (put the real arm in `HOME` and compare with a sim render).

---

### Task 1: Calibration and board-state API

**Files:**

- Create: `lelab/board_state.py`
- Modify: `lelab/server.py` (imports; three routes next to the camera-reference routes)
- Create: `tests/test_chess_api.py`

**Interfaces:**

- `GET /chess/board-calibration` → calibration dict, or 404 `{"status": "error", "message": "No board calibration saved"}`.
- `PUT /chess/board-calibration` (JSON body = `BoardCalibration.to_dict()` shape) → `{"status": "success"}`, 400 on `CalibrationError`.
- `POST /chess/board-state` (body = PNG of a `top` frame) → `{"robot", "squares": {64}, "extras": {3}}`; 409 when calibration or the `top` reference is missing; 400 when the body is not a decodable image.
- `lelab/board_state.py`: `BoardStateError(message, status_code)`, `decode_png(data) -> np.ndarray`, `read_board_state(png: bytes) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chess_api.py  (Apache header above)
import cv2
import pytest
from fastapi.testclient import TestClient

from lelab import camera_reference
from lelab.chess.game import GameState
from lelab.server import app
from tests.chess_render import calibration, empty_board, render


def png(frame) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()


@pytest.fixture
def client(tmp_lerobot_home) -> TestClient:
    return TestClient(app)


def test_calibration_round_trip_and_errors(client: TestClient) -> None:
    assert client.get("/chess/board-calibration").status_code == 404
    body = calibration().to_dict()
    bad = client.put("/chess/board-calibration", json={**body, "bin": []})
    assert bad.status_code == 400
    assert client.put("/chess/board-calibration", json=body).json() == {"status": "success"}
    assert client.get("/chess/board-calibration").json() == body


def test_board_state_needs_calibration_and_reference(client: TestClient) -> None:
    cal = calibration()
    frame = png(render(cal, GameState().occupancy()))
    missing = client.post("/chess/board-state", content=frame, headers={"Content-Type": "image/png"})
    assert missing.status_code == 409
    client.put("/chess/board-calibration", json=cal.to_dict())
    no_reference = client.post("/chess/board-state", content=frame, headers={"Content-Type": "image/png"})
    assert no_reference.status_code == 409
    assert "reference" in no_reference.json()["message"].lower()


def test_board_state_reads_the_position(client: TestClient) -> None:
    cal = calibration()
    client.put("/chess/board-calibration", json=cal.to_dict())
    camera_reference.save_reference("arm", "top", png(empty_board(cal)))
    frame = png(render(cal, {**GameState().occupancy(), "bin": "black"}))
    response = client.post("/chess/board-state", content=frame, headers={"Content-Type": "image/png"})
    assert response.status_code == 200
    body = response.json()
    assert body["robot"] == "arm"
    assert body["squares"] == GameState().occupancy()
    assert body["extras"] == {"bin": "black", "park_white": "empty", "park_black": "empty"}
    garbage = client.post("/chess/board-state", content=b"nope", headers={"Content-Type": "image/png"})
    assert garbage.status_code == 400
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_chess_api.py -q`
Expected: 404s where 200/400/409 are expected.

- [ ] **Step 3: Implement**

```python
# lelab/board_state.py  (Apache header above)
"""Read the chess position from one `top` frame using the saved calibration and
the `top` camera reference as the empty board."""

from __future__ import annotations

import cv2
import numpy as np

from . import camera_reference
from .chess.calibration import load_calibration
from .chess.game import EXTRA_LOCATIONS
from .chess.vision import classify_locations

TOP_CAMERA = "top"


class BoardStateError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def decode_png(data: bytes) -> np.ndarray:
    """PNG/JPEG bytes -> RGB uint8 array."""
    array = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(array, cv2.IMREAD_COLOR) if array.size else None
    if bgr is None:
        raise BoardStateError("Body must be a decodable PNG image", 400)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_board_state(png: bytes) -> dict:
    calibration = load_calibration()
    if calibration is None:
        raise BoardStateError("Calibrate the board on the Setup page first", 409)
    reference_png = camera_reference.load_reference(calibration.robot, TOP_CAMERA)
    if reference_png is None:
        raise BoardStateError(
            f"Capture the '{TOP_CAMERA}' camera reference for robot '{calibration.robot}' first", 409
        )
    frame = decode_png(png)
    reference = decode_png(reference_png)
    occupancy = classify_locations(frame, reference, calibration)
    return {
        "robot": calibration.robot,
        "squares": {k: v for k, v in occupancy.items() if k not in EXTRA_LOCATIONS},
        "extras": {k: occupancy[k] for k in EXTRA_LOCATIONS},
    }
```

In `lelab/server.py`, next to `from . import camera_reference` (or wherever `camera_reference` is imported) add:

```python
from . import board_state
from .chess.calibration import BoardCalibration, CalibrationError, load_calibration, save_calibration
```

After the `delete_camera_reference` route add:

```python
@app.get("/chess/board-calibration")
def get_board_calibration():
    """The saved board calibration (corners, bin, parking squares, thresholds)."""
    try:
        calibration = load_calibration()
    except CalibrationError as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    if calibration is None:
        return JSONResponse(status_code=404, content={"status": "error", "message": "No board calibration saved"})
    return calibration.to_dict()


@app.put("/chess/board-calibration")
async def put_board_calibration(request: Request):
    try:
        save_calibration(BoardCalibration.from_dict(await request.json()))
    except (CalibrationError, ValueError) as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})
    return {"status": "success"}


@app.post("/chess/board-state")
async def post_board_state(request: Request):
    """Classify every square, the bin and the parking squares in the posted PNG."""
    try:
        return board_state.read_board_state(await request.body())
    except board_state.BoardStateError as e:
        return JSONResponse(status_code=e.status_code, content={"status": "error", "message": str(e)})
```

- [ ] **Step 4: Run it**

Run: `uv run pytest tests/test_chess_api.py tests/test_camera_reference.py tests/test_server.py -q`
Expected: pass. A malformed JSON body raises `json.JSONDecodeError`, a `ValueError` subclass, hence the `(CalibrationError, ValueError)` tuple.

- [ ] **Step 5: Commit**

```bash
git add lelab/board_state.py tests/test_chess_api.py
git add -p lelab/server.py   # stage only the board-state import and the three routes
git commit -m "feat: board calibration and board-state endpoints"
```

---

### Task 2: Transfer sampling and the recording hooks

**Files:**

- Modify: `lelab/chess/transfers.py` (M1 created it with `CHESS_TASK`, `TRANSFER_PLAN`, `transfer_class` and the sidecar)
- Modify: `tests/test_chess_transfers.py` (append tests)
- Modify: `lelab/record.py`
- Modify: `tests/test_record.py` (append tests)

**Interfaces:**

- `lelab/chess/transfers.py` gains `RANK_BANDS`, `TransferSampler(rng=None).sample() -> Transfer` and `chess_mode_problem(cameras: dict) -> str | None`. It keeps M1's `CHESS_TASK`, `write_transfer_sidecar` and `read_transfer_sidecar` unchanged, so real and sim datasets carry the same sidecar.
- `lelab/record.py`: `RecordingRequest.chess_transfers: bool = False`; module globals `current_transfer`, `_overlay_step`, `_transfer_sampler`; status gains `chess_transfers` and `transfer`; `/camera-feed/top` shows the overlay; `meta/garage_transfers.json` written after each saved episode.

- [ ] **Step 1: Write the failing sampler and precondition tests**

Merge these imports into the top of the existing `tests/test_chess_transfers.py` and append the tests:

```python
import random

from lelab.chess.calibration import save_calibration
from lelab.chess.game import PARK_BLACK, PARK_WHITE
from lelab.chess.transfers import TransferSampler, chess_mode_problem
from tests.chess_render import calibration


def test_sampler_mix_and_validity() -> None:
    sampler = TransferSampler(random.Random(7))
    kinds: Counter[str] = Counter()
    for _ in range(3000):
        source, target = sampler.sample()
        assert source != target
        if target == BIN:
            kinds["bin"] += 1
            assert source in chess.SQUARE_NAMES
        elif source in (PARK_WHITE, PARK_BLACK):
            kinds["park"] += 1
            assert target in chess.SQUARE_NAMES
        else:
            kinds["board"] += 1
            assert source in chess.SQUARE_NAMES and target in chess.SQUARE_NAMES
    assert abs(kinds["board"] / 3000 - 0.70) < 0.04
    assert abs(kinds["bin"] / 3000 - 0.20) < 0.04
    assert abs(kinds["park"] / 3000 - 0.10) < 0.04


def test_sampler_cycles_source_rank_bands() -> None:
    sampler = TransferSampler(random.Random(1))
    bands = []
    while len(bands) < 30:
        source, target = sampler.sample()
        if target != BIN and source not in (PARK_WHITE, PARK_BLACK):
            bands.append(int(source[1]))
    near = [r for r in bands if r <= 3]
    mid = [r for r in bands if r in (4, 5)]
    far = [r for r in bands if r >= 6]
    assert near and mid and far


def test_chess_mode_problem_checks_top_camera_rotation_and_calibration(tmp_lerobot_home) -> None:
    assert "top" in chess_mode_problem({"wrist": {"rotation": 0}})
    assert "rotation" in chess_mode_problem({"top": {"rotation": 90}})
    assert "Calibrate" in chess_mode_problem({"top": {"rotation": 0}})
    save_calibration(calibration())
    assert chess_mode_problem({"top": {"rotation": 0}, "wrist": {"rotation": 180}}) is None
    assert chess_mode_problem({"top": {}}) is None
```

Append to `tests/test_record.py`:

```python
def test_recording_request_chess_transfers_defaults_off() -> None:
    from lelab.record import RecordingRequest

    request = RecordingRequest(
        leader_port="/dev/l",
        follower_port="/dev/f",
        leader_config="l",
        follower_config="f",
        dataset_repo_id="u/d",
        single_task="x",
    )
    assert request.chess_transfers is False


def test_start_recording_in_chess_mode_refuses_without_top_camera(tmp_lerobot_home) -> None:
    from lelab import record
    from lelab.record import RecordingRequest, handle_start_recording

    request = RecordingRequest(
        leader_port="/dev/l",
        follower_port="/dev/f",
        leader_config="l",
        follower_config="f",
        dataset_repo_id="u/d",
        single_task="whatever",
        chess_transfers=True,
        cameras={"wrist": {"type": "opencv", "camera_index": 0, "rotation": 0}},
        test_mode=True,
    )
    result = handle_start_recording(request)
    assert result["success"] is False
    assert "top" in result["message"]
    assert record.recording_active is False
    assert record.recording_thread is None or not record.recording_thread.is_alive()


def test_recording_status_reports_the_current_transfer(monkeypatch: pytest.MonkeyPatch) -> None:
    from lelab import record
    from lelab.record import RecordingRequest, handle_recording_status

    request = RecordingRequest(
        leader_port="/dev/l",
        follower_port="/dev/f",
        leader_config="l",
        follower_config="f",
        dataset_repo_id="u/d",
        single_task="x",
        chess_transfers=True,
        cameras={"top": {"type": "opencv", "camera_index": 0, "rotation": 0}},
    )
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "recording_config", request)
    monkeypatch.setattr(record, "current_transfer", ("e2", "e4"))
    status = handle_recording_status()
    assert status["chess_transfers"] is True
    assert status["transfer"] == {"source": "e2", "target": "e4"}
    monkeypatch.setattr(record, "current_transfer", None)
    assert handle_recording_status()["transfer"] is None
```

If `tests/test_record.py` does not already `import pytest`, add it.

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/test_chess_transfers.py tests/test_record.py -q`
Expected: `ImportError` for `TransferSampler`; the three new record tests fail on the missing field/keys.

- [ ] **Step 3: Implement the sampler and the preconditions**

In `lelab/chess/transfers.py`, extend the docstring to mention the sampler and chess-mode checks, add `import random`, add `from .calibration import load_calibration`, then add after `SIDECAR_NAME`:

```python
TOP_CAMERA = "top"

# Ranks grouped by distance from the arm, which sits behind rank 1.
RANK_BANDS = ("123", "45", "678")
FILES = "abcdefgh"
BOARD_SHARE = 0.70
BIN_SHARE = 0.20  # the remaining 0.10 is parking -> board


class TransferSampler:
    """70 % board-to-board (source rank band cycles near/mid/far), 20 % board -> bin,
    10 % parking -> board."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()
        self._band = 0

    def _square(self, ranks: str = "12345678") -> str:
        return self._rng.choice(FILES) + self._rng.choice(ranks)

    def sample(self) -> Transfer:
        roll = self._rng.random()
        if roll < BOARD_SHARE:
            source = self._square(RANK_BANDS[self._band])
            self._band = (self._band + 1) % len(RANK_BANDS)
            target = self._square()
            while target == source:
                target = self._square()
            return (source, target)
        if roll < BOARD_SHARE + BIN_SHARE:
            return (self._square(), BIN)
        return (self._rng.choice((PARK_WHITE, PARK_BLACK)), self._square())
```

and at the end of the module:

```python
def chess_mode_problem(cameras: dict[str, dict]) -> str | None:
    """Why chess transfer mode cannot start, or None."""
    top = cameras.get(TOP_CAMERA)
    if top is None:
        return f"Chess transfer mode needs a camera named '{TOP_CAMERA}'."
    if top.get("rotation", 0) != 0:
        return (
            f"Chess transfer mode needs the '{TOP_CAMERA}' camera at rotation 0, "
            "so the board calibration lines up with recorded frames."
        )
    if load_calibration() is None:
        return "Calibrate the board on the Setup page before recording chess transfers."
    return None
```

`lelab/sim/positions.py` has its own sampler over a simulated position; this one only picks squares, because on the real board the operator sets up whatever piece the markers ask for.



- [ ] **Step 4: Hook `record.py`**

1. Imports (top of `lelab/record.py`, with the other `lelab` imports):

```python
from .chess.overlay import GoalOverlayStep
from .chess.transfers import CHESS_TASK, TransferSampler, chess_mode_problem, write_transfer_sidecar
```

2. `RecordingRequest`: after `test_mode: bool = False  # Skip robot connection for testing` add

```python
    # Chess transfer mode: fixed task string, red/green goal markers painted into
    # the `top` stream, and a `meta/garage_transfers.json` sidecar.
    chess_transfers: bool = False
```

3. Module globals, after `current_robot = None`:

```python
# Chess transfer mode (see lelab/chess/transfers.py)
current_transfer: tuple[str, str] | None = None
_overlay_step: GoalOverlayStep | None = None
_transfer_sampler: TransferSampler | None = None
```

4. `handle_start_recording`: add `current_transfer` to the `global` list. After the four mutex checks (`Replay is currently active...`) and before `recording_config = None` (or the first state reset), insert:

```python
        if request.chess_transfers:
            problem = chess_mode_problem(request.cameras)
            if problem is not None:
                return {"success": False, "message": problem}
            request.single_task = CHESS_TASK
        current_transfer = None
```

5. `record_with_web_events`: add `_overlay_step`, `_transfer_sampler` and `current_transfer` to the `global` statement at the top of the function (the one that already lists `current_phase`, `current_robot`, …). Then, after `teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()`, insert:

```python
    _overlay_step = None
    _transfer_sampler = None
    if recording_config is not None and recording_config.chess_transfers:
        from lerobot.processor import (
            RobotObservation,
            RobotProcessorPipeline,
            observation_to_transition,
            transition_to_observation,
        )

        from .chess.calibration import load_calibration

        calibration = load_calibration()
        if calibration is None:
            raise RuntimeError("Board calibration missing; calibrate on the Setup page first")
        _overlay_step = GoalOverlayStep(calibration=calibration)
        _transfer_sampler = TransferSampler()
        robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
            steps=[*robot_observation_processor.steps, _overlay_step],
            to_transition=observation_to_transition,
            to_output=transition_to_observation,
        )
```

Add this module-level helper above `record_with_web_events`:

```python
def _ensure_transfer() -> None:
    """Pick the next transfer once per episode and show it in the overlay.

    Called at the start of every reset and recording phase: the operator sees
    where to place the piece during reset, and a re-recorded episode keeps its
    transfer because it is only cleared after a successful save.
    """
    global current_transfer
    if _overlay_step is None or _transfer_sampler is None:
        return
    if current_transfer is None:
        current_transfer = _transfer_sampler.sample()
        logger.info("♟️ Next transfer: %s -> %s", current_transfer[0], current_transfer[1])
    _overlay_step.set_goal(*current_transfer)
```

Call `_ensure_transfer()` on the line after each of the three `current_phase = "resetting"` assignments and after `current_phase = "recording"` inside the episode loop.

After `saved_episodes += 1` (right after `dataset.save_episode()`) insert:

```python
            if _overlay_step is not None and current_transfer is not None:
                write_transfer_sidecar(Path(dataset.root), dataset.num_episodes - 1, current_transfer)
                current_transfer = None
                _overlay_step.clear_goal()
```

In the teardown where `current_robot = None` is set (line ~1100, same function), add `_overlay_step = None` and `current_transfer = None` next to it.

6. `handle_recording_status`: inside `if recording_active and recording_config:` after `status["cameras"] = ...` add

```python
        if recording_config.chess_transfers:
            status["chess_transfers"] = True
            status["transfer"] = (
                {"source": current_transfer[0], "target": current_transfer[1]} if current_transfer else None
            )
```

7. `camera_feed_frames`: after the `frame = cam.read_latest()` try/except block add

```python
        overlay = _overlay_step
        if cam_key == "top" and overlay is not None:
            frame = overlay.observation({cam_key: frame})[cam_key]
```

- [ ] **Step 5: Run everything**

Run: `uv run pytest -q`
Expected: pass. `test_start_recording_in_chess_mode_refuses_without_top_camera` must return before any thread starts, so `recording_config` stays `None`.

Run: `uv run ruff check lelab tests && uv run ruff format --check lelab tests`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add lelab/chess/transfers.py tests/test_chess_transfers.py lelab/record.py tests/test_record.py
git commit -m "feat: chess transfer recording mode with goal overlay and sidecar"
```

---

### Task 3: Frontend — board calibration on Setup, transfer mode on Record

**Files:**

- Create: `frontend/src/lib/boardCalibrationApi.ts`, `frontend/src/lib/boardCalibrationApi.test.ts`
- Create: `frontend/src/components/setup/BoardCalibrationCard.tsx`, `frontend/src/components/setup/BoardCalibrationCard.test.tsx`
- Modify: `frontend/src/pages/Setup.tsx`
- Modify: `frontend/src/hooks/useRecording.ts`, `frontend/src/components/landing/RecordingModal.tsx`, `frontend/src/pages/Recording.tsx`

**Interfaces:**

- `boardCalibrationApi.ts`: `Point = [number, number]`, `BoardCalibration`, `BoardState`, `CHESS_TASK`, `getBoardCalibration(baseUrl, fetcher)`, `saveBoardCalibration(baseUrl, fetcher, cal)`, `readBoardState(baseUrl, fetcher, png: Blob)`, `imagePoint(img, clientX, clientY): Point` (client → natural pixel coordinates).
- `BoardCalibrationCard({ robot, topDeviceId })`: renders the `top` reference, collects 16 clicks in order (board a1, h1, h8, a8; bin ×4; park_white ×4; park_black ×4), Save, Reset, "Check position" that grabs one frame from `topDeviceId` via `getUserMedia`, posts it and shows the 8×8 grid.

- [ ] **Step 1: API module test**

```ts
// frontend/src/lib/boardCalibrationApi.test.ts
import { describe, expect, it, vi } from "vitest";
import { getBoardCalibration, imagePoint, readBoardState, saveBoardCalibration } from "./boardCalibrationApi";

const BASE = "http://localhost:8000";

describe("boardCalibrationApi", () => {
  it("returns null on 404 and the body otherwise", async () => {
    const fetcher = vi.fn().mockResolvedValueOnce({ ok: false, status: 404 } as Response);
    expect(await getBoardCalibration(BASE, fetcher)).toBeNull();
    const body = { robot: "arm", image_size: [4, 4], board_corners: [], bin: [], park_white: [], park_black: [] };
    fetcher.mockResolvedValueOnce({ ok: true, status: 200, json: () => Promise.resolve(body) } as Response);
    expect(await getBoardCalibration(BASE, fetcher)).toEqual(body);
  });

  it("PUTs JSON and POSTs a PNG", async () => {
    const fetcher = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ status: "success" }),
    } as Response);
    const cal = { robot: "arm", image_size: [4, 4] as [number, number], board_corners: [], bin: [], park_white: [], park_black: [] };
    await saveBoardCalibration(BASE, fetcher, cal);
    expect(fetcher).toHaveBeenLastCalledWith(
      `${BASE}/chess/board-calibration`,
      expect.objectContaining({ method: "PUT", body: JSON.stringify(cal) }),
    );
    await readBoardState(BASE, fetcher, new Blob(["x"], { type: "image/png" }));
    const [url, init] = fetcher.mock.lastCall as [string, RequestInit];
    expect(url).toBe(`${BASE}/chess/board-state`);
    expect(init.method).toBe("POST");
    expect((init.headers as Record<string, string>)["Content-Type"]).toBe("image/png");
  });

  it("maps client coordinates onto natural pixels", () => {
    const img = {
      naturalWidth: 640,
      naturalHeight: 480,
      getBoundingClientRect: () => ({ left: 10, top: 20, width: 320, height: 240 }),
    } as unknown as HTMLImageElement;
    expect(imagePoint(img, 170, 140)).toEqual([320, 240]);
  });
});
```

- [ ] **Step 2: Run it**

Run: `cd frontend && npm test -- --run src/lib/boardCalibrationApi.test.ts`
Expected: cannot resolve `./boardCalibrationApi`.

- [ ] **Step 3: Implement the API module**

```ts
// frontend/src/lib/boardCalibrationApi.ts
import { ApiError, Fetcher } from "./apiClient";

export type Point = [number, number];

export interface BoardCalibration {
  robot: string;
  image_size: [number, number];
  board_corners: Point[];
  bin: Point[];
  park_white: Point[];
  park_black: Point[];
  pixel_diff_threshold?: number;
  occupied_fraction?: number;
  colour_threshold?: number;
}

export type Occupancy = "empty" | "white" | "black";

export interface BoardState {
  robot: string;
  squares: Record<string, Occupancy>;
  extras: Record<string, Occupancy>;
}

export const CHESS_TASK = "move the piece on the red square to the green square";

async function fail(action: string, r: Response): Promise<never> {
  let message = `${action} failed: ${r.status}`;
  try {
    const body = (await r.json()) as { message?: string };
    if (body?.message) message = body.message;
  } catch {
    /* no JSON body */
  }
  throw new ApiError(message, r.status, null);
}

export async function getBoardCalibration(baseUrl: string, fetcher: Fetcher): Promise<BoardCalibration | null> {
  const r = await fetcher(`${baseUrl}/chess/board-calibration`);
  if (r.status === 404) return null;
  if (!r.ok) return fail("Load board calibration", r);
  return (await r.json()) as BoardCalibration;
}

export async function saveBoardCalibration(baseUrl: string, fetcher: Fetcher, cal: BoardCalibration): Promise<void> {
  const r = await fetcher(`${baseUrl}/chess/board-calibration`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cal),
  });
  if (!r.ok) await fail("Save board calibration", r);
}

export async function readBoardState(baseUrl: string, fetcher: Fetcher, png: Blob): Promise<BoardState> {
  const r = await fetcher(`${baseUrl}/chess/board-state`, {
    method: "POST",
    headers: { "Content-Type": "image/png" },
    body: png,
  });
  if (!r.ok) return fail("Read board", r);
  return (await r.json()) as BoardState;
}

/** Click position on a displayed <img> expressed in the image's natural pixels. */
export function imagePoint(img: HTMLImageElement, clientX: number, clientY: number): Point {
  const rect = img.getBoundingClientRect();
  const x = ((clientX - rect.left) * img.naturalWidth) / rect.width;
  const y = ((clientY - rect.top) * img.naturalHeight) / rect.height;
  return [Math.round(x * 10) / 10, Math.round(y * 10) / 10];
}
```

Check `ApiError`'s constructor signature in `frontend/src/lib/apiClient.ts` and match it.

- [ ] **Step 4: Run it**

Run: `cd frontend && npm test -- --run src/lib/boardCalibrationApi.test.ts`
Expected: pass.

- [ ] **Step 5: Card test**

```tsx
// frontend/src/components/setup/BoardCalibrationCard.test.tsx
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiProvider } from "@/contexts/ApiContext";
import BoardCalibrationCard from "./BoardCalibrationCard";

const BASE = "http://localhost:8000";
const PNG = new Blob([new Uint8Array([137, 80, 78, 71])], { type: "image/png" });

function stubFetch(calibration: unknown | null) {
  const fetchMock = vi.fn((url: string, init?: RequestInit) => {
    if (url === `${BASE}/robots/arm/camera-reference/top`) {
      return Promise.resolve({ ok: true, status: 200, blob: () => Promise.resolve(PNG) } as Response);
    }
    if (url === `${BASE}/chess/board-calibration` && (!init || !init.method || init.method === "GET")) {
      return calibration
        ? Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(calibration) } as Response)
        : Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve({}) } as Response);
    }
    if (url === `${BASE}/chess/board-calibration` && init?.method === "PUT") {
      return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ status: "success" }) } as Response);
    }
    return Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve({}) } as Response);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function mountImage(img: HTMLImageElement) {
  Object.defineProperty(img, "naturalWidth", { value: 640, configurable: true });
  Object.defineProperty(img, "naturalHeight", { value: 480, configurable: true });
  img.getBoundingClientRect = () => ({ left: 0, top: 0, width: 640, height: 480, right: 640, bottom: 480, x: 0, y: 0, toJSON: () => ({}) });
}

beforeEach(() => {
  // jsdom has no object URLs; the card only needs a string for <img src>.
  URL.createObjectURL = vi.fn(() => "blob:ref");
  URL.revokeObjectURL = vi.fn();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("BoardCalibrationCard", () => {
  it("collects sixteen clicks in order and saves them", async () => {
    const fetchMock = stubFetch(null);
    render(
      <ApiProvider>
        <BoardCalibrationCard robot="arm" topDeviceId={null} />
      </ApiProvider>,
    );
    const img = (await screen.findByAltText("top camera reference")) as HTMLImageElement;
    mountImage(img);
    expect(screen.getByText(/Click a1/)).toBeInTheDocument();
    const save = screen.getByRole("button", { name: "Save calibration" });
    expect(save).toBeDisabled();

    const points: Array<[number, number]> = [];
    for (let i = 0; i < 16; i++) points.push([20 + i * 30, 40 + (i % 4) * 50]);
    for (const [x, y] of points) fireEvent.click(img, { clientX: x, clientY: y });

    expect(screen.getByText(/All 16 points set/)).toBeInTheDocument();
    expect(save).toBeEnabled();
    fireEvent.click(save);

    await waitFor(() => expect(screen.getByText(/Saved/)).toBeInTheDocument());
    const put = fetchMock.mock.calls.find(([, init]) => (init as RequestInit)?.method === "PUT");
    expect(put).toBeDefined();
    const body = JSON.parse(String((put![1] as RequestInit).body));
    expect(body.robot).toBe("arm");
    expect(body.image_size).toEqual([640, 480]);
    expect(body.board_corners).toEqual(points.slice(0, 4));
    expect(body.bin).toEqual(points.slice(4, 8));
    expect(body.park_white).toEqual(points.slice(8, 12));
    expect(body.park_black).toEqual(points.slice(12, 16));
  });

  it("explains when the top reference is missing", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve({}) } as Response)),
    );
    render(
      <ApiProvider>
        <BoardCalibrationCard robot="arm" topDeviceId={null} />
      </ApiProvider>,
    );
    expect(await screen.findByText(/Capture the top camera reference/)).toBeInTheDocument();
  });
});
```

- [ ] **Step 6: Run it**

Run: `cd frontend && npm test -- --run src/components/setup/BoardCalibrationCard.test.tsx`
Expected: cannot resolve `./BoardCalibrationCard`.

- [ ] **Step 7: Implement the card and wire Setup**

```tsx
// frontend/src/components/setup/BoardCalibrationCard.tsx
import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { useApi } from "@/contexts/ApiContext";
import { useCameraReference } from "@/hooks/useCameraReference";
import {
  BoardCalibration,
  BoardState,
  Point,
  getBoardCalibration,
  imagePoint,
  readBoardState,
  saveBoardCalibration,
} from "@/lib/boardCalibrationApi";
import { captureVideoFrame } from "@/lib/cameraReferenceApi";

type Group = "board_corners" | "bin" | "park_white" | "park_black";

const STEPS: Array<{ group: Group; label: string; hints: string[] }> = [
  { group: "board_corners", label: "Board", hints: ["a1", "h1", "h8", "a8"] },
  { group: "bin", label: "Bin", hints: ["corner 1", "corner 2", "corner 3", "corner 4"] },
  { group: "park_white", label: "White parking", hints: ["corner 1", "corner 2", "corner 3", "corner 4"] },
  { group: "park_black", label: "Black parking", hints: ["corner 1", "corner 2", "corner 3", "corner 4"] },
];
const TOTAL = 16;
const COLOURS: Record<Group, string> = {
  board_corners: "#f59e0b",
  bin: "#ef4444",
  park_white: "#e5e7eb",
  park_black: "#1f2937",
};
const FILES = "abcdefgh";

function flatten(groups: Record<Group, Point[]>): Array<{ group: Group; point: Point }> {
  return STEPS.flatMap(({ group }) => groups[group].map((point) => ({ group, point })));
}

function emptyGroups(): Record<Group, Point[]> {
  return { board_corners: [], bin: [], park_white: [], park_black: [] };
}

export default function BoardCalibrationCard({ robot, topDeviceId }: { robot: string | undefined; topDeviceId: string | null }) {
  const { baseUrl, fetchWithHeaders } = useApi();
  const reference = useCameraReference(robot, "top");
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [groups, setGroups] = useState<Record<Group, Point[]>>(emptyGroups);
  const [existing, setExisting] = useState<BoardCalibration | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [state, setState] = useState<BoardState | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getBoardCalibration(baseUrl, fetchWithHeaders)
      .then((cal) => {
        if (!cancelled) setExisting(cal);
      })
      .catch((e) => {
        if (!cancelled) setMessage(String(e.message ?? e));
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders]);

  const placed = flatten(groups);
  const current = STEPS.find(({ group }) => groups[group].length < 4);
  const complete = placed.length === TOTAL;

  const onImageClick = useCallback(
    (e: React.MouseEvent<HTMLImageElement>) => {
      if (!current || !imgRef.current) return;
      const point = imagePoint(imgRef.current, e.clientX, e.clientY);
      setGroups((prev) => ({ ...prev, [current.group]: [...prev[current.group], point] }));
      setMessage(null);
    },
    [current],
  );

  const save = async () => {
    const img = imgRef.current;
    if (!robot || !img || !complete) return;
    setBusy(true);
    try {
      const cal: BoardCalibration = {
        robot,
        image_size: [img.naturalWidth, img.naturalHeight],
        ...groups,
        ...(existing
          ? {
              pixel_diff_threshold: existing.pixel_diff_threshold,
              occupied_fraction: existing.occupied_fraction,
              colour_threshold: existing.colour_threshold,
            }
          : {}),
      };
      await saveBoardCalibration(baseUrl, fetchWithHeaders, cal);
      setExisting(cal);
      setMessage("Saved. Place the pieces and check the position below.");
    } catch (e) {
      setMessage(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const checkPosition = async () => {
    if (!topDeviceId) {
      setMessage("Bind the top camera on the Calibration page to check the position live.");
      return;
    }
    setBusy(true);
    let stream: MediaStream | null = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: topDeviceId } } });
      const video = document.createElement("video");
      video.srcObject = stream;
      video.muted = true;
      await video.play();
      await new Promise((resolve) => setTimeout(resolve, 300));
      const png = await captureVideoFrame(video);
      if (!png) throw new Error("Could not read a frame from the top camera");
      setState(await readBoardState(baseUrl, fetchWithHeaders, png));
      setMessage(null);
    } catch (e) {
      setMessage(e instanceof Error ? e.message : String(e));
    } finally {
      stream?.getTracks().forEach((track) => track.stop());
      setBusy(false);
    }
  };

  if (!robot) {
    return <p className="text-sm text-muted-foreground">Select a robot on Home first.</p>;
  }
  if (reference.error) {
    return <p className="text-sm text-destructive">{reference.error}</p>;
  }
  if (!reference.url) {
    return (
      <p className="text-sm text-muted-foreground">
        Capture the top camera reference on the Calibration page with the board empty, then come back here.
      </p>
    );
  }

  const prompt = current
    ? `Click ${current.label === "Board" ? "" : `${current.label.toLowerCase()} `}${current.hints[groups[current.group].length]}`
    : `All ${TOTAL} points set.`;

  return (
    <div className="flex flex-col gap-3">
      <p className="text-sm text-muted-foreground">
        {existing ? "A calibration is saved. Click again to replace it. " : ""}
        Order: board corners a1, h1, h8, a8; then the bin; then the white and black parking squares.
      </p>
      <div className="relative inline-block self-start">
        <img
          ref={imgRef}
          src={reference.url}
          alt="top camera reference"
          onClick={onImageClick}
          className="max-w-full cursor-crosshair rounded-md border border-border"
        />
        {placed.map(({ group, point }, i) => {
          const img = imgRef.current;
          const sx = img && img.naturalWidth ? img.clientWidth / img.naturalWidth : 1;
          const sy = img && img.naturalHeight ? img.clientHeight / img.naturalHeight : 1;
          return (
            <span
              key={i}
              className="pointer-events-none absolute h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border border-white"
              style={{ left: point[0] * sx, top: point[1] * sy, background: COLOURS[group] }}
            />
          );
        })}
      </div>
      <p className="text-sm font-medium">{prompt}</p>
      <div className="flex flex-wrap gap-2">
        <Button onClick={save} disabled={!complete || busy}>
          Save calibration
        </Button>
        <Button variant="outline" onClick={() => setGroups(emptyGroups())} disabled={placed.length === 0 || busy}>
          Reset points
        </Button>
        <Button variant="outline" onClick={checkPosition} disabled={!existing || busy}>
          Check position
        </Button>
      </div>
      {message ? <p className="text-sm text-muted-foreground">{message}</p> : null}
      {state ? (
        <div className="grid w-64 grid-cols-8 gap-px rounded-md border border-border bg-border p-px">
          {[8, 7, 6, 5, 4, 3, 2, 1].flatMap((rank) =>
            [...FILES].map((file) => {
              const name = `${file}${rank}`;
              const value = state.squares[name];
              const bg = value === "white" ? "bg-gray-100" : value === "black" ? "bg-gray-800" : "bg-background";
              return <div key={name} title={`${name}: ${value}`} className={`aspect-square ${bg}`} />;
            }),
          )}
        </div>
      ) : null}
    </div>
  );
}
```

In `frontend/src/pages/Setup.tsx`: import `BoardCalibrationCard` and add a list item after the "Calibrate arms and cameras" item:

```tsx
        <li className="rounded-md border border-border p-4">
          <span className="font-bold">Board calibration</span>
          <span className="mt-1 mb-3 block text-sm text-muted-foreground">
            Mark the board, bin and parking squares on the top camera, then verify the detected position.
          </span>
          <BoardCalibrationCard robot={robotName ?? undefined} topDeviceId={topDeviceId} />
        </li>
```

where `topDeviceId` comes from the selected robot's cameras. `useRobots()` returns `records` (a `Record<string, RobotRecord>` keyed by robot name; each record has `cameras: CameraConfig[]` with `name` and `device_id`):

```ts
  const { records, selectedName, availableNames, isLoading } = useRobots();
  const topDeviceId =
    (robotName ? records[robotName] : undefined)?.cameras.find((c) => c.name === "top")?.device_id ?? null;
```

- [ ] **Step 8: Run it**

Run: `cd frontend && npm test -- --run src/components/setup && npx tsc --noEmit -p tsconfig.app.json`
Expected: pass. If `useCameraReference` fetches `/chess/board-calibration` before the reference and the "missing reference" test sees the wrong text, the 404 stub covers both URLs, so only the ordering of `findByText` matters.

- [ ] **Step 9: Recording transfer mode**

`frontend/src/hooks/useRecording.ts`:

- add `const [chessTransfers, setChessTransfers] = useState(false);` after `singleTask`;
- in `recordingConfig` set `single_task: chessTransfers ? CHESS_TASK : singleTask,` and add `chess_transfers: chessTransfers,`; import `CHESS_TASK` from `@/lib/boardCalibrationApi`;
- pass `chessTransfers, setChessTransfers` through `modalProps`.

`frontend/src/components/landing/RecordingModal.tsx`:

- add `chessTransfers: boolean; setChessTransfers: (value: boolean) => void;` to the props and destructure them;
- above the "Task Description" field add

```tsx
                <div className="flex items-center justify-between rounded-md border border-gray-700 p-3">
                  <div>
                    <Label htmlFor="chessTransfers" className="text-sm font-medium text-gray-300">
                      Chess transfer mode
                    </Label>
                    <p className="text-xs text-gray-400">
                      Fixed task, red/green goal markers on the top camera, one sampled transfer per episode.
                    </p>
                  </div>
                  <Switch id="chessTransfers" checked={chessTransfers} onCheckedChange={setChessTransfers} />
                </div>
```

- give the task `Input` `disabled={chessTransfers}` and `value={chessTransfers ? CHESS_TASK : singleTask}`; import `Switch` from `@/components/ui/switch` and `CHESS_TASK`.

`frontend/src/pages/Recording.tsx`:

- `RecordingConfig` gains `chess_transfers?: boolean;`; `BackendStatus` gains `chess_transfers?: boolean; transfer?: { source: string; target: string } | null;`;
- under the status text (`getStatusText()` usage) render, when `backendStatus.chess_transfers`:

```tsx
            {backendStatus.chess_transfers ? (
              <p className="mt-2 text-lg">
                Transfer:{" "}
                {backendStatus.transfer ? (
                  <>
                    <span className="font-bold text-red-400">{backendStatus.transfer.source}</span>
                    {" → "}
                    <span className="font-bold text-green-400">{backendStatus.transfer.target}</span>
                  </>
                ) : (
                  <span className="text-muted-foreground">sampling…</span>
                )}
              </p>
            ) : null}
```

Place it in the same block as the phase pill so it appears during reset and recording. The `top` tile already shows the markers because `/camera-feed/top` is overlaid server-side.

- [ ] **Step 10: Run everything**

Run: `cd frontend && npm test -- --run && npx tsc --noEmit -p tsconfig.app.json && npm run build`
Expected: pass; the build writes `frontend/dist/` (not committed).

Run: `uv run pytest -q`
Expected: pass.

- [ ] **Step 11: Commit**

```bash
git add frontend/src/lib/boardCalibrationApi.ts frontend/src/lib/boardCalibrationApi.test.ts frontend/src/components/setup/BoardCalibrationCard.tsx frontend/src/components/setup/BoardCalibrationCard.test.tsx frontend/src/pages/Setup.tsx frontend/src/hooks/useRecording.ts frontend/src/components/landing/RecordingModal.tsx frontend/src/pages/Recording.tsx
git commit -m "feat: board calibration on Setup and chess transfer mode on Record"
```

---

## Manual hardware checklist (operator, after Task 3)

Not part of the automated run. Record the outcome in the M3 section of `CHESS_DEMO_GUIDE.md`.

1. The cell matches `sim_cell.json`: square size, board offset from the arm base, bin and parking positions within 5 mm. Pieces match the measured sizes in the cell file.
2. Piece grip: the gripper closes on the widest piece (king) with margin and does not knock neighbours when descending on a crowded centre.
3. `top` camera: rotation 0, sees the whole board, bin and both parking squares with the arm parked; capture the reference on Calibration with the board empty. Its pose and field of view are written into `sim_cell.json` so M2's randomization centres on it.
4. Setup → Board calibration: click 16 points, Save, place the starting position, Check position: 32 squares correct. If white pieces read as black on light squares, raise `colour_threshold` in the JSON; if empty squares read occupied under shadows, raise `pixel_diff_threshold`.
5. Record with Chess transfer mode on, 3 episodes: the banner shows a transfer during reset, markers are visible in the `top` tile, `meta/garage_transfers.json` has three entries, and the recorded `top` video shows the discs when opened in the dataset browser.
6. Push `forge/chess` to `origin`.

## Tasks 4–6, detailed at M3 start

These depend on interfaces M2 creates (`lelab/sim/evaluate.py`, the training config for sim datasets, the `sim-best` Registry alias). Write them as full TDD tasks when M3 starts, from this outline and spec sections 5.5–5.6.

- **Task 4: Real evaluation of `sim-best`.** Run the `sim-best` checkpoint on the 20-transfer plan on the real arm through `lerobot-rollout` with the overlay in its observation pipeline. The operator judges each transfer (success, or the failure class of `lelab/sim/judge.py`: approach, grasp, drop, place, topple, disturbed). Log the run to the same Weave project as sim evaluation (`so101-chess-sim-eval`) with `domain="real"`, so one table compares sim and real per transfer class. Go/no-go: ≥ 50 % real success → corrections only; 20–49 % → corrections plus a look at the randomization ranges; < 20 % → stop and decide between more sim work and the IK fallback. The real evaluation session and Weave rows already exist (cleanup plan Task 3); Task 4 adds the overlay to the rollout and runs `sim-best`.
- **Task 5: Correction recording and co-training.** The operator records real episodes only for the failing transfer classes (chess transfer mode, Task 2), in batches of 10–20 into `so101-chess-real`. Add a `classes` filter to `TransferSampler` (`near`, `mid`, `far`, `bin`, `park`, matching `transfer_class`) and a class selector to the Record page's chess transfer mode. Training mixes `so101-chess-sim` and `so101-chess-real` with real sampled at 30 % of each batch; the run config names both Artifact versions. Check whether LeRobot v0.7.0's training CLI can weight two datasets; if not, add a weighted sampler in Garage's training wrapper and a test that the ratio holds.
- **Task 6: The M3 gate.** Re-run Task 4 after each co-training round. Done when the real plan reaches ≥ 80 % with fewer than 100 real episodes in total; the winning checkpoint becomes `real-best` in the Registry.

## Out of scope for M3

- Voice (M4).
- Player LLM, puzzles, Stockfish analysis and the game loop on `SimBackend` (M5).
- `lelab.chess.rollout_entry`, the inference-time overlay for games and `ArmBackend` (M6).
- Full real game, Agent Lens, README storyline (M7).
