# Garage Chess M1 — Chess Core and the Simulated Cell

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The kitting agent is gone. The app knows chess (moves → transfers, board calibration, board reading, goal overlay), and a MuJoCo simulation of the SO-101 chess cell generates overlay-conditioned transfer demonstrations with a scripted expert that succeeds on ≥ 90 % of the 20-transfer plan. The M1 deliverable is a 1,000-episode sim dataset with the real cell's feature names, a transfer sidecar and generation metadata, logged as a W&B Artifact. No arm moves in a unit test.

**Architecture:** `lelab/chess/` is pure (python-chess, numpy, OpenCV; `overlay.py` also imports `lerobot.processor`). `lelab/sim/` builds an MJCF scene from a position (`scene.py`), attaches MuJoCo Menagerie's SO-101 from a pinned `lerobot/so101-tabletop-mujoco` revision (`assets.py`, `world.py`), steps it in LeRobot's `so101_follower` units, projects the board into the `top` camera to get a `BoardCalibration` (so `overlay.py` and `vision.py` run unchanged on sim frames), judges each transfer from the physics state (`judge.py`), drives it with a damped-least-squares IK expert (`expert.py`), wraps it as a Gymnasium env (`env.py`) and writes LeRobot datasets (`generate.py`). `python -m lelab.sim gate|preview|generate` runs it.

**Tech Stack:** Python 3.12, python-chess, OpenCV, numpy, MuJoCo ≥ 3.3 (4.0.1 checked), Gymnasium, LeRobot v0.7.0 datasets and processors, huggingface_hub, React + Vite + vitest.

Spec: `docs/superpowers/specs/2026-11-23-garage-chess-voice-design.md` (sections 4, 5.1–5.3, 6, 12.2, milestone M1).

## Global Constraints

- Work on branch `forge/chess`. Push only to `origin` (`https://github.com/Ivo-B/leLab.git`). Do not push `upstream`.
- Do not start M2. No domain randomization, policy training, sim evaluation of policies, collection renames, Sandbox or Registry changes. `lelab/forge/registry.py`, `lelab/forge/tracking.py` and `lelab/evaluate.py` keep their kitting strings until M2.
- Do not start the robot arm in a unit test. `lelab/chess/` never imports `lelab.sim`, `lerobot.robots`, `lelab.record`, `lelab.forge` or `openai`. Only `lelab/sim/` imports `mujoco` or `gymnasium`.
- Task string is fixed: `move the piece on the red square to the green square`. Markers are filled discs, diameter 40 % of the square's shorter side, pure red `(255, 0, 0)` for the source and pure green `(0, 255, 0)` for the target, in RGB.
- Sim units match LeRobot's `so101_follower` with `use_degrees=True`: degrees for the five arm joints, 0–100 for the gripper, absolute targets at 30 fps. Sim datasets use the real follower's feature names (`shoulder_pan.pos` … `gripper.pos`, `observation.images.top|wrist|side`) and `robot_type="so101_follower"`.
- The arm model comes only from `lerobot/so101-tabletop-mujoco` at revision `e68145d8bd793c27fa2c13e31369a616895c5acb` (Apache-2.0); do not vendor it into the repo.
- MuJoCo rendering needs a GL context. On macOS it works through CGL **outside** the Cursor sandbox only; run rendering tests and `python -m lelab.sim` with full permissions. Rendering tests skip themselves when no context can be created (CI).
- Paths come from `lelab/utils/config.py` (`GARAGE_CONFIG_PATH`, `BOARD_CALIBRATION_FILE`, `SIM_CELL_FILE`, `SIM_ASSETS_PATH`) and are read through the module at call time, so tests can redirect them.
- `WANDB_API_KEY` and `CWSANDBOX_API_KEY` never appear in a response, a log line, or a logger call.
- New Python modules start with the Apache-2.0 header in `tests/test_forge_config.py` lines 1–13, then a blank line.
- Run Python tests with `uv run pytest`. Run frontend tests with `npm test -- --run` inside `frontend/`. Lint with `uv run ruff check` and `uv run ruff format --check`.
- The working tree already holds uncommitted edits to `frontend/vite.config.ts`, `lelab/scripts/lelab.py`, `lelab/server.py` (CORS `:8080` → `:8081`), `tests/test_scripts_lelab.py` and an untracked `lerobot.code-workspace`. Do not commit them, do not revert them. Stage files by name, use `git add -p` for `lelab/server.py`, never `git add -A` or `git commit -a`.
- Commit only the files the task lists. Do not skip hooks.

Checked on 2026-11-23: python-chess 1.11.2 gives the transfer lists in Task 2. The homography, vision thresholds and overlay pixel counts in Tasks 3–5 were reproduced on synthetic boards (starting position read 64/64). A dataclass `ObservationProcessorStep` runs inside a real `RobotProcessorPipeline`. MuJoCo 4.0.1 installs with `uv`; a chess scene with 32 pieces and the Menagerie SO-101 compiles in 0.06 s; the `gripperframe` site's local x axis is the approach axis; vertical IK reaches ranks 1–7 on a 30 mm board starting 10 cm from the base, a8/h8 miss by 19 mm; a scripted grasp lifts a pawn 9 cm but release toppled it and the elbow knocked pieces on ranks 7–8 — Task 11 tunes these against the gate. LeRobot v0.7.0 has `LeRobotDataset.create(repo_id, fps, features, root, robot_type, use_videos, …)`, `add_frame(frame)` with a `task` key, `save_episode()`, `clear_episode_buffer()`, `finalize()`, and `lerobot.utils.feature_utils.hw_to_dataset_features`.

---

### Task 1: Remove the order-picking agent and seat Play in the nav

**Files:**

- Delete: `lelab/agent/__init__.py`, `lelab/agent/loop.py`, `lelab/agent/executor.py`, `lelab/agent/planner.py`, `lelab/agent/session.py`, `lelab/agent/tracing.py`
- Delete: `tests/test_agent_loop.py`, `tests/test_agent_executor.py`, `tests/test_agent_planner.py`, `tests/test_agent_session.py`, `tests/test_agent_tracing.py`
- Delete: `frontend/src/pages/Orders.tsx`, `frontend/src/pages/Orders.test.tsx`
- Create: `frontend/src/pages/Play.tsx`
- Modify: `lelab/server.py` (the `from .agent.session import (...)` block near line 41 and the `OrderAnswerBody` model plus the four `/orders` routes near lines 556–589)
- Modify: `frontend/src/App.tsx`, `frontend/src/components/shell/nav.ts`, `frontend/src/components/shell/nav.test.ts`
- Modify: `README.md` (lines 13, 27, 37, 118), `tests/test_forge_readme.py`

The generic helpers `action_from_message` and `_message_for_planner` in `lelab/agent/planner.py` come back in M3 from commit `9d150a2`. Nothing else from `lelab/agent/` is reused.

- [ ] **Step 1: Make the nav test describe the new shell**

Edit `frontend/src/components/shell/nav.test.ts`: replace `"Orders"` with `"Play"` in the label list, replace `expect(active("/orders")).toEqual(["Orders"]);` with `expect(active("/play")).toEqual(["Play"]);`, and replace `"/orders"` with `"/play"` in the route list.

- [ ] **Step 2: Run the frontend tests and watch nav fail**

Run: `cd frontend && npm test -- --run src/components/shell/nav.test.ts`
Expected: 3 failures mentioning `Play`.

- [ ] **Step 3: Replace Orders with Play**

Create `frontend/src/pages/Play.tsx`:

```tsx
import { PlaceholderPage } from "./PlaceholderPage";

export default function Play() {
  return (
    <PlaceholderPage
      title="Play"
      body="Talk to the chess agent and play a game against the arm. Voice and the game loop arrive in later milestones; calibrate the board on Setup and record transfers first."
    />
  );
}
```

In `frontend/src/App.tsx` change `import Orders from "@/pages/Orders";` to `import Play from "@/pages/Play";` and `<Route path="/orders" element={<Orders />} />` to `<Route path="/play" element={<Play />} />`.

In `frontend/src/components/shell/nav.ts` replace the Orders entry with:

```ts
  { label: "Play", to: "/play", isActive: (p) => p === "/play" },
```

Delete `frontend/src/pages/Orders.tsx` and `frontend/src/pages/Orders.test.tsx`.

- [ ] **Step 4: Remove the backend agent**

Delete the six files under `lelab/agent/` and the five `tests/test_agent_*.py` files.

In `lelab/server.py` delete the import block:

```python
from .agent.session import (
    OrderBusyError,
    OrderRequest,
    handle_answer,
    handle_order_status,
    handle_start_order,
    handle_stop_order,
)
```

and delete the `class OrderAnswerBody(BaseModel)` model together with the `@app.post("/orders")`, `@app.get("/orders/current")`, `@app.post("/orders/current/stop")` and `@app.post("/orders/current/answer")` handlers. Run `rg -n "Order|orders" lelab/server.py`; expected: no matches.

- [ ] **Step 5: Trim the README and its test**

In `README.md`:

- Line 13: replace `let an agent fill picking orders with the trained skills, and improve the weakest skill from what failed` with `let a voice agent play chess with the trained skill, and improve the weakest transfers from what failed`.
- Line 27: replace `7. **Order:** chain the skills into one kitting order and run it.` with `7. **Play:** play a game against the arm by voice.`
- Line 37: replace the `| Orders | ... |` row with `| Play | Serverless Inference, CoreWeave Agent Lens | Placeholder; the chess agent arrives in M3–M6 |`.
- Line 118: replace the `lelab/agent/` bullet with

```markdown
- `lelab/chess/`: game rules, board calibration, board reading and the goal overlay.
```

In `tests/test_forge_readme.py` change `"Order"` to `"Play"` in the step tuple.

- [ ] **Step 6: Run everything**

Run: `uv run pytest -q`
Expected: pass, no `test_agent_*` collected.

Run: `cd frontend && npm test -- --run && npx tsc --noEmit -p tsconfig.app.json`
Expected: pass.

Run: `uv run ruff check lelab tests && uv run ruff format --check lelab tests`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add -u lelab/agent tests/test_agent_loop.py tests/test_agent_executor.py tests/test_agent_planner.py tests/test_agent_session.py tests/test_agent_tracing.py frontend/src/pages/Orders.tsx frontend/src/pages/Orders.test.tsx
git add lelab/server.py frontend/src/App.tsx frontend/src/components/shell/nav.ts frontend/src/components/shell/nav.test.ts frontend/src/pages/Play.tsx README.md tests/test_forge_readme.py
git commit -m "refactor: remove the order-picking agent, add the Play placeholder"
```

Before committing, run `git diff --cached --stat` and confirm `frontend/vite.config.ts`, `lelab/scripts/lelab.py` and `tests/test_scripts_lelab.py` are not staged. `lelab/server.py` carries a pre-existing uncommitted hunk (CORS origins `:8080` → `:8081` near line 194): use `git add -p lelab/server.py` and stage only the agent-removal hunks, leaving the CORS hunk unstaged. M3 touches `server.py` and `record.py` the same way.

---

### Task 2: `chess/game.py` — moves to transfers

**Files:**

- Modify: `pyproject.toml` (`dependencies`: add `"chess>=1.11"` after `"python-dotenv>=1.0",`), `uv.lock` (regenerated)
- Create: `lelab/chess/__init__.py` (header + one-line docstring), `lelab/chess/game.py`
- Create: `tests/test_chess_game.py`

**Interfaces:**

- Produces: `BIN`, `PARK_WHITE`, `PARK_BLACK`, `EXTRA_LOCATIONS`, `Transfer = tuple[str, str]`, `Occupancy = Literal["empty", "white", "black"]`, `park_square(color)`, `piece_colours(board) -> dict[str, Occupancy]`, `move_to_transfers(board, move) -> list[Transfer]`, `GameState` with `fen`, `turn`, `legal_moves()`, `parse(text)`, `transfers_for(move)`, `apply(move)`, `result()`, `occupancy()`.

- [ ] **Step 1: Add the dependency**

Add `"chess>=1.11",` to `dependencies` in `pyproject.toml`, then run `uv lock && uv sync`. Expected: `uv run python -c "import chess; print(chess.__version__)"` prints `1.11.x`.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_chess_game.py  (Apache header above)
import chess
import pytest

from lelab.chess.game import BIN, PARK_BLACK, PARK_WHITE, GameState, move_to_transfers


def transfers(fen: str, uci: str) -> list[tuple[str, str]]:
    board = chess.Board(fen)
    return move_to_transfers(board, chess.Move.from_uci(uci))


def test_simple_move_is_one_transfer() -> None:
    assert transfers(chess.STARTING_FEN, "e2e4") == [("e2", "e4")]


def test_capture_clears_the_target_into_the_bin_first() -> None:
    fen = "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2"
    assert transfers(fen, "d4e5") == [("e5", BIN), ("d4", "e5")]


def test_en_passant_removes_the_pawn_beside_the_mover() -> None:
    fen = "rnbqkbnr/pp2pppp/8/2ppP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
    assert transfers(fen, "e5d6") == [("d5", BIN), ("e5", "d6")]


@pytest.mark.parametrize(
    ("uci", "expected"),
    [
        ("e1g1", [("e1", "g1"), ("h1", "f1")]),
        ("e1c1", [("e1", "c1"), ("a1", "d1")]),
    ],
)
def test_castling_moves_king_then_rook(uci: str, expected: list[tuple[str, str]]) -> None:
    fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
    assert transfers(fen, uci) == expected


def test_black_castling_uses_the_eighth_rank() -> None:
    fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1"
    assert transfers(fen, "e8g8") == [("e8", "g8"), ("h8", "f8")]


def test_promotion_bins_the_pawn_and_fetches_the_parked_queen() -> None:
    assert transfers("8/P6k/8/8/8/8/8/K7 w - - 0 1", "a7a8q") == [("a7", BIN), (PARK_WHITE, "a8")]
    assert transfers("k7/8/8/8/8/8/p7/7K b - - 0 1", "a2a1q") == [("a2", BIN), (PARK_BLACK, "a1")]


def test_promotion_with_capture_bins_the_victim_first() -> None:
    assert transfers("1n5k/P7/8/8/8/8/8/K7 w - - 0 1", "a7b8q") == [
        ("b8", BIN),
        ("a7", BIN),
        (PARK_WHITE, "b8"),
    ]


def test_illegal_move_is_rejected() -> None:
    with pytest.raises(ValueError):
        transfers(chess.STARTING_FEN, "e2e5")


def test_game_state_parses_san_and_uci_and_applies() -> None:
    game = GameState()
    assert game.turn == "white"
    move = game.parse("Nf3")
    assert move == chess.Move.from_uci("g1f3")
    assert game.parse("e2e4") == chess.Move.from_uci("e2e4")
    assert game.apply(move) == [("g1", "f3")]
    assert game.turn == "black"
    assert game.fen.startswith("rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b")
    with pytest.raises(ValueError):
        game.parse("Nf3")
    with pytest.raises(ValueError):
        game.parse("castle please")


def test_game_state_reports_result_and_occupancy() -> None:
    game = GameState("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert game.result() == "0-1"
    occupancy = GameState().occupancy()
    assert len(occupancy) == 64
    assert occupancy["e2"] == "white"
    assert occupancy["e7"] == "black"
    assert occupancy["e4"] == "empty"
    assert GameState().result() is None
```

- [ ] **Step 3: Run it**

Run: `uv run pytest tests/test_chess_game.py -q`
Expected: `ModuleNotFoundError: No module named 'lelab.chess'`.

- [ ] **Step 4: Implement**

`lelab/chess/__init__.py`: header, then `"""Chess rules, board calibration, board reading and the goal overlay."""`.

```python
# lelab/chess/game.py  (Apache header above)
"""Game state over python-chess and the move -> transfers mapping.

A transfer is one pick-and-place `(source, target)` the arm performs. Sources and
targets are square names (`e2`), the capture `bin`, or the parking squares that
hold a spare queen per colour for promotions.
"""

from __future__ import annotations

from typing import Literal

import chess

BIN = "bin"
PARK_WHITE = "park_white"
PARK_BLACK = "park_black"
EXTRA_LOCATIONS = (BIN, PARK_WHITE, PARK_BLACK)

Transfer = tuple[str, str]
Occupancy = Literal["empty", "white", "black"]


def park_square(color: chess.Color) -> str:
    return PARK_WHITE if color == chess.WHITE else PARK_BLACK


def piece_colours(board: chess.Board) -> dict[str, Occupancy]:
    """Occupancy of all 64 squares as the camera should see them."""
    out: dict[str, Occupancy] = {}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            out[chess.square_name(square)] = "empty"
        else:
            out[chess.square_name(square)] = "white" if piece.color == chess.WHITE else "black"
    return out


def _castling_rook(move: chess.Move, kingside: bool) -> Transfer:
    rank = chess.square_rank(move.from_square)
    rook_from = chess.square(7 if kingside else 0, rank)
    rook_to = chess.square(5 if kingside else 3, rank)
    return (chess.square_name(rook_from), chess.square_name(rook_to))


def move_to_transfers(board: chess.Board, move: chess.Move) -> list[Transfer]:
    """Ordered transfers that realise `move` on the physical board.

    Captured pieces go to the bin before the mover arrives. Castling moves the
    king, then the rook. A promotion bins the pawn and brings the parked queen.
    """
    if move not in board.legal_moves:
        raise ValueError(f"{move.uci()} is not legal in {board.fen()}")
    source = chess.square_name(move.from_square)
    target = chess.square_name(move.to_square)
    transfers: list[Transfer] = []

    if board.is_en_passant(move):
        captured = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
        transfers.append((chess.square_name(captured), BIN))
    elif board.is_capture(move):
        transfers.append((target, BIN))

    if board.is_castling(move):
        transfers.append((source, target))
        transfers.append(_castling_rook(move, board.is_kingside_castling(move)))
        return transfers

    if move.promotion is not None:
        transfers.append((source, BIN))
        transfers.append((park_square(board.turn), target))
        return transfers

    transfers.append((source, target))
    return transfers


class GameState:
    """The digital board: whose turn, legal moves, parsing and results."""

    def __init__(self, fen: str = chess.STARTING_FEN) -> None:
        self.board = chess.Board(fen)

    @property
    def fen(self) -> str:
        return self.board.fen()

    @property
    def turn(self) -> str:
        return "white" if self.board.turn == chess.WHITE else "black"

    def legal_moves(self) -> list[chess.Move]:
        return list(self.board.legal_moves)

    def parse(self, text: str) -> chess.Move:
        """Read a move in SAN (`Nf3`, `O-O`) or UCI (`g1f3`, `e7e8q`)."""
        text = text.strip()
        try:
            return self.board.parse_san(text)
        except ValueError:
            pass
        try:
            move = chess.Move.from_uci(text.lower())
        except ValueError as e:
            raise ValueError(f"Cannot read move {text!r}") from e
        if move not in self.board.legal_moves:
            raise ValueError(f"{text} is not legal here")
        return move

    def transfers_for(self, move: chess.Move) -> list[Transfer]:
        return move_to_transfers(self.board, move)

    def apply(self, move: chess.Move) -> list[Transfer]:
        transfers = move_to_transfers(self.board, move)
        self.board.push(move)
        return transfers

    def result(self) -> str | None:
        """`1-0`, `0-1` or `1/2-1/2` once the game is over, else None."""
        if not self.board.is_game_over(claim_draw=True):
            return None
        return self.board.result(claim_draw=True)

    def occupancy(self) -> dict[str, Occupancy]:
        return piece_colours(self.board)
```

- [ ] **Step 5: Run it**

Run: `uv run pytest tests/test_chess_game.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock lelab/chess/__init__.py lelab/chess/game.py tests/test_chess_game.py
git commit -m "feat: chess game state and move-to-transfer mapping"
```

---

### Task 3: `chess/calibration.py` — polygons, homography, JSON

**Files:**

- Modify: `lelab/utils/config.py` (add two constants after `CAMERA_REFERENCES_PATH`)
- Modify: `tests/conftest.py` (`tmp_lerobot_home` redirects the new constants)
- Create: `lelab/chess/calibration.py`
- Create: `tests/chess_render.py` (synthetic board renderer shared by the chess tests), `tests/test_chess_calibration.py`

**Interfaces:**

- Produces: `BoardCalibration` (fields `robot`, `image_size`, `board_corners`, `bin`, `park_white`, `park_black`, `pixel_diff_threshold=40`, `occupied_fraction=0.15`, `colour_threshold=128.0`; methods `square_polygon(name)`, `polygons()`, `square_center(name)`, `marker_radius(name)`, `scaled_to(width, height)`, `to_dict()`, `from_dict(data)`), `CalibrationError`, `BOARD_CORNER_ORDER = ("a1", "h1", "h8", "a8")`, `LOCATION_NAMES` (64 squares + the three extras), `load_calibration()`, `save_calibration(cal)`.

- [ ] **Step 1: Constants and fixture**

In `lelab/utils/config.py`, after `CAMERA_REFERENCES_PATH`:

```python
# Garage demo state: the chess board calibration (see lelab/chess/calibration.py)
GARAGE_CONFIG_PATH = os.path.expanduser("~/.cache/huggingface/lerobot/garage")
BOARD_CALIBRATION_FILE = os.path.join(GARAGE_CONFIG_PATH, "board_calibration.json")
```

In `tests/conftest.py` inside `tmp_lerobot_home`, after the `FOLLOWER_CONFIG_FILE` patch:

```python
    garage_dir = cache / "garage"
    monkeypatch.setattr(cfg, "GARAGE_CONFIG_PATH", str(garage_dir))
    monkeypatch.setattr(cfg, "BOARD_CALIBRATION_FILE", str(garage_dir / "board_calibration.json"))
    monkeypatch.setattr(cfg, "CAMERA_REFERENCES_PATH", str(cache / "camera_references"))
```

- [ ] **Step 2: Write the renderer and the failing tests**

```python
# tests/chess_render.py  (Apache header above)
"""Synthetic top-camera frames for the chess vision tests.

A 400x400 image with the board filling 40..360 in both axes, a1 at the bottom
left as a camera above the board sees it, a bin and two parking squares to the
right. Pieces are discs: light for white, dark for black.
"""

from __future__ import annotations

import chess
import cv2
import numpy as np

from lelab.chess.calibration import BoardCalibration

SIZE = 400
LIGHT = 170
DARK = 90
BACKGROUND = 120


def calibration(robot: str = "arm") -> BoardCalibration:
    return BoardCalibration(
        robot=robot,
        image_size=(SIZE, SIZE),
        board_corners=[(40, 360), (360, 360), (360, 40), (40, 40)],
        bin=[(365, 300), (395, 300), (395, 360), (365, 360)],
        park_white=[(365, 40), (395, 40), (395, 70), (365, 70)],
        park_black=[(365, 80), (395, 80), (395, 110), (365, 110)],
    )


def _poly(cal: BoardCalibration, name: str) -> np.ndarray:
    return np.round(cal.square_polygon(name)).astype(np.int32)


def empty_board(cal: BoardCalibration) -> np.ndarray:
    frame = np.full((SIZE, SIZE, 3), BACKGROUND, dtype=np.uint8)
    for name in chess.SQUARE_NAMES:
        square = chess.parse_square(name)
        shade = LIGHT if (chess.square_file(square) + chess.square_rank(square)) % 2 else DARK
        cv2.fillPoly(frame, [_poly(cal, name)], (shade, shade, shade))
    for name in ("bin", "park_white", "park_black"):
        cv2.fillPoly(frame, [_poly(cal, name)], (LIGHT, LIGHT, LIGHT))
    return frame


def render(cal: BoardCalibration, occupancy: dict[str, str]) -> np.ndarray:
    """The empty board with a disc on every non-empty location."""
    frame = empty_board(cal)
    for name, value in occupancy.items():
        if value == "empty":
            continue
        colour = (235, 235, 235) if value == "white" else (15, 15, 15)
        cx, cy = cal.square_center(name)
        cv2.circle(frame, (int(round(cx)), int(round(cy))), cal.marker_radius(name) + 6, colour, -1)
    return frame
```

```python
# tests/test_chess_calibration.py  (Apache header above)
import json

import cv2
import numpy as np
import pytest

from lelab.chess.calibration import (
    BOARD_CORNER_ORDER,
    LOCATION_NAMES,
    BoardCalibration,
    CalibrationError,
    load_calibration,
    save_calibration,
)
from tests.chess_render import calibration


def test_axis_aligned_board_maps_squares_to_40px_cells() -> None:
    cal = calibration()
    assert BOARD_CORNER_ORDER == ("a1", "h1", "h8", "a8")
    a1 = cal.square_polygon("a1")
    np.testing.assert_allclose(a1, [[40, 360], [80, 360], [80, 320], [40, 320]], atol=1e-3)
    h8 = cal.square_polygon("h8")
    np.testing.assert_allclose(h8, [[320, 80], [360, 80], [360, 40], [320, 40]], atol=1e-3)
    assert cal.square_center("e4") == pytest.approx((220.0, 220.0))
    assert cal.marker_radius("e4") == 8
    assert cal.square_center("bin") == pytest.approx((380.0, 330.0))


def test_every_centre_lies_in_exactly_its_own_polygon() -> None:
    cal = calibration()
    polygons = cal.polygons()
    assert set(polygons) == set(LOCATION_NAMES) and len(polygons) == 67
    for name in LOCATION_NAMES:
        cx, cy = cal.square_center(name)
        inside = [
            other
            for other, poly in polygons.items()
            if cv2.pointPolygonTest(poly.astype(np.float32), (cx, cy), False) >= 0
        ]
        assert inside == [name]


def test_perspective_corners_still_give_a_grid() -> None:
    cal = BoardCalibration(
        robot="arm",
        image_size=(640, 480),
        board_corners=[(120, 440), (520, 440), (440, 80), (200, 80)],
        bin=[(560, 300), (620, 300), (620, 400), (560, 400)],
        park_white=[(560, 60), (620, 60), (620, 120), (560, 120)],
        park_black=[(560, 140), (620, 140), (620, 200), (560, 200)],
    )
    e4 = cal.square_center("e4")
    d5 = cal.square_center("d5")
    assert e4[1] > d5[1]  # rank 4 is closer to the camera-bottom than rank 5
    assert e4[0] > d5[0]  # e is to the right of d


def test_scaled_to_rescales_every_point_and_keeps_thresholds() -> None:
    cal = calibration()
    scaled = cal.scaled_to(800, 400)
    assert scaled.image_size == (800, 400)
    assert scaled.square_center("e4") == pytest.approx((440.0, 220.0))
    assert scaled.pixel_diff_threshold == cal.pixel_diff_threshold
    assert cal.scaled_to(400, 400) is cal


def test_rejects_bad_shapes_and_names() -> None:
    good = calibration().to_dict()
    with pytest.raises(CalibrationError):
        BoardCalibration.from_dict({**good, "board_corners": good["board_corners"][:3]})
    with pytest.raises(CalibrationError):
        BoardCalibration.from_dict({**good, "robot": "../etc"})
    with pytest.raises(CalibrationError):
        BoardCalibration.from_dict({**good, "image_size": [0, 400]})
    with pytest.raises(CalibrationError):
        BoardCalibration.from_dict({k: v for k, v in good.items() if k != "bin"})
    with pytest.raises(ValueError):
        calibration().square_polygon("z9")


def test_round_trips_through_json_and_disk(tmp_lerobot_home) -> None:
    assert load_calibration() is None
    cal = calibration()
    save_calibration(cal)
    loaded = load_calibration()
    assert loaded is not None
    assert loaded.to_dict() == cal.to_dict()
    assert json.loads((tmp_lerobot_home / "garage" / "board_calibration.json").read_text())["robot"] == "arm"
    assert not list((tmp_lerobot_home / "garage").glob("*.tmp"))
```

- [ ] **Step 3: Run it**

Run: `uv run pytest tests/test_chess_calibration.py -q`
Expected: `ModuleNotFoundError: No module named 'lelab.chess.calibration'`.

- [ ] **Step 4: Implement**

```python
# lelab/chess/calibration.py  (Apache header above)
"""Where the board, the bin and the parking squares are in the `top` frame.

The operator clicks the four board corners (a1, h1, h8, a8) and the four corners
of the bin and of each parking square on the saved `top` camera reference. A
homography from board units to pixels yields the 64 square polygons; the other
three polygons are stored as clicked. Pixel coordinates refer to `image_size`;
`scaled_to()` adapts them to a frame of another size.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import chess
import cv2
import numpy as np

from lelab.utils import config

from .game import BIN, EXTRA_LOCATIONS, PARK_BLACK, PARK_WHITE

Point = tuple[float, float]

BOARD_CORNER_ORDER = ("a1", "h1", "h8", "a8")
LOCATION_NAMES: tuple[str, ...] = tuple(chess.SQUARE_NAMES) + EXTRA_LOCATIONS
MARKER_DIAMETER_FRACTION = 0.4

_UNIT_CORNERS = np.array([(0, 0), (8, 0), (8, 8), (0, 8)], dtype=np.float32)


class CalibrationError(ValueError):
    """The calibration payload is unusable."""


def _quad(points: object, name: str) -> list[Point]:
    if not isinstance(points, list | tuple) or len(points) != 4:
        raise CalibrationError(f"{name} needs exactly four corner points")
    out: list[Point] = []
    for point in points:
        if not isinstance(point, list | tuple) or len(point) != 2:
            raise CalibrationError(f"{name} corners must be [x, y] pairs")
        out.append((float(point[0]), float(point[1])))
    return out


@dataclass
class BoardCalibration:
    robot: str
    image_size: tuple[int, int]
    board_corners: list[Point]
    bin: list[Point]
    park_white: list[Point]
    park_black: list[Point]
    pixel_diff_threshold: int = 40
    occupied_fraction: float = 0.15
    colour_threshold: float = 128.0

    def __post_init__(self) -> None:
        if not config.is_valid_robot_name(self.robot):
            raise CalibrationError("Invalid robot name")
        try:
            width, height = (int(v) for v in self.image_size)
        except (TypeError, ValueError) as e:
            raise CalibrationError("image_size must be [width, height]") from e
        if width <= 0 or height <= 0:
            raise CalibrationError("image_size must be positive")
        self.image_size = (width, height)
        self.board_corners = _quad(self.board_corners, "board_corners")
        self.bin = _quad(self.bin, BIN)
        self.park_white = _quad(self.park_white, PARK_WHITE)
        self.park_black = _quad(self.park_black, PARK_BLACK)
        self._homography = cv2.getPerspectiveTransform(
            _UNIT_CORNERS, np.array(self.board_corners, dtype=np.float32)
        )

    def square_polygon(self, name: str) -> np.ndarray:
        """Four pixel corners of a square or extra location, shape (4, 2), float32."""
        if name == BIN:
            return np.array(self.bin, dtype=np.float32)
        if name == PARK_WHITE:
            return np.array(self.park_white, dtype=np.float32)
        if name == PARK_BLACK:
            return np.array(self.park_black, dtype=np.float32)
        square = chess.parse_square(name)  # ValueError for anything else
        f, r = chess.square_file(square), chess.square_rank(square)
        unit = np.array([(f, r), (f + 1, r), (f + 1, r + 1), (f, r + 1)], dtype=np.float32)
        return cv2.perspectiveTransform(unit.reshape(-1, 1, 2), self._homography).reshape(4, 2)

    def polygons(self) -> dict[str, np.ndarray]:
        return {name: self.square_polygon(name) for name in LOCATION_NAMES}

    def square_center(self, name: str) -> Point:
        centre = self.square_polygon(name).mean(axis=0)
        return (float(centre[0]), float(centre[1]))

    def marker_radius(self, name: str) -> int:
        """Radius of a goal disc: diameter is 40 % of the polygon's shortest side."""
        poly = self.square_polygon(name)
        sides = [float(np.linalg.norm(poly[i] - poly[(i + 1) % 4])) for i in range(4)]
        return max(1, int(round(MARKER_DIAMETER_FRACTION * min(sides) / 2)))

    def scaled_to(self, width: int, height: int) -> BoardCalibration:
        if (width, height) == self.image_size:
            return self
        sx, sy = width / self.image_size[0], height / self.image_size[1]

        def scale(points: list[Point]) -> list[Point]:
            return [(x * sx, y * sy) for x, y in points]

        return BoardCalibration(
            robot=self.robot,
            image_size=(width, height),
            board_corners=scale(self.board_corners),
            bin=scale(self.bin),
            park_white=scale(self.park_white),
            park_black=scale(self.park_black),
            pixel_diff_threshold=self.pixel_diff_threshold,
            occupied_fraction=self.occupied_fraction,
            colour_threshold=self.colour_threshold,
        )

    def to_dict(self) -> dict:
        return {
            "robot": self.robot,
            "image_size": list(self.image_size),
            "board_corners": [list(p) for p in self.board_corners],
            "bin": [list(p) for p in self.bin],
            "park_white": [list(p) for p in self.park_white],
            "park_black": [list(p) for p in self.park_black],
            "pixel_diff_threshold": self.pixel_diff_threshold,
            "occupied_fraction": self.occupied_fraction,
            "colour_threshold": self.colour_threshold,
        }

    @classmethod
    def from_dict(cls, data: object) -> BoardCalibration:
        if not isinstance(data, dict):
            raise CalibrationError("Calibration must be a JSON object")
        try:
            return cls(
                robot=data["robot"],
                image_size=data["image_size"],
                board_corners=data["board_corners"],
                bin=data["bin"],
                park_white=data["park_white"],
                park_black=data["park_black"],
                pixel_diff_threshold=int(data.get("pixel_diff_threshold", 40)),
                occupied_fraction=float(data.get("occupied_fraction", 0.15)),
                colour_threshold=float(data.get("colour_threshold", 128.0)),
            )
        except CalibrationError:
            raise
        except (KeyError, TypeError, ValueError) as e:
            raise CalibrationError(f"Invalid calibration: {e}") from e


def load_calibration() -> BoardCalibration | None:
    path = config.BOARD_CALIBRATION_FILE
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return BoardCalibration.from_dict(json.load(f))


def save_calibration(cal: BoardCalibration) -> None:
    config._atomic_write_text(config.BOARD_CALIBRATION_FILE, json.dumps(cal.to_dict(), indent=2))
```

- [ ] **Step 5: Run it**

Run: `uv run pytest tests/test_chess_calibration.py tests/test_utils_config.py -q`
Expected: pass. If `test_axis_aligned_board_maps_squares_to_40px_cells` fails on `marker_radius`, the side is 40 px, so `0.4 * 40 / 2 = 8`; check the polygon, not the constant.

- [ ] **Step 6: Commit**

```bash
git add lelab/utils/config.py tests/conftest.py lelab/chess/calibration.py tests/chess_render.py tests/test_chess_calibration.py
git commit -m "feat: board calibration with homography and JSON storage"
```

---

### Task 4: `chess/overlay.py` — goal markers and the processor step

**Files:**

- Create: `lelab/chess/overlay.py`
- Create: `tests/test_chess_overlay.py`

**Interfaces:**

- Produces: `SOURCE_RGB = (255, 0, 0)`, `TARGET_RGB = (0, 255, 0)`, `draw_goal_markers(frame, source, target, calibration) -> np.ndarray` (returns a copy), `GoalOverlayStep(calibration, camera_key="top")` with `set_goal(source, target)`, `clear_goal()`, `goal` property, `observation(obs)`, `get_config()`, `transform_features(features)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chess_overlay.py  (Apache header above)
import cv2
import numpy as np
import pytest

from lelab.chess.overlay import SOURCE_RGB, TARGET_RGB, GoalOverlayStep, draw_goal_markers
from tests.chess_render import calibration, empty_board


def _pixels_matching(frame: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    ys, xs = np.where(np.all(frame == np.array(rgb, dtype=np.uint8), axis=-1))
    return np.stack([xs, ys], axis=1)


def _all_inside(points: np.ndarray, polygon: np.ndarray) -> bool:
    poly = polygon.astype(np.float32)
    return all(cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for x, y in points)


def test_markers_land_inside_their_squares_and_nowhere_else() -> None:
    cal = calibration()
    frame = empty_board(cal)
    before = frame.copy()
    out = draw_goal_markers(frame, "e2", "bin", cal)
    assert np.array_equal(frame, before), "input frame must not be modified"
    red = _pixels_matching(out, SOURCE_RGB)
    green = _pixels_matching(out, TARGET_RGB)
    assert 150 < len(red) < 260  # disc of radius 8 is ~200 px
    assert len(green) > 50
    assert _all_inside(red, cal.square_polygon("e2"))
    assert _all_inside(green, cal.square_polygon("bin"))
    assert len(_pixels_matching(empty_board(cal), SOURCE_RGB)) == 0


def test_step_overlays_only_the_top_key_when_a_goal_is_set() -> None:
    cal = calibration()
    step = GoalOverlayStep(calibration=cal)
    frame = empty_board(cal)
    obs = {"top": frame, "wrist": frame.copy(), "shoulder_pan.pos": 1.0}

    untouched = step.observation(dict(obs))
    assert untouched["top"] is frame
    assert step.goal is None

    step.set_goal("g1", "f3")
    assert step.goal == ("g1", "f3")
    out = step.observation(dict(obs))
    assert len(_pixels_matching(out["top"], SOURCE_RGB)) > 0
    assert np.array_equal(out["wrist"], frame)
    assert out["shoulder_pan.pos"] == 1.0
    assert step.get_config() == {"camera_key": "top", "source": "g1", "target": "f3"}

    step.clear_goal()
    assert step.observation(dict(obs))["top"] is frame


def test_step_ignores_observations_without_the_camera_and_rejects_bad_squares() -> None:
    step = GoalOverlayStep(calibration=calibration())
    step.set_goal("a1", "h8")
    obs = {"wrist": np.zeros((4, 4, 3), np.uint8)}
    assert step.observation(obs) is obs
    assert not obs["wrist"].any()
    with pytest.raises(ValueError):
        step.set_goal("a1", "z9")
    assert step.goal == ("a1", "h8"), "a rejected goal must not change the current one"
    features = {"observation": {}}
    assert step.transform_features(features) is features
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_chess_overlay.py -q`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# lelab/chess/overlay.py  (Apache header above)
"""Goal markers: a red disc on the source and a green disc on the target.

The same drawing runs at record time (so the dataset video carries the markers)
and at inference time (so the policy sees what it was trained on). It is wrapped
in a LeRobot observation processor step so both paths share it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from lerobot.processor import ObservationProcessorStep

from .calibration import BoardCalibration

SOURCE_RGB = (255, 0, 0)
TARGET_RGB = (0, 255, 0)


def _disc(frame: np.ndarray, name: str, colour: tuple[int, int, int], cal: BoardCalibration) -> None:
    cx, cy = cal.square_center(name)
    cv2.circle(frame, (int(round(cx)), int(round(cy))), cal.marker_radius(name), colour, thickness=-1)


def draw_goal_markers(
    frame: np.ndarray, source: str, target: str, calibration: BoardCalibration
) -> np.ndarray:
    """A copy of the RGB `frame` with the two markers painted in."""
    height, width = frame.shape[:2]
    cal = calibration.scaled_to(width, height)
    out = np.ascontiguousarray(frame).copy()
    _disc(out, source, SOURCE_RGB, cal)
    _disc(out, target, TARGET_RGB, cal)
    return out


@dataclass
class GoalOverlayStep(ObservationProcessorStep):
    """Paints the current goal into `observation[camera_key]`, nothing else."""

    calibration: BoardCalibration
    camera_key: str = "top"
    source: str | None = field(default=None)
    target: str | None = field(default=None)

    @property
    def goal(self) -> tuple[str, str] | None:
        if self.source is None or self.target is None:
            return None
        return (self.source, self.target)

    def set_goal(self, source: str, target: str) -> None:
        # Validate both names before touching state, so a typo cannot leave a half-set goal.
        self.calibration.square_polygon(source)
        self.calibration.square_polygon(target)
        self.source, self.target = source, target

    def clear_goal(self) -> None:
        self.source = self.target = None

    def observation(self, observation: dict) -> dict:
        goal = self.goal
        frame = observation.get(self.camera_key)
        if goal is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            return observation
        observation[self.camera_key] = draw_goal_markers(frame, goal[0], goal[1], self.calibration)
        return observation

    def get_config(self) -> dict:
        return {"camera_key": self.camera_key, "source": self.source, "target": self.target}

    def transform_features(self, features):
        return features
```

If `ObservationProcessorStep` requires `__call__` on a dataclass subclass to see `_current_transition`, nothing changes: the base class sets it inside `__call__`.

- [ ] **Step 4: Run it**

Run: `uv run pytest tests/test_chess_overlay.py -q`
Expected: pass. If the `is frame` assertions fail, `observation()` must return the same dict and the same array object when no goal is set — do not copy early.

- [ ] **Step 5: Commit**

```bash
git add lelab/chess/overlay.py tests/test_chess_overlay.py
git commit -m "feat: goal overlay markers as a LeRobot observation step"
```

---

### Task 5: `chess/vision.py` — occupancy, human move, verification

**Files:**

- Create: `lelab/chess/vision.py`
- Create: `tests/test_chess_vision.py`

**Interfaces:**

- Produces: `classify_locations(frame, reference, calibration, names=None) -> dict[str, Occupancy]`, `board_occupancy(frame, reference, calibration)` (64 squares), `candidate_moves(game, observed) -> list[chess.Move]`, `detect_human_move(game, frame, reference, calibration) -> list[chess.Move]`, `verify_squares(observed, expected) -> list[str]` (sorted names that differ).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chess_vision.py  (Apache header above)
import chess

from lelab.chess.game import GameState
from lelab.chess.vision import (
    board_occupancy,
    candidate_moves,
    classify_locations,
    detect_human_move,
    verify_squares,
)
from tests.chess_render import calibration, empty_board, render


def test_reads_the_starting_position_from_a_rendered_frame() -> None:
    cal = calibration()
    reference = empty_board(cal)
    game = GameState()
    observed = board_occupancy(render(cal, game.occupancy()), reference, cal)
    assert observed == game.occupancy()


def test_reads_bin_and_parking_too() -> None:
    cal = calibration()
    reference = empty_board(cal)
    frame = render(cal, {"bin": "black", "park_white": "white"})
    extras = classify_locations(frame, reference, cal, names=("bin", "park_white", "park_black"))
    assert extras == {"bin": "black", "park_white": "white", "park_black": "empty"}


def test_reference_of_another_size_is_rescaled() -> None:
    import cv2

    cal = calibration()
    # Nearest-neighbour 2x up, then INTER_AREA 2x down inside vision: lossless, so
    # this tests the resize path without blur at the square borders.
    reference = cv2.resize(empty_board(cal), (800, 800), interpolation=cv2.INTER_NEAREST)
    observed = board_occupancy(render(cal, {"e4": "white"}), reference, cal)
    assert observed["e4"] == "white"
    assert observed["e5"] == "empty"


def test_detects_the_single_human_move() -> None:
    cal = calibration()
    reference = empty_board(cal)
    game = GameState()
    after = GameState()
    after.apply(chess.Move.from_uci("e2e4"))
    moves = detect_human_move(game, render(cal, after.occupancy()), reference, cal)
    assert moves == [chess.Move.from_uci("e2e4")]


def test_no_candidate_when_the_board_did_not_change_or_is_impossible() -> None:
    game = GameState()
    assert candidate_moves(game, game.occupancy()) == []
    impossible = {**game.occupancy(), "e4": "black"}
    assert candidate_moves(game, impossible) == []


def test_promotion_yields_several_candidates() -> None:
    game = GameState("8/P6k/8/8/8/8/8/K7 w - - 0 1")
    observed = {**game.occupancy(), "a7": "empty", "a8": "white"}
    moves = candidate_moves(game, observed)
    assert len(moves) == 4
    assert {m.promotion for m in moves} == {chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT}


def test_verify_squares_names_only_the_differences() -> None:
    expected = GameState().occupancy()
    observed = {**expected, "e2": "empty", "e4": "white"}
    assert verify_squares(observed, expected) == ["e2", "e4"]
    assert verify_squares(expected, expected) == []
    assert verify_squares({}, {"bin": "black"}) == ["bin"]
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_chess_vision.py -q`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# lelab/chess/vision.py  (Apache header above)
"""Classical board reading against the empty-board reference frame.

Per location: the fraction of pixels whose grey level differs from the reference
by more than `pixel_diff_threshold` decides occupied vs empty; the mean grey of
those changed pixels against `colour_threshold` decides white vs black. The
thresholds live in the calibration so the operator can tune them per cell.
"""

from __future__ import annotations

from collections.abc import Iterable

import chess
import cv2
import numpy as np

from .calibration import LOCATION_NAMES, BoardCalibration
from .game import GameState, Occupancy, piece_colours


def _grey(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2GRAY)


def _classify(
    frame_grey: np.ndarray, reference_grey: np.ndarray, polygon: np.ndarray, cal: BoardCalibration
) -> Occupancy:
    mask = np.zeros(frame_grey.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
    inside = mask > 0
    if not inside.any():
        return "empty"
    diff = cv2.absdiff(frame_grey, reference_grey)
    changed = inside & (diff > cal.pixel_diff_threshold)
    if changed.sum() / inside.sum() < cal.occupied_fraction:
        return "empty"
    return "white" if float(frame_grey[changed].mean()) >= cal.colour_threshold else "black"


def classify_locations(
    frame: np.ndarray,
    reference: np.ndarray,
    calibration: BoardCalibration,
    names: Iterable[str] | None = None,
) -> dict[str, Occupancy]:
    """Occupancy of the named locations (default: all 67) in an RGB `frame`."""
    height, width = frame.shape[:2]
    cal = calibration.scaled_to(width, height)
    if reference.shape[:2] != (height, width):
        reference = cv2.resize(reference, (width, height), interpolation=cv2.INTER_AREA)
    frame_grey, reference_grey = _grey(frame), _grey(reference)
    return {
        name: _classify(frame_grey, reference_grey, cal.square_polygon(name), cal)
        for name in (LOCATION_NAMES if names is None else names)
    }


def board_occupancy(
    frame: np.ndarray, reference: np.ndarray, calibration: BoardCalibration
) -> dict[str, Occupancy]:
    return classify_locations(frame, reference, calibration, names=chess.SQUARE_NAMES)


def candidate_moves(game: GameState, observed: dict[str, Occupancy]) -> list[chess.Move]:
    """Legal moves whose resulting position matches the observed occupancy."""
    out: list[chess.Move] = []
    for move in game.legal_moves():
        board = game.board.copy(stack=False)
        board.push(move)
        if piece_colours(board) == observed:
            out.append(move)
    return out


def detect_human_move(
    game: GameState, frame: np.ndarray, reference: np.ndarray, calibration: BoardCalibration
) -> list[chess.Move]:
    return candidate_moves(game, board_occupancy(frame, reference, calibration))


def verify_squares(observed: dict[str, Occupancy], expected: dict[str, Occupancy]) -> list[str]:
    """Names in `expected` whose observed value differs (sorted)."""
    return sorted(name for name, value in expected.items() if observed.get(name) != value)
```

- [ ] **Step 4: Run it**

Run: `uv run pytest tests/test_chess_vision.py -q`
Expected: pass. If the starting position misreads a square, print `classify_locations(...)` for that square: a disc of radius 14 covers ~38 % of a 40 px square and differs from the background by ≥ 60 grey levels, so `occupied_fraction=0.15` and `pixel_diff_threshold=40` hold; look at polygon rounding first.

- [ ] **Step 5: Commit**

```bash
git add lelab/chess/vision.py tests/test_chess_vision.py
git commit -m "feat: board occupancy reading and human move detection"
```

---

### Task 6: Task string, transfer plan and sidecar

**Files:**

- Create: `lelab/chess/transfers.py`
- Create: `tests/test_chess_transfers.py`

**Interfaces:**

- Produces: `CHESS_TASK`, `SIDECAR_NAME = "garage_transfers.json"`, `TRANSFER_PLAN` (20 transfers), `transfer_class(transfer) -> str` (`near`, `mid`, `far`, `bin`, `park`), `write_transfer_sidecar(dataset_root, episode_index, transfer)`, `read_transfer_sidecar(dataset_root)`. M3 adds the real recording sampler and `chess_mode_problem` to this module.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chess_transfers.py  (Apache header above)
import json
from collections import Counter
from pathlib import Path

import chess

from lelab.chess.game import BIN, EXTRA_LOCATIONS
from lelab.chess.transfers import (
    CHESS_TASK,
    TRANSFER_PLAN,
    read_transfer_sidecar,
    transfer_class,
    write_transfer_sidecar,
)

LOCATIONS = set(chess.SQUARE_NAMES) | set(EXTRA_LOCATIONS)


def test_task_string_is_fixed() -> None:
    assert CHESS_TASK == "move the piece on the red square to the green square"


def test_transfer_plan_covers_the_board_bin_and_parking() -> None:
    assert len(TRANSFER_PLAN) == 20
    assert len(set(TRANSFER_PLAN)) == 20
    for source, target in TRANSFER_PLAN:
        assert source in LOCATIONS and target in LOCATIONS and source != target
    classes = Counter(transfer_class(t) for t in TRANSFER_PLAN)
    assert classes["bin"] == 2 and classes["park"] == 2
    assert classes["near"] and classes["mid"] and classes["far"]
    sources = {s for s, _ in TRANSFER_PLAN} | {t for _, t in TRANSFER_PLAN}
    assert {"a1", "h1", "a8", "h8"} <= sources


def test_transfer_class_uses_the_farther_square() -> None:
    assert transfer_class(("e2", "e3")) == "near"
    assert transfer_class(("e2", "e5")) == "mid"
    assert transfer_class(("b2", "g7")) == "far"
    assert transfer_class(("e4", BIN)) == "bin"
    assert transfer_class(("park_white", "c8")) == "park"


def test_sidecar_appends_sorted_and_replaces_same_index(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    write_transfer_sidecar(root, 1, ("e2", "e4"))
    write_transfer_sidecar(root, 0, ("a7", BIN))
    write_transfer_sidecar(root, 1, ("g1", "f3"))
    entries = read_transfer_sidecar(root)
    assert entries == [
        {"episode_index": 0, "source": "a7", "target": BIN},
        {"episode_index": 1, "source": "g1", "target": "f3"},
    ]
    assert json.loads((root / "meta" / "garage_transfers.json").read_text()) == entries
    assert read_transfer_sidecar(tmp_path / "missing") == []
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_chess_transfers.py -q`
Expected: `ModuleNotFoundError: No module named 'lelab.chess.transfers'`.

- [ ] **Step 3: Implement**

```python
# lelab/chess/transfers.py  (Apache header above)
"""The fixed task string, the evaluation transfer plan and the per-episode sidecar."""

from __future__ import annotations

import json
from pathlib import Path

from lelab.utils import config

from .game import BIN, PARK_BLACK, PARK_WHITE, Transfer

CHESS_TASK = "move the piece on the red square to the green square"
SIDECAR_NAME = "garage_transfers.json"

# The same 20 transfers run in simulation and on the real arm, so the sim-vs-real
# gap is one table: corners, centre, near/mid/far rows, the bin and both parking pads.
TRANSFER_PLAN: tuple[Transfer, ...] = (
    ("a1", "c3"),
    ("h1", "f3"),
    ("a8", "c6"),
    ("h8", "f6"),
    ("c3", "a1"),
    ("f3", "h1"),
    ("c6", "a8"),
    ("f6", "h8"),
    ("d4", "e5"),
    ("e5", "d4"),
    ("b2", "g7"),
    ("g7", "b2"),
    ("d1", "d8"),
    ("e8", "e1"),
    ("a4", "h5"),
    ("e2", "e3"),
    ("e4", BIN),
    ("b7", BIN),
    (PARK_WHITE, "c8"),
    (PARK_BLACK, "f1"),
)


def transfer_class(transfer: Transfer) -> str:
    """`bin`, `park`, or the row band of the farther square: near (1-3), mid (4-5), far (6-8)."""
    source, target = transfer
    if target == BIN:
        return "bin"
    if source in (PARK_WHITE, PARK_BLACK):
        return "park"
    rank = max(int(source[1]), int(target[1]))
    return "near" if rank <= 3 else "mid" if rank <= 5 else "far"


def _sidecar_path(dataset_root: Path) -> Path:
    return Path(dataset_root) / "meta" / SIDECAR_NAME


def read_transfer_sidecar(dataset_root: Path) -> list[dict]:
    path = _sidecar_path(dataset_root)
    if not path.is_file():
        return []
    return json.loads(path.read_text())


def write_transfer_sidecar(dataset_root: Path, episode_index: int, transfer: Transfer) -> Path:
    """Record which transfer episode `episode_index` shows. Re-writes an existing index."""
    path = _sidecar_path(dataset_root)
    entries = [e for e in read_transfer_sidecar(dataset_root) if e.get("episode_index") != episode_index]
    entries.append({"episode_index": episode_index, "source": transfer[0], "target": transfer[1]})
    entries.sort(key=lambda e: e["episode_index"])
    config._atomic_write_text(str(path), json.dumps(entries, indent=2))
    return path
```

`transfer_class(("e2", "e3"))` takes the farther rank, 3, so `near`; `("a4", "h5")` is `mid`.

- [ ] **Step 4: Run it**

Run: `uv run pytest tests/test_chess_transfers.py -q`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add lelab/chess/transfers.py tests/test_chess_transfers.py
git commit -m "feat: chess task string, transfer plan and episode sidecar"
```

---

### Task 7: `sim` extra, arm assets and the cell file

**Files:**

- Modify: `pyproject.toml` (new extra `sim`; add it to the `dev` dependency group), `uv.lock`
- Modify: `lelab/utils/config.py` (`SIM_CELL_FILE`, `SIM_ASSETS_PATH` under `GARAGE_CONFIG_PATH`), `tests/conftest.py` (redirect both)
- Create: `lelab/sim/__init__.py`, `lelab/sim/assets.py`, `lelab/sim/cell.py`
- Create: `tests/sim_helpers.py`, `tests/test_sim_cell.py`

**Interfaces:**

- `assets.py`: `ASSETS_REPO = "lerobot/so101-tabletop-mujoco"`, `ASSETS_REVISION = "e68145d8bd793c27fa2c13e31369a616895c5acb"`, `ARM_DIR = "robotstudio_so101"`, `arm_model_path() -> Path` (downloads `robotstudio_so101/*` once into `config.SIM_ASSETS_PATH`, returns the path of `so101.xml`), `SimUnavailable(RuntimeError)`, `require_mujoco()`.
- `cell.py`: `PieceSize(radius, height)`, `CameraPose(pos, xyaxes, fovy)`, `SimCell` (fields below; `half`, `board_center`, `location_xy(name)`, `location_rect(name)`, `to_dict`, `from_dict`), `load_cell()`, `save_cell(cell)`.
- `tests/sim_helpers.py`: `mujoco = pytest.importorskip("mujoco")`-style helpers: `requires_mujoco` marker, `renderer_or_skip(model)`, `arm_assets_or_skip()`.

- [ ] **Step 1: Dependencies and paths**

In `pyproject.toml` under `[project.optional-dependencies]` add

```toml
# MuJoCo simulation of the SO-101 chess cell (lelab/sim/). The dev group installs it.
sim = [
    "mujoco>=3.3",
    "gymnasium>=1.0",
]
```

and add `"lelab[sim]"` to the `dev` list under `[dependency-groups]`. Run `uv lock && uv sync`. Expected: `uv run python -c "import mujoco; print(mujoco.__version__)"` prints `3.3` or newer (4.0.1 on 2026-11-23).

In `lelab/utils/config.py` after `BOARD_CALIBRATION_FILE`:

```python
SIM_CELL_FILE = os.path.join(GARAGE_CONFIG_PATH, "sim_cell.json")
SIM_ASSETS_PATH = os.path.join(GARAGE_CONFIG_PATH, "sim_assets")
```

In `tests/conftest.py` `tmp_lerobot_home`, next to the garage patches:

```python
    monkeypatch.setattr(cfg, "SIM_CELL_FILE", str(garage_dir / "sim_cell.json"))
```

`SIM_ASSETS_PATH` is not redirected: tests share the real download cache so the arm is fetched once per machine.

- [ ] **Step 2: Write the failing tests**

```python
# tests/sim_helpers.py  (Apache header above)
"""Skips for the simulation tests: MuJoCo missing, no GL context, or no network for the arm assets."""

from __future__ import annotations

import importlib.util

import pytest

requires_mujoco = pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="lelab[sim] not installed")


def renderer_or_skip(model, height: int = 120, width: int = 160):
    import mujoco

    try:
        return mujoco.Renderer(model, height, width)
    except Exception as e:  # no GL context: headless CI, or the Cursor sandbox on macOS
        pytest.skip(f"no OpenGL context for MuJoCo rendering: {e}")


def arm_assets_or_skip():
    from lelab.sim.assets import SimUnavailable, arm_model_path

    try:
        return arm_model_path()
    except SimUnavailable as e:
        pytest.skip(str(e))
```

```python
# tests/test_sim_cell.py  (Apache header above)
from pathlib import Path

import pytest

from lelab.sim import assets
from lelab.sim.cell import SimCell, load_cell, save_cell


def test_default_cell_geometry() -> None:
    cell = SimCell()
    assert cell.half == pytest.approx(4 * cell.square_size)
    a1 = cell.location_xy("a1")
    h1 = cell.location_xy("h1")
    a8 = cell.location_xy("a8")
    # Rank 1 is nearest the arm (smallest x); file a is on the arm's left (+y).
    assert a1[0] == pytest.approx(cell.board_x0 + cell.square_size / 2)
    assert a1[1] > 0 > h1[1]
    assert a8[0] > a1[0]
    assert cell.location_xy("e4") == pytest.approx(
        (cell.board_x0 + 3.5 * cell.square_size, -0.5 * cell.square_size)
    )
    for name in ("bin", "park_white", "park_black"):
        x, y = cell.location_xy(name)
        assert abs(y) > cell.half, f"{name} must sit beside the board"
        (x0, y0), (x1, y1) = cell.location_rect(name)
        assert x0 < x < x1 and y0 < y < y1


def test_cell_round_trip(tmp_lerobot_home) -> None:
    assert load_cell() == SimCell()
    cell = SimCell(square_size=0.028, board_x0=0.12)
    save_cell(cell)
    assert load_cell() == cell
    assert SimCell.from_dict(cell.to_dict()) == cell


def test_cell_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        SimCell(square_size=0)
    with pytest.raises(ValueError):
        SimCell.from_dict({"square_size": "big"})
    with pytest.raises(ValueError):
        SimCell().location_xy("z9")


def test_arm_assets_download_once_at_the_pinned_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["local_dir"]) / assets.ARM_DIR
        target.mkdir(parents=True)
        (target / "so101.xml").write_text("<mujoco/>")
        return kwargs["local_dir"]

    monkeypatch.setattr("lelab.utils.config.SIM_ASSETS_PATH", str(tmp_path / "assets"))
    monkeypatch.setattr(assets, "snapshot_download", fake_download)
    path = assets.arm_model_path()
    assert path.name == "so101.xml" and path.is_file()
    assert calls[0]["repo_id"] == assets.ASSETS_REPO
    assert calls[0]["revision"] == assets.ASSETS_REVISION
    assert calls[0]["allow_patterns"] == [f"{assets.ARM_DIR}/*"]
    assert assets.arm_model_path() == path
    assert len(calls) == 1


def test_arm_assets_raise_sim_unavailable_on_download_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(**kwargs):
        raise OSError("offline")

    monkeypatch.setattr("lelab.utils.config.SIM_ASSETS_PATH", str(tmp_path / "assets"))
    monkeypatch.setattr(assets, "snapshot_download", failing)
    with pytest.raises(assets.SimUnavailable, match="lerobot/so101-tabletop-mujoco"):
        assets.arm_model_path()
```

- [ ] **Step 3: Run them**

Run: `uv run pytest tests/test_sim_cell.py -q`
Expected: `ModuleNotFoundError: No module named 'lelab.sim'`.

- [ ] **Step 4: Implement**

`lelab/sim/__init__.py`: header, then `"""MuJoCo simulation of the SO-101 chess cell."""`.

```python
# lelab/sim/assets.py  (Apache header above)
"""The SO-101 arm model for the simulator.

MuJoCo Menagerie's `robotstudio_so101` (Apache-2.0), taken from LeRobot's SO-101
tabletop environment at a pinned revision so every machine simulates the same arm.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from huggingface_hub import snapshot_download

from lelab.utils import config

ASSETS_REPO = "lerobot/so101-tabletop-mujoco"
ASSETS_REVISION = "e68145d8bd793c27fa2c13e31369a616895c5acb"
ARM_DIR = "robotstudio_so101"


class SimUnavailable(RuntimeError):
    """MuJoCo, the arm assets or a GL context is missing."""


def require_mujoco() -> None:
    if importlib.util.find_spec("mujoco") is None:
        raise SimUnavailable("MuJoCo is not installed; install the extra: uv sync --extra sim")


def arm_model_path() -> Path:
    root = Path(config.SIM_ASSETS_PATH)
    model = root / ARM_DIR / "so101.xml"
    if model.is_file():
        return model
    try:
        snapshot_download(
            repo_id=ASSETS_REPO,
            revision=ASSETS_REVISION,
            allow_patterns=[f"{ARM_DIR}/*"],
            local_dir=str(root),
        )
    except Exception as e:
        raise SimUnavailable(
            f"Could not download the SO-101 model from {ASSETS_REPO}@{ASSETS_REVISION[:8]}: {e}"
        ) from e
    if not model.is_file():
        raise SimUnavailable(f"{ASSETS_REPO}@{ASSETS_REVISION[:8]} has no {ARM_DIR}/so101.xml")
    return model
```

```python
# lelab/sim/cell.py  (Apache header above)
"""Where everything sits in the simulated cell, in metres, in the arm base frame.

x points away from the arm, y to its left, z up; the table surface is z = 0.
Rank 1 is nearest the arm and file a is on its left, as White sees the board.
The real cell is built to match this file; M3 writes measured camera poses back.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import chess

from lelab.utils import config

from ..chess.game import BIN, PARK_BLACK, PARK_WHITE


@dataclass(frozen=True)
class PieceSize:
    radius: float  # collision cylinder radius
    height: float


@dataclass(frozen=True)
class CameraPose:
    pos: tuple[float, float, float]
    xyaxes: tuple[float, float, float, float, float, float]
    fovy: float


def _default_pieces() -> dict[str, PieceSize]:
    return {
        "p": PieceSize(0.0085, 0.030),
        "n": PieceSize(0.0095, 0.036),
        "b": PieceSize(0.0095, 0.040),
        "r": PieceSize(0.0095, 0.034),
        "q": PieceSize(0.0100, 0.046),
        "k": PieceSize(0.0100, 0.050),
    }


@dataclass(frozen=True)
class SimCell:
    square_size: float = 0.028
    board_x0: float = 0.11  # x of the rank-1 edge
    board_thickness: float = 0.004
    bin_center: tuple[float, float] = (0.30, -0.17)
    bin_half: tuple[float, float] = (0.04, 0.035)
    bin_wall_height: float = 0.03
    park_white_center: tuple[float, float] = (0.17, 0.15)
    park_black_center: tuple[float, float] = (0.23, 0.15)
    park_half: float = 0.016
    pieces: dict[str, PieceSize] = field(default_factory=_default_pieces)
    top_camera: CameraPose = CameraPose((0.22, 0.0, 0.55), (0, -1, 0, 1, 0, 0), 45.0)
    side_camera: CameraPose = CameraPose((0.22, -0.45, 0.30), (1, 0, 0, 0, 0.55, 0.835), 50.0)

    def __post_init__(self) -> None:
        try:
            values = (float(self.square_size), float(self.board_x0), float(self.park_half))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid sim cell: {e}") from e
        if min(values[0], values[2]) <= 0:
            raise ValueError("square_size and park_half must be positive")

    @property
    def half(self) -> float:
        return 4 * self.square_size

    @property
    def board_center(self) -> tuple[float, float]:
        return (self.board_x0 + self.half, 0.0)

    def location_xy(self, name: str) -> tuple[float, float]:
        if name == BIN:
            return self.bin_center
        if name == PARK_WHITE:
            return self.park_white_center
        if name == PARK_BLACK:
            return self.park_black_center
        square = chess.parse_square(name)  # ValueError for anything else
        f, r = chess.square_file(square), chess.square_rank(square)
        return (self.board_x0 + (r + 0.5) * self.square_size, self.half - (f + 0.5) * self.square_size)

    def location_rect(self, name: str) -> tuple[tuple[float, float], tuple[float, float]]:
        """Axis-aligned (min, max) corners of a location on the table."""
        x, y = self.location_xy(name)
        if name == BIN:
            hx, hy = self.bin_half
        elif name in (PARK_WHITE, PARK_BLACK):
            hx = hy = self.park_half
        else:
            hx = hy = self.square_size / 2
        return ((x - hx, y - hy), (x + hx, y + hy))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SimCell:
        try:
            data = dict(data)
            if "pieces" in data:
                data["pieces"] = {k: PieceSize(**v) for k, v in data["pieces"].items()}
            for key in ("top_camera", "side_camera"):
                if key in data:
                    pose = data[key]
                    data[key] = CameraPose(tuple(pose["pos"]), tuple(pose["xyaxes"]), float(pose["fovy"]))
            for key in ("bin_center", "bin_half", "park_white_center", "park_black_center"):
                if key in data:
                    data[key] = tuple(data[key])
            return cls(**data)
        except (TypeError, KeyError) as e:
            raise ValueError(f"Invalid sim cell: {e}") from e


def load_cell() -> SimCell:
    path = config.SIM_CELL_FILE
    if not os.path.isfile(path):
        return SimCell()
    with open(path) as f:
        return SimCell.from_dict(json.load(f))


def save_cell(cell: SimCell) -> None:
    config._atomic_write_text(config.SIM_CELL_FILE, json.dumps(cell.to_dict(), indent=2))
```

Note `SimCell` is frozen with a dict field; equality works, hashing is not needed. If `SimCell.from_dict({"square_size": "big"})` passes `__post_init__` because `float("big")` raises `ValueError`, that is the intended path.

- [ ] **Step 5: Run them**

Run: `uv run pytest tests/test_sim_cell.py -q`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock lelab/utils/config.py tests/conftest.py lelab/sim/__init__.py lelab/sim/assets.py lelab/sim/cell.py tests/sim_helpers.py tests/test_sim_cell.py
git commit -m "feat: sim extra, pinned SO-101 assets and the sim cell file"
```

---

### Task 8: `sim/scene.py` — MJCF from a position

**Files:**

- Create: `lelab/sim/scene.py`
- Create: `tests/test_sim_scene.py`

**Interfaces:**

- `SimPosition(board: chess.Board, park_white: bool = True, park_black: bool = True, bin_pieces: tuple[str, ...] = ())` with `occupancy() -> dict[str, Occupancy]` over the 64 squares plus the three extras, and `piece_at(location) -> str | None` (symbol).
- `piece_body_name(location, symbol, index=0) -> str` (`p_e2_P`, `p_bin_q_1`), `scene_xml(cell, position) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sim_scene.py  (Apache header above)
import xml.etree.ElementTree as ET  # nosec B405 - parsing our own generated XML

import chess

from lelab.sim.cell import SimCell
from lelab.sim.scene import SimPosition, piece_body_name, scene_xml


def bodies(xml: str) -> dict[str, ET.Element]:
    root = ET.fromstring(xml)  # nosec B314
    return {b.get("name"): b for b in root.iter("body")}


def test_starting_position_has_32_free_pieces_and_two_parked_queens() -> None:
    xml = scene_xml(SimCell(), SimPosition(chess.Board()))
    pieces = {name: b for name, b in bodies(xml).items() if name.startswith("p_")}
    assert len(pieces) == 34
    assert piece_body_name("e2", "P") in pieces
    assert piece_body_name("park_white", "Q") in pieces
    assert piece_body_name("park_black", "q") in pieces
    assert all(b.find("freejoint") is not None for b in pieces.values())


def test_piece_positions_follow_the_cell() -> None:
    cell = SimCell()
    xml = scene_xml(cell, SimPosition(chess.Board(), park_white=False, park_black=False))
    e2 = bodies(xml)[piece_body_name("e2", "P")]
    x, y, z = (float(v) for v in e2.get("pos").split())
    ex, ey = cell.location_xy("e2")
    assert abs(x - ex) < 1e-4 and abs(y - ey) < 1e-4
    assert z > cell.board_thickness
    assert piece_body_name("park_white", "Q") not in bodies(xml)


def test_bin_contents_cameras_and_mount() -> None:
    position = SimPosition(chess.Board(chess.STARTING_FEN), bin_pieces=("p", "N"))
    xml = scene_xml(SimCell(), position)
    root = ET.fromstring(xml)  # nosec B314
    names = bodies(xml)
    assert piece_body_name("bin", "p", 0) in names and piece_body_name("bin", "N", 1) in names
    assert {c.get("name") for c in root.iter("camera")} == {"top", "side"}
    assert any(s.get("name") == "so101_mount" for s in root.iter("site"))
    assert root.find(".//material[@name='board']").get("texrepeat") == "4 4"


def test_occupancy_includes_extras() -> None:
    position = SimPosition(chess.Board(), park_white=True, park_black=False, bin_pieces=("p",))
    occ = position.occupancy()
    assert occ["e2"] == "white" and occ["e7"] == "black" and occ["e4"] == "empty"
    assert occ["park_white"] == "white" and occ["park_black"] == "empty" and occ["bin"] == "black"
    assert position.piece_at("park_white") == "Q"
    assert position.piece_at("e4") is None
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_sim_scene.py -q`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# lelab/sim/scene.py  (Apache header above)
"""MJCF for one chess position in the simulated cell.

The arm is not in this file: `world.py` attaches MuJoCo Menagerie's SO-101 at the
`so101_mount` site. Each piece is a free body with one collision cylinder and
visual-only base and head geoms; the collision shape stays a primitive so contact
is fast and stable, whatever the visuals look like.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess

from ..chess.game import BIN, PARK_BLACK, PARK_WHITE, Occupancy, piece_colours
from .cell import CameraPose, SimCell

PIECE_MASS = 0.012  # kg; light wooden pieces


@dataclass
class SimPosition:
    board: chess.Board
    park_white: bool = True
    park_black: bool = True
    bin_pieces: tuple[str, ...] = ()

    def piece_at(self, location: str) -> str | None:
        if location == PARK_WHITE:
            return "Q" if self.park_white else None
        if location == PARK_BLACK:
            return "q" if self.park_black else None
        if location == BIN:
            return self.bin_pieces[-1] if self.bin_pieces else None
        piece = self.board.piece_at(chess.parse_square(location))
        return piece.symbol() if piece else None

    def occupancy(self) -> dict[str, Occupancy]:
        occ: dict[str, Occupancy] = dict(piece_colours(self.board))
        occ[PARK_WHITE] = "white" if self.park_white else "empty"
        occ[PARK_BLACK] = "black" if self.park_black else "empty"
        top = self.piece_at(BIN)
        occ[BIN] = "empty" if top is None else ("white" if top.isupper() else "black")
        return occ


def piece_body_name(location: str, symbol: str, index: int = 0) -> str:
    return f"p_{location}_{symbol}" + (f"_{index}" if location == BIN else "")


def _piece(cell: SimCell, name: str, symbol: str, x: float, y: float, z0: float) -> str:
    size = cell.pieces[symbol.lower()]
    r, h = size.radius, size.height
    mat = "white_piece" if symbol.isupper() else "black_piece"
    z = z0 + h / 2 + 0.0005
    return (
        f'<body name="{name}" pos="{x:.5f} {y:.5f} {z:.5f}"><freejoint/>'
        f'<geom class="piece" type="cylinder" size="{r:.5f} {h / 2:.5f}" material="{mat}"/>'
        f'<geom class="piece_visual" type="cylinder" size="{1.45 * r:.5f} {0.06 * h:.5f}" '
        f'pos="0 0 {-0.44 * h:.5f}" material="{mat}"/>'
        f'<geom class="piece_visual" type="sphere" size="{1.1 * r:.5f}" pos="0 0 {h / 2 - r:.5f}" material="{mat}"/>'
        "</body>"
    )


def _camera(name: str, pose: CameraPose) -> str:
    pos = " ".join(f"{v:.4f}" for v in pose.pos)
    axes = " ".join(f"{v:.4f}" for v in pose.xyaxes)
    return f'<camera name="{name}" pos="{pos}" xyaxes="{axes}" fovy="{pose.fovy:.2f}"/>'


def scene_xml(cell: SimCell, position: SimPosition) -> str:
    top_z = cell.board_thickness
    bodies: list[str] = []
    for square, piece in position.board.piece_map().items():
        location = chess.square_name(square)
        x, y = cell.location_xy(location)
        bodies.append(_piece(cell, piece_body_name(location, piece.symbol()), piece.symbol(), x, y, top_z))
    for location, symbol, present in ((PARK_WHITE, "Q", position.park_white), (PARK_BLACK, "q", position.park_black)):
        if present:
            x, y = cell.location_xy(location)
            bodies.append(_piece(cell, piece_body_name(location, symbol), symbol, x, y, 0.002))
    bx, by = cell.bin_center
    for i, symbol in enumerate(position.bin_pieces):
        # Captured pieces lie in a row on the bin floor; they only need to be there, not tidy.
        x = bx - cell.bin_half[0] + 0.012 + (i % 5) * 0.016
        y = by - cell.bin_half[1] + 0.012 + (i // 5) * 0.016
        bodies.append(_piece(cell, piece_body_name(BIN, symbol, i), symbol, x, y, 0.003))

    cx, _ = cell.board_center
    hx, hy = cell.bin_half
    wall = cell.bin_wall_height / 2
    return f"""<mujoco model="garage_chess">
  <statistic center="{cx:.3f} 0 0.1" extent="0.6"/>
  <visual>
    <headlight diffuse="0.4 0.4 0.4" ambient="0.25 0.25 0.25" specular="0 0 0"/>
    <global offwidth="1280" offheight="960"/>
  </visual>
  <asset>
    <texture name="checker" type="2d" builtin="checker" rgb1="0.93 0.86 0.71" rgb2="0.46 0.30 0.18" width="512" height="512"/>
    <material name="board" texture="checker" texrepeat="4 4"/>
    <material name="table" rgba="0.80 0.72 0.60 1"/>
    <material name="white_piece" rgba="0.95 0.93 0.88 1"/>
    <material name="black_piece" rgba="0.08 0.08 0.08 1"/>
    <material name="bin" rgba="0.30 0.30 0.35 1"/>
    <material name="park" rgba="0.55 0.65 0.80 1"/>
  </asset>
  <default>
    <default class="piece">
      <geom mass="{PIECE_MASS}" condim="4" friction="1 0.02 0.002" solref="0.01 1" priority="1"/>
    </default>
    <default class="piece_visual">
      <geom contype="0" conaffinity="0" mass="0" group="1"/>
    </default>
    <default class="bin_wall"><geom type="box" material="bin"/></default>
  </default>
  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <light name="key" pos="0.3 -0.4 0.8" dir="-0.3 0.5 -1" diffuse="0.25 0.25 0.25" castshadow="true"/>
    <geom name="table" type="box" size="0.40 0.40 0.015" pos="0.20 0 -0.015" material="table"/>
    <geom name="board" type="box" size="{cell.half:.5f} {cell.half:.5f} {top_z / 2:.5f}" pos="{cx:.5f} 0 {top_z / 2:.5f}" material="board"/>
    <geom name="park_white_pad" type="box" size="{cell.park_half:.4f} {cell.park_half:.4f} 0.001" pos="{cell.park_white_center[0]:.4f} {cell.park_white_center[1]:.4f} 0.001" material="park"/>
    <geom name="park_black_pad" type="box" size="{cell.park_half:.4f} {cell.park_half:.4f} 0.001" pos="{cell.park_black_center[0]:.4f} {cell.park_black_center[1]:.4f} 0.001" material="park"/>
    <body name="bin" pos="{bx:.4f} {by:.4f} 0">
      <geom class="bin_wall" size="{hx:.4f} {hy:.4f} 0.0015" pos="0 0 0.0015"/>
      <geom class="bin_wall" size="0.002 {hy:.4f} {wall:.4f}" pos="{hx:.4f} 0 {wall:.4f}"/>
      <geom class="bin_wall" size="0.002 {hy:.4f} {wall:.4f}" pos="{-hx:.4f} 0 {wall:.4f}"/>
      <geom class="bin_wall" size="{hx:.4f} 0.002 {wall:.4f}" pos="0 {hy:.4f} {wall:.4f}"/>
      <geom class="bin_wall" size="{hx:.4f} 0.002 {wall:.4f}" pos="0 {-hy:.4f} {wall:.4f}"/>
    </body>
    <site name="so101_mount" pos="0 0 0"/>
    {"".join(bodies)}
    {_camera("top", cell.top_camera)}
    {_camera("side", cell.side_camera)}
  </worldbody>
</mujoco>"""
```

The builtin checker texture is 2×2 squares, so `texrepeat="4 4"` on the board box gives 8×8. A light square must sit at h1 (the square on White's right); Task 9 checks this on a rendered frame and flips the texture colours if it is wrong.

- [ ] **Step 4: Run it**

Run: `uv run pytest tests/test_sim_scene.py -q`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add lelab/sim/scene.py tests/test_sim_scene.py
git commit -m "feat: MJCF chess scene generated from a position"
```

---

### Task 9: `sim/world.py` — physics, rendering, projection

**Files:**

- Create: `lelab/sim/world.py`
- Create: `tests/test_sim_world.py`

**Interfaces:**

- `JOINTS`, `JOINT_NAMES` (`shoulder_pan.pos`, …), `HOME = (0, -60, 60, 35, 0, 30)` (LeRobot tabletop home), `PARK` (arm folded clear of the board; tuned in this task), `GRIPPER_OPEN = 70.0`, `GRIPPER_CLOSED = 2.0`.
- `compile_scene(xml: str) -> mujoco.MjModel` (scene + arm at `so101_mount`, the arm's physics options kept).
- `ChessWorld(cell, position, *, fps=30, width=640, height=480, render=True, initial_joints=PARK)` with `model`, `data`, `substeps`, `reset_joints(values)`, `step(action_deg)`, `joint_state() -> np.ndarray[6] float32`, `render(camera) -> RGB uint8`, `project(camera, points_xyz) -> (N, 2) pixels`, `board_calibration(camera="top") -> BoardCalibration`, `body_id(name)`, `piece_bodies() -> dict[str, int]`, `body_at(location) -> str | None`, `close()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sim_world.py  (Apache header above)
import chess
import numpy as np
import pytest

from tests.sim_helpers import arm_assets_or_skip, renderer_or_skip, requires_mujoco

pytestmark = requires_mujoco


@pytest.fixture
def world():
    arm_assets_or_skip()
    from lelab.sim.cell import SimCell
    from lelab.sim.scene import SimPosition
    from lelab.sim.world import ChessWorld

    w = ChessWorld(SimCell(), SimPosition(chess.Board()), render=False)
    yield w
    w.close()


def test_world_steps_in_lerobot_units(world) -> None:
    from lelab.sim.world import PARK

    assert world.joint_state() == pytest.approx(np.array(PARK, dtype=np.float32), abs=1.5)
    target = np.array([10.0, -50.0, 50.0, 40.0, 0.0, 30.0], dtype=np.float32)
    for _ in range(60):
        world.step(target)
    assert world.joint_state()[:5] == pytest.approx(target[:5], abs=3.0)


def test_pieces_rest_on_their_squares_and_body_lookup(world) -> None:
    from lelab.sim.world import PARK

    for _ in range(30):
        world.step(np.array(PARK, dtype=np.float32))
    assert world.body_at("e2") == "p_e2_P"
    assert world.body_at("e4") is None
    x, y = world.cell.location_xy("e2")
    pos = world.data.xpos[world.body_id("p_e2_P")]
    assert abs(pos[0] - x) < 0.002 and abs(pos[1] - y) < 0.002


def test_projection_matches_rendered_board_and_vision_reads_the_start(world) -> None:
    from lelab.chess.game import GameState
    from lelab.chess.vision import board_occupancy
    from lelab.sim.cell import SimCell
    from lelab.sim.scene import SimPosition
    from lelab.sim.world import ChessWorld

    renderer_or_skip(world.model)
    empty = ChessWorld(SimCell(), SimPosition(chess.Board(None), False, False))
    full = ChessWorld(SimCell(), SimPosition(chess.Board()))
    try:
        cal = full.board_calibration("top")
        reference = empty.render("top")
        frame = full.render("top")
        # Light square on h1 (White's right) and a dark square on a1.
        hx, hy = (int(round(v)) for v in cal.square_center("h1"))
        ax, ay = (int(round(v)) for v in cal.square_center("a3"))
        assert reference[hy, hx].mean() > reference[ay, ax].mean() + 40
        assert board_occupancy(frame, reference, cal) == GameState().occupancy()
    finally:
        empty.close()
        full.close()
```

`a3` is used for the dark-square check because it is dark and never covered by a piece in the empty reference; `h1` is light.

- [ ] **Step 2: Run it**

Run (outside the Cursor sandbox, so CGL can open a context): `uv run pytest tests/test_sim_world.py -q`
Expected: `ModuleNotFoundError: No module named 'lelab.sim.world'`. The first run downloads the arm (~10 MB).

- [ ] **Step 3: Implement**

```python
# lelab/sim/world.py  (Apache header above)
"""One simulated chess cell: the compiled scene with the SO-101, stepped in LeRobot units.

State and actions use LeRobot's `so101_follower` units with `use_degrees=True`:
degrees for the five arm joints and 0-100 for the gripper, absolute position
targets held for one control period (1/fps s).
"""

from __future__ import annotations

import mujoco
import numpy as np

from ..chess.calibration import BoardCalibration
from ..chess.game import BIN, PARK_BLACK, PARK_WHITE
from .assets import arm_model_path, require_mujoco
from .cell import SimCell
from .scene import SimPosition, scene_xml

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
JOINT_NAMES = [f"{joint}.pos" for joint in JOINTS]
HOME = (0.0, -60.0, 60.0, 35.0, 0.0, 30.0)
# Folded back over the base, clear of the board, so `top` sees every square.
PARK = (0.0, -100.0, 95.0, 70.0, 0.0, 30.0)
GRIPPER_OPEN = 70.0
GRIPPER_CLOSED = 2.0
MOUNT_SITE = "so101_mount"
GRIPPER_SITE = "gripperframe"


def compile_scene(xml: str) -> mujoco.MjModel:
    require_mujoco()
    spec = mujoco.MjSpec.from_string(xml)
    arm = mujoco.MjSpec.from_file(str(arm_model_path()))
    # Attaching drops the arm's tuned solver options; the scene sets none, so keep the arm's.
    for name in dir(arm.option):
        if not name.startswith("_"):
            try:
                setattr(spec.option, name, getattr(arm.option, name))
            except (AttributeError, TypeError):
                pass
    spec.attach(arm, site=spec.site(MOUNT_SITE), prefix="")
    return spec.compile()


class ChessWorld:
    def __init__(
        self,
        cell: SimCell,
        position: SimPosition,
        *,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        render: bool = True,
        initial_joints=PARK,
    ) -> None:
        self.cell = cell
        self.position = position
        self.width, self.height = width, height
        self.model = compile_scene(scene_xml(cell, position))
        self.substeps = max(1, round(1 / (fps * self.model.opt.timestep)))
        self.model.opt.timestep = 1 / (fps * self.substeps)
        self.data = mujoco.MjData(self.model)
        self.qadr = np.array([self.model.joint(n).qposadr[0] for n in JOINTS])
        self.dofadr = np.array([self.model.joint(n).dofadr[0] for n in JOINTS])
        self.actuators = np.array([self.model.actuator(n).id for n in JOINTS])
        self.lo, self.hi = self.model.actuator_ctrlrange[self.actuators].T.copy()
        self.gripper_site = self.model.site(GRIPPER_SITE).id
        self.renderer = mujoco.Renderer(self.model, height, width) if render else None
        self.reset_joints(initial_joints)

    # -- units -------------------------------------------------------------
    def to_ctrl(self, values) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        ctrl = np.deg2rad(values)
        ctrl[5] = self.lo[5] + values[5] / 100 * (self.hi[5] - self.lo[5])
        return np.clip(ctrl, self.lo, self.hi)

    def from_q(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        state = np.rad2deg(q)
        state[5] = (q[5] - self.lo[5]) / (self.hi[5] - self.lo[5]) * 100
        return state.astype(np.float32)

    def joint_state(self) -> np.ndarray:
        return self.from_q(self.data.qpos[self.qadr])

    # -- stepping ----------------------------------------------------------
    def reset_joints(self, values) -> None:
        ctrl = self.to_ctrl(values)
        self.data.qpos[self.qadr] = ctrl
        self.data.qvel[self.dofadr] = 0
        self.data.ctrl[self.actuators] = ctrl
        mujoco.mj_forward(self.model, self.data)

    def step(self, action) -> None:
        self.data.ctrl[self.actuators] = self.to_ctrl(action)
        mujoco.mj_step(self.model, self.data, nstep=self.substeps)

    # -- bodies ------------------------------------------------------------
    def body_id(self, name: str) -> int:
        return self.model.body(name).id

    def piece_bodies(self) -> dict[str, int]:
        return {
            self.model.body(i).name: i for i in range(self.model.nbody) if self.model.body(i).name.startswith("p_")
        }

    def body_at(self, location: str) -> str | None:
        """Name of the piece body that started on `location` (the top one for the bin)."""
        prefix = f"p_{location}_"
        names = sorted(n for n in self.piece_bodies() if n.startswith(prefix))
        return names[-1] if names else None

    # -- cameras -----------------------------------------------------------
    def render(self, camera: str) -> np.ndarray:
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, self.height, self.width)
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def project(self, camera: str, points) -> np.ndarray:
        """World points (N, 3) -> pixel coordinates (N, 2) in a `width`x`height` render."""
        cam = self.model.camera(camera).id
        pos = self.data.cam_xpos[cam]
        rot = self.data.cam_xmat[cam].reshape(3, 3)
        local = (np.asarray(points, dtype=np.float64) - pos) @ rot  # camera looks along -z
        focal = 0.5 * self.height / np.tan(np.deg2rad(self.model.cam_fovy[cam]) / 2)
        u = self.width / 2 + focal * local[:, 0] / -local[:, 2]
        v = self.height / 2 - focal * local[:, 1] / -local[:, 2]
        return np.stack([u, v], axis=1)

    def board_calibration(self, camera: str = "top") -> BoardCalibration:
        cell = self.cell
        z = cell.board_thickness
        x0, x1, h = cell.board_x0, cell.board_x0 + 8 * cell.square_size, cell.half
        corners = [(x0, h, z), (x0, -h, z), (x1, -h, z), (x1, h, z)]  # a1, h1, h8, a8

        def rect(name: str) -> list[tuple[float, float]]:
            (ax, ay), (bx, by) = cell.location_rect(name)
            pts = self.project(camera, [(ax, ay, 0.0), (ax, by, 0.0), (bx, by, 0.0), (bx, ay, 0.0)])
            return [tuple(map(float, p)) for p in pts]

        return BoardCalibration(
            robot="sim",
            image_size=(self.width, self.height),
            board_corners=[tuple(map(float, p)) for p in self.project(camera, corners)],
            bin=rect(BIN),
            park_white=rect(PARK_WHITE),
            park_black=rect(PARK_BLACK),
        )

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
```

- [ ] **Step 4: Run it and tune `PARK`**

Run: `uv run pytest tests/test_sim_world.py -q` (outside the sandbox).

- If `test_world_steps_in_lerobot_units` fails on the first assertion, the PARK pose collides with the table or the base and settles elsewhere: lower `shoulder_lift` magnitude or `elbow_flex` until it holds.
- If the vision test misreads ranks 1–2, the parked arm occludes them in `top`: render `top` with `PIL.Image.fromarray(world.render("top")).save("/tmp/top.png")`, adjust `PARK` (fold further back, or pan the arm 90° aside) and the default `top_camera` until every square is visible.
- If the light/dark check fails, swap `rgb1` and `rgb2` of the checker texture in `scene.py`.
- If white pieces read as `black`: pieces are rendered dark from shading; raise their `rgba` or lower `colour_threshold` in `board_calibration()` for sim only.

Expected after tuning: pass.

- [ ] **Step 5: Commit**

```bash
git add lelab/sim/world.py tests/test_sim_world.py
git commit -m "feat: simulated chess world with LeRobot units and camera projection"
```

---

### Task 10: `sim/judge.py` — outcome of a transfer

**Files:**

- Create: `lelab/sim/judge.py`
- Create: `tests/test_sim_judge.py`

**Interfaces:**

- `FAILURES = ("approach", "grasp", "drop", "place", "topple", "disturbed")`, `Outcome(success: bool, failure: str | None, max_lift: float, final_error: float)`.
- Pure: `classify(*, reached, max_lift, final_xy, final_up, target_rect, others_moved, released_near_target) -> Outcome`.
- `TransferTracker(world, transfer)` with `update()` after every step and `outcome() -> Outcome`.

- [ ] **Step 1: Write the failing test (pure, no MuJoCo)**

```python
# tests/test_sim_judge.py  (Apache header above)
from lelab.sim.judge import classify

RECT = ((0.10, -0.01), (0.12, 0.01))  # target square 2 cm wide around (0.11, 0.0)


def outcome(**overrides):
    kwargs = dict(
        reached=True,
        max_lift=0.05,
        final_xy=(0.11, 0.0),
        final_up=0.99,
        target_rect=RECT,
        others_moved=0,
        released_near_target=True,
        square_size=0.02,
        is_bin=False,
    )
    kwargs.update(overrides)
    return classify(**kwargs)


def test_success() -> None:
    result = outcome()
    assert result.success and result.failure is None and result.final_error < 1e-9


def test_failure_order() -> None:
    assert outcome(reached=False).failure == "approach"
    assert outcome(max_lift=0.01).failure == "grasp"
    assert outcome(released_near_target=False, final_xy=(0.3, 0.3)).failure == "drop"
    assert outcome(final_xy=(0.11, 0.009)).failure == "place"  # 9 mm > 35 % of 2 cm
    assert outcome(final_up=0.3).failure == "topple"
    assert outcome(others_moved=1).failure == "disturbed"


def test_bin_target_only_needs_the_piece_inside() -> None:
    bin_rect = ((0.26, -0.20), (0.34, -0.14))
    assert outcome(target_rect=bin_rect, final_xy=(0.27, -0.15), final_up=0.1, is_bin=True).success
    assert outcome(target_rect=bin_rect, final_xy=(0.25, -0.15), is_bin=True).failure == "place"
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_sim_judge.py -q`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# lelab/sim/judge.py  (Apache header above)
"""Did a simulated transfer succeed, and if not, where did it fail first."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..chess.game import BIN, Transfer

FAILURES = ("approach", "grasp", "drop", "place", "topple", "disturbed")
LIFT_M = 0.02
APPROACH_M = 0.01
PLACE_TOLERANCE = 0.35  # of a square
UPRIGHT_COS = 0.9
DISTURB_M = 0.005


@dataclass(frozen=True)
class Outcome:
    success: bool
    failure: str | None
    max_lift: float
    final_error: float


def classify(
    *,
    reached: bool,
    max_lift: float,
    final_xy,
    final_up: float,
    target_rect,
    others_moved: int,
    released_near_target: bool,
    square_size: float,
    is_bin: bool,
) -> Outcome:
    (x0, y0), (x1, y1) = target_rect
    centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    error = float(np.linalg.norm(np.asarray(final_xy) - centre))
    inside = x0 <= final_xy[0] <= x1 and y0 <= final_xy[1] <= y1
    on_target = inside if is_bin else error <= PLACE_TOLERANCE * square_size

    def fail(name: str) -> Outcome:
        return Outcome(False, name, max_lift, error)

    if not reached:
        return fail("approach")
    if max_lift < LIFT_M:
        return fail("grasp")
    if not on_target:
        return fail("place" if released_near_target else "drop")
    if not is_bin and final_up < UPRIGHT_COS:
        return fail("topple")
    if others_moved:
        return fail("disturbed")
    return Outcome(True, None, max_lift, error)


class TransferTracker:
    """Follows the moved piece and its neighbours through one transfer."""

    def __init__(self, world, transfer: Transfer) -> None:
        self.world = world
        self.transfer = transfer
        name = world.body_at(transfer[0])
        if name is None:
            raise ValueError(f"No piece on {transfer[0]}")
        self.body = world.body_id(name)
        self.others = {i: world.data.xpos[i].copy() for n, i in world.piece_bodies().items() if i != self.body}
        self.start_z = float(world.data.xpos[self.body][2])
        self.max_lift = 0.0
        self.reached = False
        self.last_lifted_xy: np.ndarray | None = None
        sx, sy = world.cell.location_xy(transfer[0])
        self.source = np.array([sx, sy])

    def update(self) -> None:
        data = self.world.data
        piece = data.xpos[self.body]
        lift = float(piece[2]) - self.start_z
        self.max_lift = max(self.max_lift, lift)
        if lift > LIFT_M / 2:
            self.last_lifted_xy = piece[:2].copy()
        tip = data.site_xpos[self.world.gripper_site]
        if np.linalg.norm(tip[:2] - self.source) < APPROACH_M and tip[2] - self.start_z < 0.05:
            self.reached = True

    def outcome(self) -> Outcome:
        data, cell = self.world.data, self.world.cell
        target = self.transfer[1]
        rect = cell.location_rect(target)
        up = float(data.xmat[self.body].reshape(3, 3)[2, 2])
        moved = 0
        for i, start in self.others.items():
            tipped = data.xmat[i].reshape(3, 3)[2, 2] < UPRIGHT_COS and not self._in_bin(i)
            if np.linalg.norm(data.xpos[i][:2] - start[:2]) > DISTURB_M or tipped:
                moved += 1
        near = self.last_lifted_xy is not None and (
            np.linalg.norm(self.last_lifted_xy - np.array(cell.location_xy(target))) < 2 * cell.square_size
        )
        return classify(
            reached=self.reached or self.max_lift >= LIFT_M,
            max_lift=self.max_lift,
            final_xy=tuple(data.xpos[self.body][:2]),
            final_up=up,
            target_rect=rect,
            others_moved=moved,
            released_near_target=bool(near),
            square_size=cell.square_size,
            is_bin=target == BIN,
        )

    def _in_bin(self, body: int) -> bool:
        (x0, y0), (x1, y1) = self.world.cell.location_rect(BIN)
        x, y = self.world.data.xpos[body][:2]
        return x0 <= x <= x1 and y0 <= y <= y1
```

- [ ] **Step 4: Run it**

Run: `uv run pytest tests/test_sim_judge.py -q`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add lelab/sim/judge.py tests/test_sim_judge.py
git commit -m "feat: classify simulated transfers into success and failure stages"
```

---

### Task 11: `sim/expert.py` — scripted demonstrations, tuned to the 90 % gate

**Files:**

- Create: `lelab/sim/expert.py`, `lelab/sim/positions.py`
- Create: `tests/test_sim_expert.py`, `tests/test_sim_positions.py`

**Interfaces:**

- `positions.py`: `random_position(rng, max_plies=60) -> SimPosition`, `prepare_position(position, transfer, rng) -> SimPosition` (puts a piece on the source if empty, empties a board target, puts a queen on a parking source), `sample_sim_transfer(position, rng) -> Transfer` (70/20/10 against the position, source ranks cycling near/mid/far through an internal counter passed in as `band`).
- `expert.py`: `ExpertParams` (`tilt_far_deg=30.0`, `tilt_rank=6`, `hover=0.06`, `travel=0.09`, `grasp_frac=0.45`, `release_above=0.004`, `open_steps=20`, `speed=0.25` m/s, `settle_steps=10`), `solve_ik(world, target_xyz, tilt_rad, q0) -> (q5, pos_err, rot_err)`, `plan_transfer(world, transfer, params) -> list[np.ndarray]` (30 fps actions in LeRobot units), `run_expert(world, transfer, params, on_step=None) -> Outcome`.

- [ ] **Step 1: Positions test**

```python
# tests/test_sim_positions.py  (Apache header above)
import random
from collections import Counter

import chess

from lelab.chess.game import BIN, PARK_BLACK, PARK_WHITE
from lelab.sim.positions import prepare_position, random_position, sample_sim_transfer
from lelab.sim.scene import SimPosition


def test_random_positions_are_legal_and_varied() -> None:
    rng = random.Random(3)
    counts = {len(random_position(rng).board.piece_map()) for _ in range(40)}
    assert max(counts) == 32 and min(counts) < 32
    assert random_position(random.Random(5)).board.is_valid()


def test_sampled_transfers_match_the_position_and_the_mix() -> None:
    rng = random.Random(11)
    kinds = Counter()
    for _ in range(1500):
        position = random_position(rng)
        source, target = sample_sim_transfer(position, rng)
        if target == BIN:
            kinds["bin"] += 1
            assert position.piece_at(source) is not None
        elif source in (PARK_WHITE, PARK_BLACK):
            kinds["park"] += 1
            assert position.board.piece_at(chess.parse_square(target)) is None
        else:
            kinds["board"] += 1
            assert position.piece_at(source) is not None
            assert position.piece_at(target) is None
    assert abs(kinds["board"] / 1500 - 0.7) < 0.05
    assert abs(kinds["bin"] / 1500 - 0.2) < 0.05


def test_prepare_position_makes_any_plan_transfer_possible() -> None:
    rng = random.Random(1)
    empty = SimPosition(chess.Board(None), park_white=False, park_black=False)
    ready = prepare_position(empty, ("a1", "c3"), rng)
    assert ready.piece_at("a1") is not None and ready.piece_at("c3") is None
    full = SimPosition(chess.Board())
    ready = prepare_position(full, ("d1", "d8"), rng)
    assert ready.piece_at("d1") is not None and ready.piece_at("d8") is None
    ready = prepare_position(empty, (PARK_BLACK, "f1"), rng)
    assert ready.park_black and ready.piece_at("f1") is None
```

- [ ] **Step 2: Implement positions**

```python
# lelab/sim/positions.py  (Apache header above)
"""Positions and transfers for simulated episodes.

Episodes start from positions reached by random legal play, so occupancy and
clutter look like real games, and the transfer is sampled against the position.
"""

from __future__ import annotations

import random

import chess

from ..chess.game import BIN, PARK_BLACK, PARK_WHITE, Transfer
from .scene import SimPosition

RANK_BANDS = ("123", "45", "678")
_PIECES = "pnbrq"


def random_position(rng: random.Random, max_plies: int = 60) -> SimPosition:
    board = chess.Board()
    captured: list[str] = []
    for _ in range(rng.randint(0, max_plies)):
        moves = list(board.legal_moves)
        if not moves or board.is_game_over():
            break
        move = rng.choice(moves)
        if board.is_capture(move):
            victim = board.piece_at(move.to_square) or chess.Piece(chess.PAWN, not board.turn)
            captured.append(victim.symbol())
        board.push(move)
    return SimPosition(
        board=chess.Board(board.fen()),
        park_white=rng.random() < 0.8,
        park_black=rng.random() < 0.8,
        bin_pieces=tuple(captured[-10:]),
    )


def _squares(position: SimPosition, occupied: bool, ranks: str = "12345678") -> list[str]:
    return [
        name
        for name in chess.SQUARE_NAMES
        if name[1] in ranks and (position.piece_at(name) is not None) == occupied
    ]


def sample_sim_transfer(position: SimPosition, rng: random.Random, band: int | None = None) -> Transfer:
    roll = rng.random()
    if roll < 0.7:
        ranks = RANK_BANDS[band % 3] if band is not None else "12345678"
        sources = _squares(position, True, ranks) or _squares(position, True)
        return (rng.choice(sources), rng.choice(_squares(position, False)))
    if roll < 0.9:
        return (rng.choice(_squares(position, True)), BIN)
    pad = PARK_WHITE if rng.random() < 0.5 else PARK_BLACK
    return (pad, rng.choice(_squares(position, False)))


def prepare_position(position: SimPosition, transfer: Transfer, rng: random.Random) -> SimPosition:
    """A copy of `position` in which `transfer` can be performed."""
    source, target = transfer
    board = position.board.copy(stack=False)
    park_white, park_black = position.park_white, position.park_black
    if source == PARK_WHITE:
        park_white = True
    elif source == PARK_BLACK:
        park_black = True
    elif board.piece_at(chess.parse_square(source)) is None:
        board.set_piece_at(
            chess.parse_square(source), chess.Piece.from_symbol(rng.choice(_PIECES + _PIECES.upper()))
        )
    if target not in (BIN, PARK_WHITE, PARK_BLACK):
        board.remove_piece_at(chess.parse_square(target))
    return SimPosition(board, park_white, park_black, position.bin_pieces)
```

`sample_sim_transfer` with `band=None` draws sources from all ranks; the generator passes a running counter for the near/mid/far cycle.

- [ ] **Step 3: Expert test**

```python
# tests/test_sim_expert.py  (Apache header above)
import chess
import pytest

from tests.sim_helpers import arm_assets_or_skip, requires_mujoco

pytestmark = requires_mujoco


@pytest.mark.parametrize("transfer", [("e2", "e4"), ("b1", "c3"), ("d7", "bin"), ("park_white", "d4")])
def test_expert_completes_named_transfers(transfer) -> None:
    arm_assets_or_skip()
    from lelab.sim.cell import SimCell
    from lelab.sim.expert import ExpertParams, run_expert
    from lelab.sim.scene import SimPosition
    from lelab.sim.world import ChessWorld

    board = chess.Board()
    if transfer[1] == "d4":
        board.remove_piece_at(chess.D4)
    world = ChessWorld(SimCell(), SimPosition(board), render=False)
    try:
        outcome = run_expert(world, transfer, ExpertParams())
    finally:
        world.close()
    assert outcome.success, outcome


def test_ik_reaches_every_square_at_hover_height() -> None:
    arm_assets_or_skip()
    import numpy as np

    from lelab.sim.cell import SimCell
    from lelab.sim.expert import ExpertParams, solve_ik, tilt_for
    from lelab.sim.scene import SimPosition
    from lelab.sim.world import PARK, ChessWorld

    world = ChessWorld(SimCell(), SimPosition(chess.Board(None), False, False), render=False)
    params = ExpertParams()
    try:
        q0 = np.deg2rad(np.array(PARK[:5]))
        misses = {}
        for name in chess.SQUARE_NAMES + ["bin", "park_white", "park_black"]:
            x, y = world.cell.location_xy(name)
            _, pos_err, _ = solve_ik(world, np.array([x, y, params.hover]), tilt_for(world.cell, name, params), q0)
            if pos_err > 0.002:
                misses[name] = round(pos_err * 1000, 1)
    finally:
        world.close()
    assert not misses, f"unreachable at hover (mm): {misses}"
```

- [ ] **Step 4: Implement the expert**

```python
# lelab/sim/expert.py  (Apache header above)
"""Scripted pick-and-place in simulation, the source of the sim demonstrations.

IK is damped least squares on the MuJoCo model for the arm's `gripperframe` site.
The site's local x axis is the approach direction (found in the 2026-11-23
prototype). Near squares are approached vertically; from `tilt_rank` on, the
approach tilts outward so far squares stay reachable and the elbow stays high.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from ..chess.game import BIN, PARK_BLACK, PARK_WHITE, Transfer
from .judge import Outcome, TransferTracker
from .world import GRIPPER_CLOSED, GRIPPER_OPEN, PARK, ChessWorld

APPROACH_AXIS = 0


@dataclass
class ExpertParams:
    tilt_far_deg: float = 30.0
    tilt_rank: int = 6
    hover: float = 0.06
    travel: float = 0.09
    grasp_frac: float = 0.45
    release_above: float = 0.004
    open_steps: int = 20
    close_steps: int = 15
    speed: float = 0.25  # m/s of the gripper site between waypoints
    settle_steps: int = 10


def tilt_for(cell, location: str, params: ExpertParams) -> float:
    if location in (BIN, PARK_WHITE, PARK_BLACK):
        return np.deg2rad(params.tilt_far_deg / 2)
    rank = int(location[1])
    if rank < params.tilt_rank:
        return 0.0
    return np.deg2rad(params.tilt_far_deg) * (rank - params.tilt_rank + 1) / (9 - params.tilt_rank)


def solve_ik(world: ChessWorld, target, tilt: float, q0, iters: int = 300):
    model = world.model
    data = mujoco.MjData(model)
    data.qpos[:] = world.data.qpos
    q = np.array(q0, dtype=np.float64)
    radial = np.array([target[0], target[1], 0.0])
    radial /= max(1e-9, np.linalg.norm(radial))
    want = np.array([radial[0] * np.sin(tilt), radial[1] * np.sin(tilt), -np.cos(tilt)])
    jp = np.zeros((3, model.nv))
    jr = np.zeros((3, model.nv))
    pos_err = rot_err = np.inf
    for _ in range(iters):
        data.qpos[world.qadr[:5]] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        site = world.gripper_site
        e_pos = np.asarray(target) - data.site_xpos[site]
        axis = data.site_xmat[site].reshape(3, 3)[:, APPROACH_AXIS]
        e_rot = np.cross(axis, want)
        pos_err, rot_err = float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot))
        if pos_err < 5e-4 and rot_err < 0.03:
            break
        mujoco.mj_jacSite(model, data, jp, jr, site)
        jac = np.vstack([jp[:, world.dofadr[:5]], 0.3 * jr[:, world.dofadr[:5]]])
        err = np.concatenate([e_pos, 0.3 * e_rot])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err)
        q = np.clip(q + dq, world.lo[:5], world.hi[:5])
    return q, pos_err, rot_err


def _piece_height(world: ChessWorld, location: str) -> float:
    name = world.body_at(location)
    if name is None:
        raise ValueError(f"No piece on {location}")
    symbol = name.split("_")[-1] if not name.startswith("p_bin_") else name.split("_")[-2]
    return world.cell.pieces[symbol.lower()].height


def plan_transfer(world: ChessWorld, transfer: Transfer, params: ExpertParams) -> list[np.ndarray]:
    source, target = transfer
    cell = world.cell
    height = _piece_height(world, source)
    base_z = cell.board_thickness if source not in (PARK_WHITE, PARK_BLACK) else 0.002
    target_z = cell.board_thickness if target not in (BIN, PARK_WHITE, PARK_BLACK) else 0.003
    grasp = base_z + params.grasp_frac * height
    release = target_z + params.grasp_frac * height + params.release_above
    if target == BIN:
        release = cell.bin_wall_height + 0.02 + params.grasp_frac * height
    sx, sy = cell.location_xy(source)
    tx, ty = cell.location_xy(target)
    st, tt = tilt_for(cell, source, params), tilt_for(cell, target, params)
    clear = max(params.travel, base_z + height + 0.04)
    waypoints = [
        ((sx, sy, base_z + height + params.hover), st, GRIPPER_OPEN, None),
        ((sx, sy, grasp), st, GRIPPER_OPEN, None),
        ((sx, sy, grasp), st, GRIPPER_CLOSED, params.close_steps),
        ((sx, sy, clear + height), st, GRIPPER_CLOSED, None),
        ((tx, ty, clear + height), tt, GRIPPER_CLOSED, None),
        ((tx, ty, release), tt, GRIPPER_CLOSED, None),
        ((tx, ty, release), tt, GRIPPER_OPEN, params.open_steps),
        ((tx, ty, target_z + height + params.hover), tt, GRIPPER_OPEN, None),
    ]
    q = np.deg2rad(np.asarray(world.joint_state()[:5], dtype=np.float64))
    current = world.joint_state().astype(np.float64)
    tip = world.data.site_xpos[world.gripper_site].copy()
    actions: list[np.ndarray] = []
    for xyz, tilt, grip, fixed_steps in waypoints:
        q, _, _ = solve_ik(world, np.array(xyz), tilt, q)
        goal = np.concatenate([np.rad2deg(q), [grip]])
        distance = float(np.linalg.norm(np.array(xyz) - tip))
        steps = fixed_steps or max(8, int(np.ceil(distance / params.speed * 30)))
        for i in range(1, steps + 1):
            s = 0.5 - 0.5 * np.cos(np.pi * i / steps)  # ease in and out
            actions.append(((1 - s) * current + s * goal).astype(np.float32))
        current, tip = goal, np.array(xyz)
    home = np.array(PARK, dtype=np.float64)
    for i in range(1, 46):
        s = 0.5 - 0.5 * np.cos(np.pi * i / 45)
        actions.append(((1 - s) * current + s * home).astype(np.float32))
    actions.extend([home.astype(np.float32)] * params.settle_steps)
    return actions


def run_expert(
    world: ChessWorld,
    transfer: Transfer,
    params: ExpertParams,
    on_step: Callable[[np.ndarray], None] | None = None,
) -> Outcome:
    """Execute the scripted transfer. `on_step(action)` runs before each step (recording)."""
    tracker = TransferTracker(world, transfer)
    for action in plan_transfer(world, transfer, params):
        if on_step is not None:
            on_step(action)
        world.step(action)
        tracker.update()
    return tracker.outcome()
```

`plan_transfer` returns the commanded joint targets; recording uses them as the `action` and the measured `joint_state()` as `observation.state`, exactly like a real teleoperated episode.

- [ ] **Step 5: Run and tune to the gate**

Run: `uv run pytest tests/test_sim_positions.py tests/test_sim_expert.py -q`

Then the gate (`python -m lelab.sim gate` arrives in Task 13; until then):

```bash
uv run python - <<'EOF'
import random
from collections import Counter

from lelab.chess.transfers import TRANSFER_PLAN, transfer_class
from lelab.sim.cell import load_cell
from lelab.sim.expert import ExpertParams, run_expert
from lelab.sim.positions import prepare_position, random_position
from lelab.sim.world import ChessWorld

cell, params, results = load_cell(), ExpertParams(), Counter()
for seed in range(5):
    rng = random.Random(seed)
    for transfer in TRANSFER_PLAN:
        position = prepare_position(random_position(rng), transfer, rng)
        world = ChessWorld(cell, position, render=False)
        outcome = run_expert(world, transfer, params)
        world.close()
        results[(transfer_class(transfer), outcome.failure or "success")] += 1
total = sum(results.values())
print({k: v for k, v in sorted(results.items())})
print("success rate", sum(v for (c, f), v in results.items() if f == "success") / total)
EOF
```

Expected after tuning: success rate ≥ 0.90. Tune in this order, one knob at a time, re-running the gate:

1. `test_ik_reaches_every_square_at_hover_height` fails on rank 8 → raise `tilt_far_deg` (up to 35) or lower `tilt_rank`; if it still fails, shrink `square_size` to 0.026 or move `board_x0` closer (0.10) in the default `SimCell` and record the choice for the real build.
2. `disturbed` dominates → raise `travel`; check the elbow over ranks 7–8 with a `side` render; more tilt keeps the elbow higher.
3. `topple` dominates → raise `release_above`, raise `open_steps`, lower `speed` on the final descent, lower `grasp_frac` so the piece hangs below the jaws.
4. `grasp` dominates → adjust `grasp_frac`, increase piece friction in `scene.py` (`friction="1.2 …"`), check `GRIPPER_CLOSED` actually squeezes (jaw contact force), check jaw opening vs `2 * radius`.
5. `place` dominates → IK residual at release height; tighten `solve_ik` tolerance or add `settle_steps` before opening.

Record the final `ExpertParams` defaults and the success table in the commit message.

- [ ] **Step 6: Commit**

```bash
git add lelab/sim/expert.py lelab/sim/positions.py tests/test_sim_expert.py tests/test_sim_positions.py lelab/sim/cell.py lelab/sim/scene.py
git commit -m "feat: scripted sim expert and episode positions; expert gate N% on the transfer plan"
```

---

### Task 12: `sim/env.py` — `ChessTransferEnv`

**Files:**

- Create: `lelab/sim/env.py`
- Create: `tests/test_sim_env.py`

**Interfaces:**

- `ChessTransferEnv(cell=None, *, fps=30, width=640, height=480, episode_seconds=30, cameras=("top", "wrist", "side"))`, Gymnasium API.
- `reset(seed=None, options={"transfer": (s, t), "position": SimPosition})` → `obs, info`; missing options are sampled. `obs = {"agent_pos": float32[6], "pixels": {"top": overlaid RGB, "wrist": RGB, "side": RGB}}`, `info = {"transfer", "is_success": False}`.
- `step(action)` → `obs, reward (1.0 on success), terminated (success, or arm parked after the piece was released), truncated (time limit), info` with `info["outcome"]` once terminated or truncated.
- Camera key `wrist` maps to the arm's `wrist_cam`.
- `observation_for_policy(obs) -> dict` returning LeRobot raw keys: `{"shoulder_pan.pos": …, …, "top": HWC, "wrist": HWC, "side": HWC}` as `predict_action` expects from a real robot.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sim_env.py  (Apache header above)
import numpy as np
import pytest

from tests.sim_helpers import arm_assets_or_skip, requires_mujoco

pytestmark = requires_mujoco


def test_env_reset_overlays_markers_and_expert_actions_succeed() -> None:
    arm_assets_or_skip()
    import mujoco

    from lelab.chess.overlay import SOURCE_RGB, TARGET_RGB
    from lelab.sim.env import ChessTransferEnv
    from lelab.sim.expert import ExpertParams, plan_transfer

    env = ChessTransferEnv(width=320, height=240)
    try:
        try:
            obs, info = env.reset(seed=0, options={"transfer": ("e2", "e4")})
        except mujoco.FatalError as e:  # pragma: no cover
            pytest.skip(str(e))
        except Exception as e:
            if "GL" in str(e) or "CGL" in str(e) or "context" in str(e).lower():
                pytest.skip(f"no OpenGL context: {e}")
            raise
        assert info["transfer"] == ("e2", "e4")
        assert obs["agent_pos"].shape == (6,)
        top = obs["pixels"]["top"]
        assert top.shape == (240, 320, 3)
        assert np.all(top == np.array(SOURCE_RGB), axis=-1).any()
        assert np.all(top == np.array(TARGET_RGB), axis=-1).any()
        assert not np.all(obs["pixels"]["wrist"] == np.array(SOURCE_RGB), axis=-1).any()

        raw = env.observation_for_policy(obs)
        assert set(raw) >= {"shoulder_pan.pos", "gripper.pos", "top", "wrist", "side"}

        terminated = truncated = False
        for action in plan_transfer(env.world, ("e2", "e4"), ExpertParams()):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        assert terminated and reward == 1.0 and info["outcome"].success
    finally:
        env.close()
```

- [ ] **Step 2: Implement**

```python
# lelab/sim/env.py  (Apache header above)
"""One chess transfer per episode, observed like the real cell.

Observations carry the overlaid `top` frame and raw `wrist`/`side` frames, so a
policy trained on sim data sees what it will see on the real arm.
"""

from __future__ import annotations

import random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..chess.overlay import GoalOverlayStep
from .cell import SimCell, load_cell
from .judge import TransferTracker
from .positions import prepare_position, random_position, sample_sim_transfer
from .world import JOINT_NAMES, PARK, ChessWorld

CAMERA_NAMES = {"top": "top", "wrist": "wrist_cam", "side": "side"}


class ChessTransferEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        cell: SimCell | None = None,
        *,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        episode_seconds: float = 30.0,
        cameras: tuple[str, ...] = ("top", "wrist", "side"),
    ) -> None:
        super().__init__()
        self.cell = cell or load_cell()
        self.fps, self.width, self.height = fps, width, height
        self.max_steps = int(episode_seconds * fps)
        self.cameras = cameras
        image = spaces.Box(0, 255, (height, width, 3), np.uint8)
        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(-360.0, 360.0, (6,), np.float32),
                "pixels": spaces.Dict({c: image for c in cameras}),
            }
        )
        self.action_space = spaces.Box(-360.0, 360.0, (6,), np.float32)
        self.world: ChessWorld | None = None
        self.overlay: GoalOverlayStep | None = None
        self.tracker: TransferTracker | None = None
        self.transfer = None
        self._steps = 0
        self._released = False

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        rng = random.Random(seed)
        position = options.get("position") or random_position(rng)
        transfer = tuple(options.get("transfer") or sample_sim_transfer(position, rng))
        position = prepare_position(position, transfer, rng)
        self.close()
        self.world = ChessWorld(self.cell, position, fps=self.fps, width=self.width, height=self.height)
        for _ in range(10):  # let the pieces settle on the board
            self.world.step(np.array(PARK, dtype=np.float32))
        self.overlay = GoalOverlayStep(calibration=self.world.board_calibration("top"))
        self.overlay.set_goal(*transfer)
        self.tracker = TransferTracker(self.world, transfer)
        self.transfer = transfer
        self._steps = 0
        self._released = False
        return self._obs(), {"transfer": transfer, "is_success": False}

    def _obs(self) -> dict[str, Any]:
        assert self.world is not None and self.overlay is not None
        pixels = {c: self.world.render(CAMERA_NAMES[c]) for c in self.cameras}
        if "top" in pixels:
            pixels["top"] = self.overlay.observation({"top": pixels["top"]})["top"]
        return {"agent_pos": self.world.joint_state(), "pixels": pixels}

    def step(self, action):
        assert self.world is not None and self.tracker is not None
        self.world.step(np.asarray(action, dtype=np.float32))
        self.tracker.update()
        self._steps += 1
        lifted = self.tracker.max_lift > 0.02
        gripper_open = self.world.joint_state()[5] > 40
        near_park = np.abs(self.world.joint_state()[:4] - np.array(PARK[:4])).max() < 8
        if lifted and gripper_open:
            self._released = True
        terminated = bool(self._released and near_park)
        truncated = self._steps >= self.max_steps
        info: dict[str, Any] = {"transfer": self.transfer, "is_success": False}
        reward = 0.0
        if terminated or truncated:
            outcome = self.tracker.outcome()
            info["outcome"] = outcome
            info["is_success"] = outcome.success
            reward = 1.0 if outcome.success else 0.0
        return self._obs(), reward, terminated, truncated, info

    def observation_for_policy(self, obs: dict[str, Any]) -> dict[str, Any]:
        raw: dict[str, Any] = {name: float(v) for name, v in zip(JOINT_NAMES, obs["agent_pos"], strict=True)}
        raw.update(obs["pixels"])
        return raw

    def close(self) -> None:
        if self.world is not None:
            self.world.close()
            self.world = None
```

Termination waits for the arm to return near PARK after releasing, so the judge sees the piece at rest.

- [ ] **Step 3: Run it**

Run (outside the sandbox): `uv run pytest tests/test_sim_env.py -q`
Expected: pass. If `terminated` never fires, the expert's final PARK approach is within 8° only after `settle_steps`; raise the tolerance or check the gripper threshold (open = 70, closed = 2).

- [ ] **Step 4: Commit**

```bash
git add lelab/sim/env.py tests/test_sim_env.py
git commit -m "feat: ChessTransferEnv with overlaid top camera and judged episodes"
```

---

### Task 13: `sim/generate.py` and the CLI — the sim dataset

**Files:**

- Create: `lelab/sim/generate.py`, `lelab/sim/__main__.py`
- Create: `tests/test_sim_generate.py`

**Interfaces:**

- `dataset_features(width, height) -> dict` built with LeRobot's `hw_to_dataset_features` from the real follower's shapes (`{f"{j}.pos": float}` and `{"top"|"wrist"|"side": (height, width, 3)}`).
- `generate_dataset(repo_id, episodes, *, root=None, seed=0, cell=None, params=None, width=640, height=480, log_artifact=True, progress=None) -> GenerationReport` (`saved`, `dropped: dict[failure, count]`, `root`).
- CLI: `uv run python -m lelab.sim generate --repo-id local/so101-chess-sim --episodes 1000 [--seed 0] [--width 640 --height 480] [--no-artifact]`, `uv run python -m lelab.sim gate [--seeds 5]` (prints the success table of Task 11 Step 5), `uv run python -m lelab.sim preview --transfer e2 e4 --out /tmp/preview.png` (saves `top|side|wrist` side by side after reset).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sim_generate.py  (Apache header above)
import json

import pytest

from tests.sim_helpers import arm_assets_or_skip, requires_mujoco


def test_features_match_the_real_follower() -> None:
    from lelab.sim.generate import dataset_features

    features = dataset_features(640, 480)
    assert features["action"]["names"] == [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]
    assert features["observation.state"]["shape"] == (6,)
    for cam in ("top", "wrist", "side"):
        assert features[f"observation.images.{cam}"]["shape"] == (480, 640, 3)
        assert features[f"observation.images.{cam}"]["dtype"] == "video"


@requires_mujoco
def test_generates_two_episodes_with_sidecar_and_metadata(tmp_lerobot_home) -> None:
    arm_assets_or_skip()
    from lelab.chess.transfers import read_transfer_sidecar
    from lelab.sim.generate import generate_dataset

    try:
        report = generate_dataset("local/test-sim", 2, seed=0, width=160, height=120, log_artifact=False)
    except Exception as e:
        if "GL" in str(e) or "context" in str(e).lower():
            pytest.skip(f"no OpenGL context: {e}")
        raise
    assert report.saved == 2
    entries = read_transfer_sidecar(report.root)
    assert [e["episode_index"] for e in entries] == [0, 1]
    meta = json.loads((report.root / "meta" / "garage_sim.json").read_text())
    assert meta["source"] == "sim" and meta["seed"] == 0 and "expert" in meta and "cell" in meta
    info = json.loads((report.root / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2
    assert info["robot_type"] == "so101_follower"
```

- [ ] **Step 2: Implement**

```python
# lelab/sim/generate.py  (Apache header above)
"""Sim demonstrations -> LeRobotDataset, with the real cell's feature names.

Each episode runs the expert twice with the same seed: a physics-only pass decides
whether the transfer succeeds (MuJoCo is deterministic), and only successful
transfers are rendered and recorded, so failed attempts cost no rendering or
video encoding.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..chess.transfers import CHESS_TASK, write_transfer_sidecar
from .assets import ASSETS_REPO, ASSETS_REVISION
from .cell import SimCell, load_cell
from .env import CAMERA_NAMES
from .expert import ExpertParams, plan_transfer, run_expert
from .positions import prepare_position, random_position, sample_sim_transfer
from .world import JOINT_NAMES, PARK, ChessWorld

CAMERAS = ("top", "wrist", "side")


@dataclass
class GenerationReport:
    root: Path
    saved: int = 0
    dropped: dict[str, int] = field(default_factory=dict)


def dataset_features(width: int, height: int) -> dict:
    from lerobot.utils.feature_utils import hw_to_dataset_features

    joints = {name: float for name in JOINT_NAMES}
    cameras = {cam: (height, width, 3) for cam in CAMERAS}
    return {
        **hw_to_dataset_features(joints, "action", True),
        **hw_to_dataset_features({**joints, **cameras}, "observation", True),
    }


def _episode(cell: SimCell, seed: int, band: int):
    rng = random.Random(seed)
    position = random_position(rng)
    transfer = sample_sim_transfer(position, rng, band=band)
    return prepare_position(position, transfer, rng), transfer


def _settle(world: ChessWorld) -> None:
    for _ in range(10):
        world.step(np.array(PARK, dtype=np.float32))


def generate_dataset(
    repo_id: str,
    episodes: int,
    *,
    root: Path | None = None,
    seed: int = 0,
    cell: SimCell | None = None,
    params: ExpertParams | None = None,
    width: int = 640,
    height: int = 480,
    log_artifact: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> GenerationReport:
    from lerobot.datasets import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME

    from ..chess.overlay import GoalOverlayStep

    cell = cell or load_cell()
    params = params or ExpertParams()
    root = Path(root) if root else Path(HF_LEROBOT_HOME) / repo_id
    dataset = LeRobotDataset.create(
        repo_id,
        30,
        dataset_features(width, height),
        root=root,
        robot_type="so101_follower",
        use_videos=True,
    )
    report = GenerationReport(root=root)
    dropped: Counter[str] = Counter()
    attempt = 0
    while report.saved < episodes:
        episode_seed = seed * 1_000_003 + attempt
        position, transfer = _episode(cell, episode_seed, band=attempt)
        attempt += 1

        dry = ChessWorld(cell, position, render=False)
        _settle(dry)
        outcome = run_expert(dry, transfer, params)
        dry.close()
        if not outcome.success:
            dropped[outcome.failure or "unknown"] += 1
            continue

        world = ChessWorld(cell, position, width=width, height=height)
        _settle(world)
        overlay = GoalOverlayStep(calibration=world.board_calibration("top"))
        overlay.set_goal(*transfer)

        def record(action: np.ndarray) -> None:
            frame = {
                "observation.state": world.joint_state(),
                "action": np.asarray(action, dtype=np.float32),
                "task": CHESS_TASK,
            }
            for cam in CAMERAS:
                image = world.render(CAMERA_NAMES[cam])
                if cam == "top":
                    image = overlay.observation({"top": image})["top"]
                frame[f"observation.images.{cam}"] = image
            dataset.add_frame(frame)

        actions = plan_transfer(world, transfer, params)
        from .judge import TransferTracker

        tracker = TransferTracker(world, transfer)
        for action in actions:
            record(action)
            world.step(action)
            tracker.update()
        world.close()
        if not tracker.outcome().success:  # determinism broken; never record a failed transfer
            dataset.clear_episode_buffer()
            dropped["nondeterministic"] += 1
            continue
        dataset.save_episode()
        write_transfer_sidecar(root, dataset.num_episodes - 1, transfer)
        report.saved += 1
        if progress:
            progress(report.saved, episodes)

    dataset.finalize()
    report.dropped = dict(dropped)
    meta = {
        "source": "sim",
        "seed": seed,
        "episodes": report.saved,
        "attempts": attempt,
        "dropped": report.dropped,
        "cell": cell.to_dict(),
        "expert": asdict(params),
        "assets": {"repo": ASSETS_REPO, "revision": ASSETS_REVISION},
        "resolution": [width, height],
    }
    (root / "meta" / "garage_sim.json").write_text(json.dumps(meta, indent=2))
    if log_artifact:
        from ..forge.tracking import log_dataset_if_configured

        log_dataset_if_configured(root, repo_id)
    return report
```

The record pass re-plans from the same state, so the actions equal the physics-only pass. `observation.state` is read before each step, `action` is the target sent in that step, matching LeRobot's record loop.

```python
# lelab/sim/__main__.py  (Apache header above)
"""`python -m lelab.sim generate | gate | preview`."""

from __future__ import annotations

import argparse
import random
from collections import Counter

import numpy as np


def _gate(seeds: int) -> int:
    from ..chess.transfers import TRANSFER_PLAN, transfer_class
    from .cell import load_cell
    from .expert import ExpertParams, run_expert
    from .positions import prepare_position, random_position
    from .world import PARK, ChessWorld

    cell, params, results = load_cell(), ExpertParams(), Counter()
    for seed in range(seeds):
        rng = random.Random(seed)
        for transfer in TRANSFER_PLAN:
            world = ChessWorld(cell, prepare_position(random_position(rng), transfer, rng), render=False)
            for _ in range(10):
                world.step(np.array(PARK, dtype=np.float32))
            outcome = run_expert(world, transfer, params)
            world.close()
            results[(transfer_class(transfer), outcome.failure or "success")] += 1
    total = sum(results.values())
    rate = sum(v for (_, f), v in results.items() if f == "success") / total
    for (cls, result), count in sorted(results.items()):
        print(f"{cls:5s} {result:10s} {count}")
    print(f"success rate {rate:.2%} over {total} transfers")
    return 0 if rate >= 0.9 else 1


def _preview(source: str, target: str, out: str) -> int:
    from PIL import Image

    from .env import ChessTransferEnv

    env = ChessTransferEnv()
    obs, _ = env.reset(seed=0, options={"transfer": (source, target)})
    env.close()
    Image.fromarray(np.concatenate([obs["pixels"][c] for c in ("top", "side", "wrist")], axis=1)).save(out)
    print(out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lelab.sim")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--repo-id", required=True)
    gen.add_argument("--episodes", type=int, default=1000)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--width", type=int, default=640)
    gen.add_argument("--height", type=int, default=480)
    gen.add_argument("--no-artifact", action="store_true")
    gate = sub.add_parser("gate")
    gate.add_argument("--seeds", type=int, default=5)
    preview = sub.add_parser("preview")
    preview.add_argument("--transfer", nargs=2, default=["e2", "e4"])
    preview.add_argument("--out", default="/tmp/garage_sim_preview.png")
    args = parser.parse_args(argv)

    if args.command == "gate":
        return _gate(args.seeds)
    if args.command == "preview":
        return _preview(*args.transfer, args.out)
    from .generate import generate_dataset

    report = generate_dataset(
        args.repo_id,
        args.episodes,
        seed=args.seed,
        width=args.width,
        height=args.height,
        log_artifact=not args.no_artifact,
        progress=lambda done, total: print(f"\r{done}/{total}", end="", flush=True),
    )
    print(f"\nsaved {report.saved} episodes to {report.root}; dropped {report.dropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run it**

Run (outside the sandbox): `uv run pytest tests/test_sim_generate.py -q`
Expected: pass. Then `uv run python -m lelab.sim preview --transfer e2 e4` and look at the image: markers on e2/e4, the board fully visible in `top`, the gripper in `wrist`.

- [ ] **Step 4: Commit**

```bash
git add lelab/sim/generate.py lelab/sim/__main__.py tests/test_sim_generate.py
git commit -m "feat: generate sim transfer datasets with the real feature names"
```

---

### Task 14: Choose the cell and produce the M1 dataset

Operational, on the demo Mac outside the sandbox. No new code unless the gate forces a change.

- [ ] **Step 1: Measure the real parts.** Gripper opening, piece base diameters and heights for the set you will buy or print; enter them in `SimCell.pieces` defaults.
- [ ] **Step 2: Gate.** `uv run python -m lelab.sim gate --seeds 5` until ≥ 90 %. Record the final `SimCell` (square size, `board_x0`, bin and parking) and `ExpertParams` defaults in code; save the cell with `save_cell(SimCell())` and note the numbers in the guide's "The cell" table — the real board is built to them in M3.
- [ ] **Step 3: Preview.** `uv run python -m lelab.sim preview --transfer h8 f6` and `--transfer e4 bin`; check far-corner and bin framing in `top` and `side`.
- [ ] **Step 4: Generate.** `uv run python -m lelab.sim generate --repo-id local/so101-chess-sim --episodes 1000`. Expect a few hours at 640×480 with three cameras. With Forge configured the dataset appears as an Artifact.
- [ ] **Step 5: Inspect.** Open the dataset in the app's dataset browser: markers visible in `top`, three streams, `meta/garage_transfers.json` with 1,000 entries, `meta/garage_sim.json` with the drop counts.
- [ ] **Step 6: Run everything and push.**

```bash
uv run pytest -q
cd frontend && npm test -- --run && cd ..
uv run ruff check lelab tests && uv run ruff format --check lelab tests
git push -u origin forge/chess
```

Commit any changed defaults with `git commit -m "chore: sim cell and expert defaults from the M1 gate"` (only `lelab/sim/cell.py`, `lelab/sim/expert.py`).

## Out of scope for M1

- Domain randomization, Staunton visuals, sim evaluation of policies, training, collection renames (M2).
- Real board calibration UI, board-state API, real recording mode, real evaluation (M3).
- Voice, player LLM, Stockfish analysis, game loop, Play page content (M4–M7).
