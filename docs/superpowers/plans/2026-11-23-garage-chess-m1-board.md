# Garage Chess M1 — Board, Calibration, Overlay and Transfer Recording

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The kitting agent is gone. The app knows chess: it turns a move into pick-and-place transfers, maps the `top` camera onto 64 squares, a bin and two parking squares, reads which squares hold a white or black piece, paints the red/green goal markers into the `top` stream, and records overlay-conditioned episodes with a `garage_transfers.json` sidecar. No arm moves in a unit test.

**Architecture:** `lelab/chess/` is pure (python-chess, numpy, OpenCV; no hardware, no Forge). `game.py` wraps `chess.Board`. `calibration.py` holds the clicked polygons and a homography and saves JSON under `~/.cache/huggingface/lerobot/garage/`. `vision.py` compares a frame with the saved `top` camera reference. `overlay.py` draws the markers and wraps them in a LeRobot `ObservationProcessorStep` so recording (now) and the rollout path (M5) overlay the same way. `transfers.py` samples recording targets and writes the sidecar. `lelab/board_state.py` and three `/chess/*` routes expose calibration and board reading to the Setup page. `record.py` gains a chess transfer mode.

**Tech Stack:** Python 3.12, FastAPI, python-chess, OpenCV (`opencv-python` via LeRobot), numpy, LeRobot v0.7.0 processors, React + Vite + vitest.

Spec: `docs/superpowers/specs/2026-11-23-garage-chess-voice-design.md` (sections 4, 5.1–5.3, 6, 10, 11.2, milestone M1).

## Global Constraints

- Work on branch `forge/chess`. Push only to `origin` (`https://github.com/Ivo-B/leLab.git`). Do not push `upstream`.
- Do not start M2. No collection renames, no Sandbox or Registry changes, no training image changes. `lelab/forge/registry.py`, `lelab/forge/tracking.py` and `lelab/evaluate.py` keep their kitting strings until M2.
- Do not start the robot arm in a unit test. `lelab/chess/` never imports `lerobot.robots`, `lelab.record`, `lelab.forge` or `openai`. `overlay.py` imports only `lerobot.processor` from LeRobot.
- Task string is fixed: `move the piece on the red square to the green square`. Markers are filled discs, diameter 40 % of the square's shorter side, pure red `(255, 0, 0)` for the source and pure green `(0, 255, 0)` for the target, in RGB (LeRobot cameras return RGB).
- The camera named `top` must be configured with rotation `0` for chess transfer mode. Refuse to start otherwise.
- Board calibration lives at `~/.cache/huggingface/lerobot/garage/board_calibration.json` through constants in `lelab/utils/config.py`. Code reads the constant through the module (`config.BOARD_CALIBRATION_FILE`) at call time, never via `from ... import`, so tests can redirect it.
- `WANDB_API_KEY` and `CWSANDBOX_API_KEY` never appear in a response, a log line, or a logger call.
- New Python modules start with the Apache-2.0 header in `tests/test_forge_config.py` lines 1–13, then a blank line.
- Run Python tests with `uv run pytest`. Run frontend tests with `npm test -- --run` inside `frontend/`. Lint with `uv run ruff check` and `uv run ruff format --check`.
- The working tree already holds uncommitted edits to `frontend/vite.config.ts`, `lelab/scripts/lelab.py`, `lelab/server.py`, `tests/test_scripts_lelab.py` and an untracked `lerobot.code-workspace`. Do not commit them, do not revert them. Stage files by name (`git add <file>`), never `git add -A` or `git commit -a`.
- Commit only the files the task lists. Do not skip hooks.

Checked on 2026-11-23: `lerobot.processor.ObservationProcessorStep.observation(self, observation) -> observation` is the only abstract method besides `transform_features` (inherited no-op in `ProcessorStep`); `RobotProcessorPipeline` takes `steps`, `to_transition`, `to_output`; `observation_to_transition` and `transition_to_observation` are importable from `lerobot.processor`. `record_loop` applies `robot_observation_processor` to every observation before `build_dataset_frame`, so a step appended there reaches the recorded video. `LeRobotDatasetMetadata.save_episode` increments `info.total_episodes`, so `dataset.num_episodes - 1` after `save_episode()` is the index just saved. `cv2` 4.13 and numpy 2.2 are already installed through LeRobot; `chess` is not.

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

Before committing, run `git diff --cached --stat` and confirm `frontend/vite.config.ts`, `lelab/scripts/lelab.py` and `tests/test_scripts_lelab.py` are not staged. `lelab/server.py` carries a pre-existing uncommitted hunk (CORS origins `:8080` → `:8081` near line 194): use `git add -p lelab/server.py` and stage only the agent-removal hunks, leaving the CORS hunk unstaged. Tasks 6 and 7 touch `server.py` and `record.py` the same way.

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
- Create: `tests/chess_render.py` (synthetic board renderer shared by Tasks 3–7), `tests/test_chess_calibration.py`

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

### Task 6: Calibration and board-state API

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

### Task 7: Transfer sampling, sidecar, and the recording hooks

**Files:**

- Create: `lelab/chess/transfers.py`
- Create: `tests/test_chess_transfers.py`
- Modify: `lelab/record.py`
- Modify: `tests/test_record.py` (append tests)

**Interfaces:**

- `lelab/chess/transfers.py`: `CHESS_TASK`, `SIDECAR_NAME = "garage_transfers.json"`, `RANK_BANDS`, `TransferSampler(rng=None).sample() -> Transfer`, `write_transfer_sidecar(dataset_root: Path, episode_index: int, transfer) -> Path`, `read_transfer_sidecar(dataset_root) -> list[dict]`, `chess_mode_problem(cameras: dict) -> str | None`.
- `lelab/record.py`: `RecordingRequest.chess_transfers: bool = False`; module globals `current_transfer`, `_overlay_step`, `_transfer_sampler`; status gains `chess_transfers` and `transfer`; `/camera-feed/top` shows the overlay; `meta/garage_transfers.json` written after each saved episode.

- [ ] **Step 1: Write the failing sampler and sidecar tests**

```python
# tests/test_chess_transfers.py  (Apache header above)
import json
import random
from collections import Counter
from pathlib import Path

import chess

from lelab.chess.calibration import save_calibration
from lelab.chess.game import BIN, PARK_BLACK, PARK_WHITE
from lelab.chess.transfers import (
    CHESS_TASK,
    TransferSampler,
    chess_mode_problem,
    read_transfer_sidecar,
    write_transfer_sidecar,
)
from tests.chess_render import calibration


def test_task_string_is_fixed() -> None:
    assert CHESS_TASK == "move the piece on the red square to the green square"


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
Expected: import error for `lelab.chess.transfers`; the three new record tests fail on the missing field/keys.

- [ ] **Step 3: Implement the sampler and sidecar**

```python
# lelab/chess/transfers.py  (Apache header above)
"""What to record: sampled `(source, target)` transfers, the fixed task string,
the per-episode sidecar, and the preconditions of chess transfer mode."""

from __future__ import annotations

import json
import random
from pathlib import Path

from lelab.utils import config

from .calibration import load_calibration
from .game import BIN, PARK_BLACK, PARK_WHITE, Transfer

CHESS_TASK = "move the piece on the red square to the green square"
SIDECAR_NAME = "garage_transfers.json"
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

The module does not import `chess`; square names are built from `FILES` and `RANK_BANDS`.

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

### Task 8: Frontend — board calibration on Setup, transfer mode on Record

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

## Manual hardware checklist (operator, after Task 8)

Not part of the automated run. Record the outcome in the M1 section of the guide.

1. Board size against the follower's reach: the arm at full extension reaches a8/h8 and the far side of the bin with the gripper vertical. Note the board edge length used.
2. Piece grip: the gripper closes on the widest piece (king) with margin and does not knock neighbours when descending on a crowded centre.
3. `top` camera: rotation 0, sees the whole board, bin and both parking squares with the arm parked; capture the reference on Calibration with the board empty.
4. Setup → Board calibration: click 16 points, Save, place the starting position, Check position: 32 squares correct. If white pieces read as black on light squares, raise `colour_threshold` in the JSON; if empty squares read occupied under shadows, raise `pixel_diff_threshold`.
5. Record with Chess transfer mode on, 3 episodes: the banner shows a transfer during reset, markers are visible in the `top` tile, `meta/garage_transfers.json` has three entries, and the recorded `top` video shows the discs when opened in the dataset browser.
6. Push `forge/chess` to `origin`.

## Out of scope for M1

- Collection renames, Sandbox image, Registry (M2).
- `lelab.chess.rollout_entry` and the inference-time overlay (M5).
- Voice, player LLM, game loop, Play page content (M3–M6).
- README storyline rewrite (M6).
