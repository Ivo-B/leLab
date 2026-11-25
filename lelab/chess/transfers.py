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

"""The fixed task string, the evaluation transfer plan, the per-episode sidecar, the sampler that picks
real recording transfers and the checks chess transfer mode needs before it starts."""

from __future__ import annotations

import json
import random
from pathlib import Path

from lelab.camera_config import lerobot_camera_settings
from lelab.utils import config

from .calibration import BoardCalibration, CalibrationError, load_calibration
from .game import BIN, PARK_BLACK, PARK_WHITE, Transfer

CHESS_TASK = "move the piece on the red square to the green square"
SIDECAR_NAME = "garage_transfers.json"
TOP_CAMERA = "top"

# The camera contract, shared by real recording, the dataset check and the sim generator.
# Sim and real datasets must have these cameras, this size and this fps to be mixed in one run.
DATASET_CAMERAS = ("top", "wrist", "side")
POLICY_CAMERAS = ("top", "wrist")  # what a policy reads by default; `side` is recorded for experiments
DATASET_SIZE = (640, 480)  # width, height as delivered, after rotation
DATASET_FPS = 30

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


# One Weave evaluation for both domains, so sim and real trials land in the same table.
EVAL_NAME = "so101-chess-sim-eval"

# How a transfer ends, in simulation (the judge) and on the real arm (the operator).
FAILURES = ("approach", "grasp", "drop", "place", "topple", "disturbed")
OUTCOMES = ("success", *FAILURES)

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


class ChessModeError(ValueError):
    """Chess transfer mode cannot start; the message says why."""


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
            problems.append(
                f"Set the {'/'.join(unset)} of the '{name}' camera ({width}x{height} at {DATASET_FPS} fps)."
            )
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


def chess_mode_calibration(cameras: dict[str, dict]) -> BoardCalibration:
    """The board calibration chess transfer mode will use, or raise `ChessModeError`."""
    top = cameras.get(TOP_CAMERA)
    if top is None:
        raise ChessModeError(f"Chess transfer mode needs a camera named '{TOP_CAMERA}'.")
    if top.get("rotation", 0) != 0:
        raise ChessModeError(
            f"Chess transfer mode needs the '{TOP_CAMERA}' camera at rotation 0, "
            "so the board calibration lines up with recorded frames."
        )
    problems = camera_contract_problems(cameras)
    if problems:
        raise ChessModeError(" ".join(problems))
    try:
        calibration = load_calibration()
    except CalibrationError as e:
        raise ChessModeError(
            f"The board calibration is unreadable ({e}); calibrate the board again on the Setup page."
        ) from e
    if calibration is None:
        raise ChessModeError("Calibrate the board on the Setup page before recording chess transfers.")
    return calibration


def chess_mode_problem(cameras: dict[str, dict]) -> str | None:
    """Why chess transfer mode cannot start, or None."""
    try:
        chess_mode_calibration(cameras)
    except ChessModeError as e:
        return str(e)
    return None
