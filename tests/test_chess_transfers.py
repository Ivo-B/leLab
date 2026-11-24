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
import random
from collections import Counter
from pathlib import Path

import chess

from lelab.chess.calibration import save_calibration
from lelab.chess.game import BIN, EXTRA_LOCATIONS, PARK_BLACK, PARK_WHITE
from lelab.chess.transfers import (
    CHESS_TASK,
    RANK_BANDS,
    TRANSFER_PLAN,
    TransferSampler,
    chess_mode_problem,
    read_transfer_sidecar,
    transfer_class,
    write_transfer_sidecar,
)
from lelab.utils import config
from tests.chess_render import calibration

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


def test_outcomes_are_success_then_the_sim_failures() -> None:
    from lelab.chess.transfers import FAILURES, OUTCOMES
    from lelab.sim.judge import FAILURES as SIM_FAILURES

    assert OUTCOMES == ("success", "approach", "grasp", "drop", "place", "topple", "disturbed")
    assert SIM_FAILURES is FAILURES


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
    sources = []
    while len(sources) < 30:
        source, target = sampler.sample()
        if target != BIN and source not in (PARK_WHITE, PARK_BLACK):
            sources.append(source)
    for i, source in enumerate(sources):
        assert source[1] in RANK_BANDS[i % len(RANK_BANDS)], (i, source)


def test_chess_mode_problem_checks_top_camera_rotation_and_calibration(tmp_lerobot_home) -> None:
    assert "top" in chess_mode_problem({"wrist": {"rotation": 0}})
    assert "rotation" in chess_mode_problem({"top": {"rotation": 90}})
    assert "Calibrate" in chess_mode_problem({"top": {"rotation": 0}})
    save_calibration(calibration())
    assert chess_mode_problem({"top": {"rotation": 0}, "wrist": {"rotation": 180}}) is None
    assert chess_mode_problem({"top": {}}) is None


def test_chess_mode_problem_reports_a_corrupt_calibration(tmp_lerobot_home) -> None:
    Path(config.BOARD_CALIBRATION_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(config.BOARD_CALIBRATION_FILE).write_text("{not json")
    problem = chess_mode_problem({"top": {"rotation": 0}})
    assert problem is not None and "calibration" in problem.lower()
