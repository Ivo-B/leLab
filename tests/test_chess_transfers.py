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


def test_outcomes_are_success_then_the_sim_failures() -> None:
    from lelab.chess.transfers import FAILURES, OUTCOMES
    from lelab.sim.judge import FAILURES as SIM_FAILURES

    assert OUTCOMES == ("success", "approach", "grasp", "drop", "place", "topple", "disturbed")
    assert SIM_FAILURES is FAILURES
