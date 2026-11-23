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
