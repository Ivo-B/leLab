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
