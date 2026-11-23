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


def test_underpromotion_is_rejected_because_only_a_queen_is_parked() -> None:
    fen = "8/P6k/8/8/8/8/8/K7 w - - 0 1"
    with pytest.raises(ValueError, match="queen"):
        transfers(fen, "a7a8n")
    game = GameState(fen)
    with pytest.raises(ValueError, match="queen"):
        game.apply(chess.Move.from_uci("a7a8n"))
    assert game.fen == fen


def test_game_state_parse_rejects_the_null_move() -> None:
    game = GameState()
    with pytest.raises(ValueError):
        game.parse("0000")


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
