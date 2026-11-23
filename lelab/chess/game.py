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
