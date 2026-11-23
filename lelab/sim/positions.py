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
