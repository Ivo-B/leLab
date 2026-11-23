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
    frame_grey: np.ndarray, diff: np.ndarray, polygon: np.ndarray, cal: BoardCalibration
) -> Occupancy:
    mask = np.zeros(frame_grey.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
    inside = mask > 0
    if not inside.any():
        return "empty"
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
    frame_grey = _grey(frame)
    diff = cv2.absdiff(frame_grey, _grey(reference))
    return {
        name: _classify(frame_grey, diff, cal.square_polygon(name), cal)
        for name in (LOCATION_NAMES if names is None else names)
    }


def board_occupancy(
    frame: np.ndarray, reference: np.ndarray, calibration: BoardCalibration
) -> dict[str, Occupancy]:
    """Occupancy of the 64 board squares in an RGB `frame`.

    Vision reads colour, not piece type, so every piece is just "white" or "black".
    """
    return classify_locations(frame, reference, calibration, names=chess.SQUARE_NAMES)


def candidate_moves(game: GameState, observed: dict[str, Occupancy]) -> list[chess.Move]:
    """Every legal move whose resulting position matches the observed colours.

    Only the 64 board squares of `observed` are compared; `bin` and `park_*` keys
    are ignored. Promotions appear once per piece type because vision reads colour,
    not piece type. Callers ask the human when more than one move matches.
    """
    squares = {name: observed.get(name) for name in chess.SQUARE_NAMES}
    out: list[chess.Move] = []
    for move in game.legal_moves():
        board = game.board.copy(stack=False)
        board.push(move)
        if piece_colours(board) == squares:
            out.append(move)
    return out


def detect_human_move(
    game: GameState, frame: np.ndarray, reference: np.ndarray, calibration: BoardCalibration
) -> list[chess.Move]:
    """Every legal move matching the board colours seen in `frame`.

    Promotions appear once per piece type because vision reads colour, not piece
    type. Callers ask the human when more than one move matches.
    """
    return candidate_moves(game, board_occupancy(frame, reference, calibration))


def verify_squares(observed: dict[str, Occupancy], expected: dict[str, Occupancy]) -> list[str]:
    """Names in `expected` whose observed value differs (sorted)."""
    return sorted(name for name, value in expected.items() if observed.get(name) != value)
