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

from tests.sim_helpers import arm_assets_or_skip, requires_mujoco

pytestmark = requires_mujoco


@pytest.mark.parametrize("transfer", [("e2", "e4"), ("b1", "c3"), ("d7", "bin"), ("park_white", "d4")])
def test_expert_completes_named_transfers(transfer) -> None:
    arm_assets_or_skip()
    from lelab.sim.cell import SimCell
    from lelab.sim.expert import ExpertParams, run_expert
    from lelab.sim.scene import SimPosition
    from lelab.sim.world import ChessWorld

    board = chess.Board()
    if transfer[1] == "d4":
        board.remove_piece_at(chess.D4)
    world = ChessWorld(SimCell(), SimPosition(board), render=False)
    try:
        outcome = run_expert(world, transfer, ExpertParams())
    finally:
        world.close()
    assert outcome.success, outcome


def test_ik_reaches_every_square_at_hover_height() -> None:
    arm_assets_or_skip()
    import numpy as np

    from lelab.sim.cell import SimCell
    from lelab.sim.expert import ExpertParams, solve_ik, tilt_for
    from lelab.sim.scene import SimPosition
    from lelab.sim.world import PARK, ChessWorld

    world = ChessWorld(SimCell(), SimPosition(chess.Board(None), False, False), render=False)
    params = ExpertParams()
    try:
        q0 = np.deg2rad(np.array(PARK[:5]))
        misses = {}
        for name in chess.SQUARE_NAMES + ["bin", "park_white", "park_black"]:
            x, y = world.cell.location_xy(name)
            _, pos_err, _ = solve_ik(
                world, np.array([x, y, params.hover]), tilt_for(world.cell, name, params), q0
            )
            if pos_err > 0.002:
                misses[name] = round(pos_err * 1000, 1)
    finally:
        world.close()
    assert not misses, f"unreachable at hover (mm): {misses}"
