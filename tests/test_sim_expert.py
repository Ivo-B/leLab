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


def test_ik_reaches_hover_over_the_tallest_piece_everywhere() -> None:
    """The descent starts `hover` above a king's top, the gripper site `hover_site_offset` off its axis,
    at the preferred tilt and some closing direction of the planner's search."""
    arm_assets_or_skip()
    import numpy as np

    from lelab.sim.cell import SimCell
    from lelab.sim.expert import ExpertParams, solve_ik, tilt_for
    from lelab.sim.scene import SimPosition
    from lelab.sim.world import PARK, ChessWorld

    world = ChessWorld(SimCell(), SimPosition(chess.Board(None), False, False), render=False)
    params = ExpertParams()
    cell = world.cell
    king = cell.pieces["k"].height
    try:
        q0 = np.deg2rad(np.array(PARK[:5]))
        misses = {}
        for name in chess.SQUARE_NAMES + ["bin", "park_white", "park_black"]:
            x, y = cell.location_xy(name)
            surface = {"bin": cell.bin_wall_height, "park_white": 0.002, "park_black": 0.002}.get(
                name, cell.board_thickness
            )
            target = np.array([x, y, surface + king + params.hover])
            offset = [0.0, 0.0, -params.hover_site_offset]
            tilt = tilt_for(cell, name, params)
            errors = [
                solve_ik(world, target, tilt, q0, closing=np.deg2rad(closing), offset=offset)[1]
                for closing in np.arange(0.0, 360.0, params.closing_step_deg)
            ]
            if min(errors) > 0.002:
                misses[name] = round(min(errors) * 1000, 1)
    finally:
        world.close()
    assert not misses, f"unreachable at hover (mm): {misses}"


def test_plan_is_deterministic_leaves_the_world_alone_and_is_what_run_expert_executes() -> None:
    arm_assets_or_skip()
    import numpy as np

    from lelab.sim.cell import SimCell
    from lelab.sim.expert import ExpertParams, plan_transfer, run_expert
    from lelab.sim.scene import SimPosition
    from lelab.sim.world import ChessWorld

    params = ExpertParams()
    worlds = [ChessWorld(SimCell(), SimPosition(chess.Board()), render=False) for _ in range(3)]
    try:
        qpos = worlds[0].data.qpos.copy()
        margin = worlds[0].model.geom_margin.copy()
        first = plan_transfer(worlds[0], ("e2", "e4"), params)
        assert np.array_equal(worlds[0].data.qpos, qpos)
        assert np.array_equal(worlds[0].model.geom_margin, margin)
        second = plan_transfer(worlds[1], ("e2", "e4"), params)
        assert len(first) == len(second)
        assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
        seen = []
        outcome = run_expert(
            worlds[2], ("e2", "e4"), params, on_step=lambda action: seen.append(action.copy())
        )
        assert outcome.success
        assert len(seen) == len(first)
        assert all(np.array_equal(a, b) for a, b in zip(first, seen, strict=True))
    finally:
        for world in worlds:
            world.close()
