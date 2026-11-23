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
import numpy as np
import pytest

from tests.sim_helpers import arm_assets_or_skip, renderer_or_skip, requires_mujoco

pytestmark = requires_mujoco


@pytest.fixture
def world():
    arm_assets_or_skip()
    from lelab.sim.cell import SimCell
    from lelab.sim.scene import SimPosition
    from lelab.sim.world import ChessWorld

    w = ChessWorld(SimCell(), SimPosition(chess.Board()), render=False)
    yield w
    w.close()


def test_world_steps_in_lerobot_units(world) -> None:
    from lelab.sim.world import PARK

    assert world.joint_state() == pytest.approx(np.array(PARK, dtype=np.float32), abs=1.5)
    target = np.array([10.0, -50.0, 50.0, 40.0, 0.0, 30.0], dtype=np.float32)
    for _ in range(60):
        world.step(target)
    assert world.joint_state()[:5] == pytest.approx(target[:5], abs=3.0)


def test_pieces_rest_on_their_squares_and_body_lookup(world) -> None:
    from lelab.sim.world import PARK

    for _ in range(30):
        world.step(np.array(PARK, dtype=np.float32))
    assert world.body_at("e2") == "p_e2_P"
    assert world.body_at("e4") is None
    x, y = world.cell.location_xy("e2")
    pos = world.data.xpos[world.body_id("p_e2_P")]
    assert abs(pos[0] - x) < 0.002 and abs(pos[1] - y) < 0.002


def test_projection_matches_rendered_board_and_vision_reads_the_start(world) -> None:
    from lelab.chess.game import GameState
    from lelab.chess.vision import board_occupancy
    from lelab.sim.cell import SimCell
    from lelab.sim.scene import SimPosition
    from lelab.sim.world import ChessWorld

    renderer_or_skip(world.model)
    empty = ChessWorld(SimCell(), SimPosition(chess.Board(None), False, False))
    full = ChessWorld(SimCell(), SimPosition(chess.Board()))
    try:
        cal = full.board_calibration("top")
        reference = empty.render("top")
        frame = full.render("top")
        # Light square on h1 (White's right) and a dark square on a1.
        hx, hy = (int(round(v)) for v in cal.square_center("h1"))
        ax, ay = (int(round(v)) for v in cal.square_center("a3"))
        assert reference[hy, hx].mean() > reference[ay, ax].mean() + 40
        assert board_occupancy(frame, reference, cal) == GameState().occupancy()
    finally:
        empty.close()
        full.close()
