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
    assert world.joint_state()[5] == pytest.approx(target[5], abs=2.0)


def test_pieces_rest_on_their_squares_and_body_lookup(world) -> None:
    from lelab.sim.world import PARK

    for _ in range(30):
        world.step(np.array(PARK, dtype=np.float32))
    assert world.body_at("e2") == "p_e2_P"
    assert world.body_at("e4") is None
    x, y = world.cell.location_xy("e2")
    pos = world.data.xpos[world.body_id("p_e2_P")]
    assert abs(pos[0] - x) < 0.002 and abs(pos[1] - y) < 0.002


def test_body_at_bin_is_the_highest_piece() -> None:
    arm_assets_or_skip()
    from lelab.sim.cell import SimCell
    from lelab.sim.scene import SimPosition
    from lelab.sim.world import PARK, ChessWorld

    # 12 pawns fill the bin floor; the 13th starts on a raised layer. Sorting names would pick p_bin_p_9.
    pieces = ("p",) * 12 + ("P",)
    w = ChessWorld(SimCell(), SimPosition(chess.Board(None), False, False, pieces), render=False)
    try:
        for _ in range(30):
            w.step(np.array(PARK, dtype=np.float32))
        bodies = [n for n in w.piece_bodies() if n.startswith("p_bin_")]
        assert len(bodies) == 13
        highest = max(bodies, key=lambda n: w.data.xpos[w.body_id(n)][2])
        assert highest == "p_bin_P_12"
        assert w.body_at("bin") == highest
    finally:
        w.close()


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
        # Light square on h1 (White's right) and a dark square on a3, which no piece covers.
        hx, hy = (int(round(v)) for v in cal.square_center("h1"))
        ax, ay = (int(round(v)) for v in cal.square_center("a3"))
        assert reference[hy, hx].mean() > reference[ay, ax].mean() + 40
        assert board_occupancy(frame, reference, cal) == GameState().occupancy()
    finally:
        empty.close()
        full.close()


def test_numpy_camera_model_matches_the_world_projection(world) -> None:
    import numpy as np

    from lelab.chess.camera_pose import project

    points = np.array([[0.2, 0.0, 0.0], [0.3, 0.05, 0.03], [0.16, -0.1, 0.004]])
    for name, pose in (("top", world.cell.top_camera), ("side", world.cell.side_camera)):
        got = project(pose.pos, pose.xyaxes, pose.fovy, (world.width, world.height), points)
        np.testing.assert_allclose(got, world.project(name, points), atol=1e-3)
