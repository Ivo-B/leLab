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

from lelab.sim.judge import classify
from tests.sim_helpers import arm_assets_or_skip, requires_mujoco

RECT = ((0.10, -0.01), (0.12, 0.01))  # target square 2 cm wide around (0.11, 0.0)


def outcome(**overrides):
    kwargs = {
        "reached": True,
        "max_lift": 0.05,
        "final_xy": (0.11, 0.0),
        "final_up": 0.99,
        "target_rect": RECT,
        "others_moved": 0,
        "released_near_target": True,
        "square_size": 0.02,
        "is_bin": False,
    }
    kwargs.update(overrides)
    return classify(**kwargs)


def test_success() -> None:
    result = outcome()
    assert result.success and result.failure is None and result.final_error < 1e-9


def test_failure_order() -> None:
    assert outcome(reached=False).failure == "approach"
    assert outcome(max_lift=0.01).failure == "grasp"
    assert outcome(released_near_target=False, final_xy=(0.3, 0.3)).failure == "drop"
    assert outcome(final_xy=(0.11, 0.009)).failure == "place"  # 9 mm > 35 % of 2 cm
    assert outcome(final_up=0.3).failure == "topple"
    assert outcome(others_moved=1).failure == "disturbed"


def test_bin_target_only_needs_the_piece_inside() -> None:
    bin_rect = ((0.26, -0.20), (0.34, -0.14))
    assert outcome(target_rect=bin_rect, final_xy=(0.27, -0.15), final_up=0.1, is_bin=True).success
    assert outcome(target_rect=bin_rect, final_xy=(0.25, -0.15), is_bin=True).failure == "place"


# -- TransferTracker on a real world ------------------------------------------


@pytest.fixture
def world():
    arm_assets_or_skip()
    from lelab.sim.cell import SimCell
    from lelab.sim.scene import SimPosition
    from lelab.sim.world import PARK, ChessWorld

    # One captured pawn in the bin, lying on its side, must not count as a tipped bystander.
    w = ChessWorld(SimCell(), SimPosition(chess.Board(), bin_pieces=("p",)), render=False)
    for _ in range(30):
        w.step(np.array(PARK, dtype=np.float32))
    yield w
    w.close()


def _qpos(world, name: str) -> int:
    body = world.body_id(name)
    return int(world.model.jnt_qposadr[world.model.body_jntadr[body]])


def _put(world, name: str, xy=None, dz: float = 0.0) -> None:
    adr = _qpos(world, name)
    if xy is not None:
        world.data.qpos[adr : adr + 2] = xy
    world.data.qpos[adr + 2] += dz
    import mujoco

    mujoco.mj_forward(world.model, world.data)


@requires_mujoco
def test_tracker_idle_transfer_fails_at_approach(world) -> None:
    from lelab.sim.judge import TransferTracker

    tracker = TransferTracker(world, ("e2", "e4"))
    for _ in range(10):
        world.step(world.joint_state())
        tracker.update()
    result = tracker.outcome()
    assert not result.success and result.failure == "approach"
    assert result.max_lift == pytest.approx(0.0, abs=1e-3)


@requires_mujoco
def test_tracker_lifted_and_placed_piece_succeeds(world) -> None:
    from lelab.sim.judge import TransferTracker

    tracker = TransferTracker(world, ("e2", "e4"))
    _put(world, "p_e2_P", dz=0.05)
    tracker.update()
    target = world.cell.location_xy("e4")
    _put(world, "p_e2_P", xy=target, dz=-0.05)
    tracker.update()
    result = tracker.outcome()
    assert result.success, result
    assert result.max_lift == pytest.approx(0.05, abs=1e-3)
    assert result.final_error < 1e-3


@requires_mujoco
def test_tracker_flags_a_nudged_neighbour(world) -> None:
    from lelab.sim.judge import TransferTracker

    tracker = TransferTracker(world, ("e2", "e4"))
    _put(world, "p_e2_P", dz=0.05)
    tracker.update()
    _put(world, "p_e2_P", xy=world.cell.location_xy("e4"), dz=-0.05)
    tracker.update()
    x, y = world.cell.location_xy("d2")
    _put(world, "p_d2_P", xy=(x + 0.01, y))
    assert tracker.outcome().failure == "disturbed"


@requires_mujoco
def test_tracker_does_not_judge_pieces_already_in_the_bin(world) -> None:
    import mujoco

    from lelab.sim.judge import TransferTracker

    tracker = TransferTracker(world, ("e2", "e4"))
    _put(world, "p_e2_P", dz=0.05)
    tracker.update()
    _put(world, "p_e2_P", xy=world.cell.location_xy("e4"), dz=-0.05)
    tracker.update()
    adr = _qpos(world, "p_bin_p_0")
    world.data.qpos[adr + 3 : adr + 7] = (np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0)  # on its side
    mujoco.mj_forward(world.model, world.data)
    assert world.data.xmat[world.body_id("p_bin_p_0")].reshape(3, 3)[2, 2] < 0.5
    assert tracker.outcome().success


@requires_mujoco
def test_tracker_requires_a_piece_on_the_source(world) -> None:
    from lelab.sim.judge import TransferTracker

    with pytest.raises(ValueError, match="No piece on e4"):
        TransferTracker(world, ("e4", "e5"))
