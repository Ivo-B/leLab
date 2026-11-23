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
        "at_rest": True,
        "on_surface": True,
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


def test_a_piece_that_is_moving_or_off_the_surface_is_not_a_success() -> None:
    assert outcome(at_rest=False).failure == "place"
    assert outcome(on_surface=False).failure == "place"
    bin_rect = ((0.26, -0.20), (0.34, -0.14))
    assert (
        outcome(target_rect=bin_rect, final_xy=(0.3, -0.17), is_bin=True, on_surface=False).failure == "place"
    )
    assert outcome(final_up=0.3, at_rest=False).failure == "topple"  # lying down is the clearer diagnosis


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


def _adr(world, name: str) -> tuple[int, int]:
    """(qpos address, dof address) of a piece's free joint."""
    joint = int(world.model.body_jntadr[world.body_id(name)])
    return int(world.model.jnt_qposadr[joint]), int(world.model.jnt_dofadr[joint])


def _z(world, name: str) -> float:
    return float(world.data.xpos[world.body_id(name)][2])


def _set(world, name: str, *, xy=None, z=None, dz: float = 0.0, quat=None, vel=None, spin=None) -> None:
    """Teleport a piece (no stepping) and recompute kinematics and contacts."""
    qadr, dadr = _adr(world, name)
    if xy is not None:
        world.data.qpos[qadr : qadr + 2] = xy
    if z is not None:
        world.data.qpos[qadr + 2] = z
    world.data.qpos[qadr + 2] += dz
    if quat is not None:
        world.data.qpos[qadr + 3 : qadr + 7] = quat
    world.data.qvel[dadr : dadr + 6] = 0
    if vel is not None:
        world.data.qvel[dadr : dadr + 3] = vel
    if spin is not None:
        world.data.qvel[dadr + 3 : dadr + 6] = spin
    import mujoco

    mujoco.mj_forward(world.model, world.data)


def _approach(world, tracker) -> None:
    """Put the gripper tip on the piece's source, as a real approach would."""
    sx, sy = tracker.source
    world.data.site_xpos[world.gripper_site] = (sx, sy, tracker.start_z + 0.02)
    tracker.update()


def _transfer(world, tracker, name: str, xy, *, z=None, lift: float = 0.05, quat=None, **rest) -> None:
    """Approach, lift `lift` m, then set the piece down at `xy` (at its start height unless `z`)."""
    start_z = _z(world, name)
    _approach(world, tracker)
    _set(world, name, dz=lift)
    tracker.update()
    _set(world, name, xy=xy, z=start_z + lift, quat=quat)  # carried over the target
    tracker.update()
    _set(world, name, xy=xy, z=start_z if z is None else z, quat=quat, **rest)
    tracker.update()


def _rest_z(world, name: str, surface: float) -> float:
    """z of the piece's centre when standing on a surface `surface` m high."""
    return surface + float(world.model.geom_size[world.model.body_geomadr[world.body_id(name)]][1])


LYING = (np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0)  # rolled 90 degrees about x


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
def test_tracker_lifting_without_the_gripper_nearby_is_still_approach(world) -> None:
    from lelab.sim.judge import TransferTracker

    tracker = TransferTracker(world, ("e2", "e4"))
    _set(world, "p_e2_P", dz=0.05)
    tracker.update()
    _set(world, "p_e2_P", xy=world.cell.location_xy("e4"), dz=-0.05)
    tracker.update()
    assert tracker.outcome().failure == "approach"


@requires_mujoco
def test_tracker_lifted_and_placed_piece_succeeds(world) -> None:
    from lelab.sim.judge import TransferTracker

    tracker = TransferTracker(world, ("e2", "e4"))
    _transfer(world, tracker, "p_e2_P", world.cell.location_xy("e4"))
    result = tracker.outcome()
    assert result.success, result
    assert result.max_lift == pytest.approx(0.05, abs=1e-3)
    assert result.final_error < 1e-3


@requires_mujoco
def test_tracker_still_moving_piece_is_not_a_success(world) -> None:
    from lelab.sim.judge import TransferTracker

    for kwargs in ({"vel": (0.05, 0, 0)}, {"spin": (0, 0, 2.0)}):
        tracker = TransferTracker(world, ("e2", "e4"))
        _transfer(world, tracker, "p_e2_P", world.cell.location_xy("e4"), **kwargs)
        assert tracker.outcome().failure == "place", kwargs
        _set(world, "p_e2_P", xy=world.cell.location_xy("e2"))  # reset for the next round


@requires_mujoco
def test_tracker_piece_held_above_the_target_is_not_a_success(world) -> None:
    from lelab.sim.judge import TransferTracker

    tracker = TransferTracker(world, ("e2", "e4"))
    _transfer(world, tracker, "p_e2_P", world.cell.location_xy("e4"), z=_z(world, "p_e2_P") + 0.05)
    assert tracker.outcome().failure == "place"


@requires_mujoco
def test_tracker_piece_sunk_below_the_surface_is_not_a_success(world) -> None:
    from lelab.sim.judge import TransferTracker

    tracker = TransferTracker(world, ("e2", "e4"))
    _transfer(world, tracker, "p_e2_P", world.cell.location_xy("e4"), z=_z(world, "p_e2_P") - 0.008)
    assert tracker.outcome().failure == "place"


@requires_mujoco
def test_tracker_bin_needs_the_piece_inside_and_below_the_wall_top(world) -> None:
    from lelab.sim.judge import TransferTracker

    cell = world.cell
    inside = _rest_z(world, "p_e2_P", 0.003)  # on the bin floor
    for z, expected in ((inside, None), (cell.bin_wall_height + 0.015, "place")):
        tracker = TransferTracker(world, ("e2", "bin"))
        _transfer(world, tracker, "p_e2_P", cell.bin_center, z=z)
        assert tracker.outcome().failure == expected, z
        _set(world, "p_e2_P", xy=cell.location_xy("e2"), z=_rest_z(world, "p_e2_P", cell.board_thickness))


@requires_mujoco
def test_tracker_parking_pad_source_succeeds(world) -> None:
    from lelab.sim.judge import TransferTracker

    assert world.body_at("park_white") == "p_park_white_Q"
    tracker = TransferTracker(world, ("park_white", "e4"))
    z = _rest_z(world, "p_park_white_Q", world.cell.board_thickness)
    _transfer(world, tracker, "p_park_white_Q", world.cell.location_xy("e4"), z=z)
    result = tracker.outcome()
    assert result.success, result


@requires_mujoco
def test_tracker_drop_is_a_piece_lifted_then_left_away_from_the_target(world) -> None:
    from lelab.sim.judge import TransferTracker

    tracker = TransferTracker(world, ("e2", "e6"))
    _transfer(world, tracker, "p_e2_P", world.cell.location_xy("e3"))
    result = tracker.outcome()
    assert not result.success and result.failure == "drop"
    assert result.max_lift >= 0.02


@requires_mujoco
def test_tracker_place_is_a_release_close_to_but_off_the_target(world) -> None:
    from lelab.sim.judge import TransferTracker

    x, y = world.cell.location_xy("e4")
    tracker = TransferTracker(world, ("e2", "e4"))
    _transfer(world, tracker, "p_e2_P", (x, y + 0.012))  # 12 mm off, beyond 35 % of 28 mm
    assert tracker.outcome().failure == "place"


@requires_mujoco
def test_tracker_topple_is_the_moved_piece_lying_on_the_target(world) -> None:
    from lelab.sim.judge import TransferTracker

    tracker = TransferTracker(world, ("e2", "e4"))
    radius = world.cell.pieces["p"].radius
    z = world.cell.board_thickness + radius
    _transfer(world, tracker, "p_e2_P", world.cell.location_xy("e4"), z=z, quat=LYING)
    assert tracker.outcome().failure == "topple"


@requires_mujoco
def test_tracker_neighbour_displacement_threshold(world) -> None:
    from lelab.sim.judge import TransferTracker

    x, y = world.cell.location_xy("d2")
    for shift, expected in ((0.004, None), (0.006, "disturbed")):
        tracker = TransferTracker(world, ("e2", "e4"))
        _transfer(world, tracker, "p_e2_P", world.cell.location_xy("e4"))
        _set(world, "p_d2_P", xy=(x + shift, y))
        assert tracker.outcome().failure == expected, shift
        _set(world, "p_d2_P", xy=(x, y))
        _set(world, "p_e2_P", xy=world.cell.location_xy("e2"))


@requires_mujoco
def test_tracker_neighbour_displacement_counts_the_peak_and_height(world) -> None:
    from lelab.sim.judge import TransferTracker

    x, y = world.cell.location_xy("d2")
    tracker = TransferTracker(world, ("e2", "e4"))
    _approach(world, tracker)
    _set(world, "p_e2_P", dz=0.05)
    tracker.update()
    _set(world, "p_d2_P", xy=(x + 0.02, y))  # knocked away mid-transfer ...
    tracker.update()
    _set(world, "p_d2_P", xy=(x, y))  # ... and back by the end
    _set(world, "p_e2_P", xy=world.cell.location_xy("e4"), dz=-0.05)
    tracker.update()
    assert tracker.outcome().failure == "disturbed"

    lifted = TransferTracker(world, ("e2", "e4"))
    _transfer(world, lifted, "p_e2_P", world.cell.location_xy("e4"))
    _set(world, "p_d2_P", dz=0.008)  # 8 mm straight up
    assert lifted.outcome().failure == "disturbed"


@requires_mujoco
def test_tracker_neighbour_tipping_is_disturbed(world) -> None:
    from lelab.sim.judge import TransferTracker

    tracker = TransferTracker(world, ("e2", "e4"))
    _transfer(world, tracker, "p_e2_P", world.cell.location_xy("e4"))
    _set(world, "p_d2_P", quat=LYING)
    assert tracker.outcome().failure == "disturbed"


@requires_mujoco
def test_tracker_ignores_pieces_that_started_in_the_bin(world) -> None:
    from lelab.sim.judge import TransferTracker

    tracker = TransferTracker(world, ("e2", "e4"))
    _transfer(world, tracker, "p_e2_P", world.cell.location_xy("e4"))
    _set(world, "p_bin_p_0", dz=0.01, quat=LYING)  # knocked about by a drop into the bin
    assert world.data.xmat[world.body_id("p_bin_p_0")].reshape(3, 3)[2, 2] < 0.5
    tracker.update()
    assert tracker.outcome().success


@requires_mujoco
def test_tracker_detects_arm_contact(world) -> None:
    from lelab.sim.judge import TransferTracker

    tracker = TransferTracker(world, ("e2", "e4"))
    assert not tracker._touching_arm()
    jaw = world.data.geom_xpos[world.model.geom("fixed_jaw_box1").id].copy()  # a collision box of the arm
    _set(world, "p_e2_P", xy=jaw[:2], z=jaw[2])
    assert tracker._touching_arm()


@requires_mujoco
def test_tracker_requires_a_piece_on_the_source(world) -> None:
    from lelab.sim.judge import TransferTracker

    with pytest.raises(ValueError, match="No piece on e4"):
        TransferTracker(world, ("e4", "e5"))


# Arm pose that puts the closed gripper's jaws around a pawn standing on e4 (found by `expert.solve_ik` with
# the default SimCell, the pawn's centre between the closed jaws; the jaws then overlap the pawn).
JAWS_ON_E4 = (3.97, 9.36, 12.27, 68.37, 0.0, 2.0)


@requires_mujoco
def test_tracker_piece_on_target_but_still_in_the_jaws_is_not_a_success(world) -> None:
    from lelab.sim.judge import TransferTracker

    tracker = TransferTracker(world, ("e2", "e4"))
    _transfer(world, tracker, "p_e2_P", world.cell.location_xy("e4"))
    assert tracker.outcome().success  # released and clear: the control
    world.reset_joints(JAWS_ON_E4)
    assert tracker._touching_arm()
    assert tracker.outcome().failure == "place"


@requires_mujoco
def test_tracker_captured_piece_that_escapes_the_bin_is_disturbed(world) -> None:
    from lelab.sim.judge import TransferTracker

    tracker = TransferTracker(world, ("e2", "e4"))
    _transfer(world, tracker, "p_e2_P", world.cell.location_xy("e4"))
    x, y = world.cell.location_xy("e4")
    _set(world, "p_bin_p_0", xy=(x + 0.03, y), z=0.019)  # knocked out onto the board
    tracker.update()
    assert tracker.outcome().failure == "disturbed"
