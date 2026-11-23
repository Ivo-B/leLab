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

"""Scripted pick-and-place in simulation, the source of the sim demonstrations.

IK is damped least squares on the MuJoCo model for the arm's `gripperframe` site.
The site's local x axis is the approach direction (found in the 2026-11-23
prototype) and its local z axis the closing direction: the fixed jaw sits
`FIXED_JAW_FACE` along -z, the moving jaw swings open towards +z. The approach
tilts outward (tips away from the base) so that both the nearest and the farthest
squares stay reachable, and the closing direction is turned, square by square,
to wherever the open jaws clear the neighbouring pieces.

The plan is open loop: `plan_transfer` reads the scene (piece positions, captured
pieces in the bin), rehearses its candidate plans on a copy of it, and returns
every action, so a replay of the plan reproduces `run_expert` exactly.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from ..chess.game import BIN, PARK_BLACK, PARK_WHITE, Transfer
from .judge import FAILURES, Outcome, TransferTracker
from .world import GRIPPER_CLOSED, PARK, ChessWorld

APPROACH_AXIS = 0
PARK_ROLL = float(np.deg2rad(PARK[4]))
CLOSING_AXIS = 2
# Measured on the Menagerie SO-101: the fixed jaw's inner face lies 19.5 mm along the site's -z.
FIXED_JAW_FACE = -0.0195
# Horizontal reach of the gripper's housing beyond a held piece's side, for clearing captured pieces.
BIN_GRIPPER_REACH = 0.03
# A plan whose IK leaves the planned path by more than this anywhere is not used.
PATH_TOL = 0.002
IK_POS_TOL = 1e-3
IK_ROT_TOL = 0.02
NEAR_MISS = 0.003


@dataclass
class ExpertParams:
    tilt_near_deg: float = 20.0  # outward approach tilt below `tilt_rank` (and on the parking pads)
    tilt_far_deg: float = 30.0  # from `tilt_rank` on, and over the bin
    tilt_rank: int = 7
    hover: float = 0.025  # height of the grasp point above the piece's top where the descent starts
    # There the site is this far off the piece's axis (nearer than once closed), so it passes over the piece.
    hover_site_offset: float = 0.008
    travel: float = 0.012  # clearance of the carried piece over the tallest piece on the board
    grasp_frac: float = 0.45  # grasp height on the piece, as a fraction of its height
    # The open moving jaw clears the piece by this much, and no more, to miss the neighbours.
    jaw_gap: float = 0.003
    grip_clearance: float = 0.001  # gap between the piece and the fixed jaw before closing
    release_above: float = 0.002  # height of the carried piece's base over the surface when it is let go
    open_steps: int = 20
    close_steps: int = 15
    dwell_steps: int = 5  # hold still before closing and before opening, so the arm catches up
    speed: float = 0.25  # m/s of the gripper site between waypoints
    slow_speed: float = 0.08  # m/s while descending onto or lifting off a square
    settle_steps: int = 10
    # Costs that rank grasps against the intrusion of `_Planner.cost`: per rad the piece is tilted
    # when let go, per rad the wrist rolls between grasp and release, per rad off `tilt_for`.
    carry_tilt_cost: float = 0.12
    roll_cost: float = 0.1
    tilt_cost: float = 0.005
    back_off: float = 0.002  # sideways step off the released piece before rising
    closing_step_deg: float = 15.0  # resolution of the search over closing directions
    rehearsals: int = 8  # cheapest pairs of grasp and release to rehearse before giving up
    refine_steps: int = 2  # re-aims of the release after a rehearsed landing off the square
    max_shift: float = 0.015  # largest correction of the release per re-aim
    tilts_deg: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0, 35.0)  # approach tilts the search may use


def tilt_for(cell, location: str, params: ExpertParams) -> float:
    """Preferred outward approach tilt (rad) over `location`; the planner may pick another."""
    if location == BIN:
        return float(np.deg2rad(params.tilt_far_deg))
    if location in (PARK_WHITE, PARK_BLACK):
        return float(np.deg2rad(params.tilt_near_deg))
    near = int(location[1]) < params.tilt_rank
    return float(np.deg2rad(params.tilt_near_deg if near else params.tilt_far_deg))


def _cross(a, b) -> np.ndarray:
    """`np.cross` for two 3-vectors, without its overhead (IK calls it in its inner loop)."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def _radial(world: ChessWorld, data: mujoco.MjData, lift: int | None = None) -> np.ndarray:
    """Horizontal unit vector along which the arm currently reaches (normal to the shoulder_lift axis)."""
    axis = data.xaxis[world.model.joint("shoulder_lift").id if lift is None else lift]
    radial = np.array([axis[1], -axis[0], 0.0])
    return radial / max(1e-9, float(np.sqrt(radial @ radial)))


def approach_frame(radial, tilt: float, closing: float | None):
    """Wanted approach axis, and the closing axis turned `closing` rad from `radial` (or None)."""
    approach = np.array([radial[0] * np.sin(tilt), radial[1] * np.sin(tilt), -np.cos(tilt)])
    if closing is None:
        return approach, None
    c, s = np.cos(closing), np.sin(closing)
    h = np.array([c * radial[0] - s * radial[1], s * radial[0] + c * radial[1], 0.0])
    h -= (h @ approach) * approach
    return approach, h / float(np.sqrt(h @ h))


def solve_ik(
    world: ChessWorld,
    target,
    tilt: float,
    q0,
    iters: int = 300,
    *,
    closing: float | None = None,
    offset=None,
):
    """Joint angles (rad) that put the gripper site at `target`, approach tilted `tilt` outward.

    Without `closing` the wrist roll stays at `q0`'s; with it, the closing axis is also turned
    `closing` rad (counter-clockwise from above) from the radial direction. `offset` is a point in
    the site frame to place at `target` instead of the site itself.
    """
    model = world.model
    data = mujoco.MjData(model)
    data.qpos[:] = world.data.qpos
    target = np.asarray(target, dtype=np.float64)
    offset = np.zeros(3) if offset is None else np.asarray(offset, dtype=np.float64)
    q = np.array(q0, dtype=np.float64)
    pan_joint = model.joint("shoulder_pan").id
    pan = world.data.xanchor[pan_joint][:2]
    # Aim the pan at the target, so IK starts in the right basin.
    if np.linalg.norm(target[:2] - pan) > 1e-6:
        sign = 1.0 if world.data.xaxis[pan_joint][2] > 0 else -1.0  # +1: positive pan turns counter-clockwise
        q[0] = sign * np.arctan2(target[1] - pan[1], target[0] - pan[0])
    n = 5 if closing is not None else 4
    cols = world.dofadr[:n]
    jp = np.zeros((3, model.nv))
    jr = np.zeros((3, model.nv))
    site = world.gripper_site
    body = int(model.site_bodyid[site])
    lift = model.joint("shoulder_lift").id
    pos_err = rot_err = np.inf
    for _ in range(iters):
        data.qpos[world.qadr[:5]] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        frame = data.site_xmat[site].reshape(3, 3)
        point = data.site_xpos[site] + frame @ offset
        approach, close = approach_frame(_radial(world, data, lift), tilt, closing)
        e_pos = target - point
        e_rot = _cross(frame[:, APPROACH_AXIS], approach)
        if close is not None:
            e_rot = e_rot + _cross(frame[:, CLOSING_AXIS], close)
        pos_err, rot_err = float(np.sqrt(e_pos @ e_pos)), float(np.sqrt(e_rot @ e_rot))
        if pos_err < 2e-4 and rot_err < 0.005:
            break
        mujoco.mj_jac(model, data, jp, jr, point, body)
        jac = np.vstack([jp[:, cols], 0.3 * jr[:, cols]])
        err = np.concatenate([e_pos, 0.3 * e_rot])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err)
        q[:n] = np.clip(q[:n] + dq, world.lo[:n], world.hi[:n])
    return q, pos_err, rot_err


# -- the scene, read once --------------------------------------------------------------------------


@dataclass
class _Piece:
    body: int
    radius: float
    height: float
    base: np.ndarray  # centre of its base


def _size(world: ChessWorld, name: str):
    """Collision size of the piece body `name` (`p_<location>_<symbol>`, with an index in the bin)."""
    symbol = name.split("_")[-2] if name.startswith(f"p_{BIN}_") else name.split("_")[-1]
    return world.cell.pieces[symbol.lower()]


def _piece(world: ChessWorld, location: str) -> _Piece:
    name = world.body_at(location)
    if name is None:
        raise ValueError(f"No piece on {location}")
    size = _size(world, name)
    body = world.body_id(name)
    base = world.data.xpos[body].copy()
    base[2] -= size.height / 2
    return _Piece(body, size.radius, size.height, base)


def _surface_z(world: ChessWorld, location: str) -> float:
    geom = world.model.geom(f"{location}_pad" if location in (PARK_WHITE, PARK_BLACK) else "board")
    return float(geom.pos[2] + geom.size[2])


def _bin_drop(world: ChessWorld, piece: _Piece) -> np.ndarray:
    """Base centre to let `piece` go from over the bin: above the freest spot of the floor, clear of the walls.

    The drop height clears the wall and the captured pieces within reach of the gripper around that spot.
    """
    cell, data = world.cell, world.data
    (x0, y0), (x1, y1) = cell.location_rect(BIN)
    inset = 0.004 + piece.radius
    captured = []
    for name, i in world.piece_bodies().items():
        x, y, z = data.xpos[i]
        if x0 <= x <= x1 and y0 <= y <= y1 and i != piece.body:
            size = _size(world, name)
            captured.append((x, y, size.radius, z + size.height / 2))
    best, best_gap = np.array(cell.bin_center), -np.inf
    for x in np.linspace(x0 + inset, x1 - inset, 9):
        for y in np.linspace(y0 + inset, y1 - inset, 9):
            gap = min((np.hypot(x - cx, y - cy) - r for cx, cy, r, _ in captured), default=1.0)
            gap -= 1e-3 * np.hypot(x - cell.bin_center[0], y - cell.bin_center[1])  # prefer the middle
            if gap > best_gap:
                best, best_gap = np.array([x, y]), gap
    reach = piece.radius + BIN_GRIPPER_REACH
    tops = [top for cx, cy, r, top in captured if np.hypot(best[0] - cx, best[1] - cy) < reach + r]
    return np.array([best[0], best[1], max([cell.bin_wall_height, *tops]) + 0.01])


class _Planner:
    """IK and clearance checks against a frozen copy of the scene."""

    def __init__(self, world: ChessWorld, params: ExpertParams, moved: int) -> None:
        self.world, self.params, self.moved = world, params, moved
        self.model = copy.copy(world.model)
        model = self.model
        arm = {
            i
            for i in range(1, model.nbody)
            if not model.body(i).name.startswith("p_") and model.body(i).name != "bin"
        }
        self.arm_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] in arm}
        model.geom_margin[list(self.arm_geoms)] = NEAR_MISS  # report near misses as well as contacts
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = world.data.qpos
        self._jaw = self._jaw_table()

    def _jaw_table(self) -> tuple[np.ndarray, np.ndarray]:
        """Gripper values and where the moving jaw's inner face then sits along the site's z."""
        model, data, world = self.model, self.data, self.world
        jaw = model.geom("moving_jaw_box2").id
        grips = np.arange(0.0, 40.0, 0.5)
        faces = []
        for grip in grips:
            data.qpos[world.qadr[5]] = world.to_ctrl([0, 0, 0, 0, 0, grip])[5]
            mujoco.mj_kinematics(model, data)
            frame = data.site_xmat[world.gripper_site].reshape(3, 3)
            local = frame.T @ (data.geom_xpos[jaw] - data.site_xpos[world.gripper_site])
            faces.append(local[CLOSING_AXIS] - model.geom_size[jaw][0])
        return grips, np.array(faces)

    def grip_for(self, face: float) -> float:
        """Smallest gripper value that puts the moving jaw's inner face at `face` (site z) or beyond."""
        grips, faces = self._jaw
        return float(np.interp(face, faces, grips))

    def cost(self, q5, grip: float, moved_near_misses: bool) -> float:
        """Intrusion of the arm into pieces, board and bin: penetration weighs 50x a near miss (< 3 mm)."""
        model, data, world = self.model, self.data, self.world
        data.qpos[world.qadr[:5]] = q5
        data.qpos[world.qadr[5]] = world.to_ctrl([0, 0, 0, 0, 0, grip])[5]
        mujoco.mj_fwdPosition(model, data)
        total = 0.0
        for contact in data.contact[: data.ncon]:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if (g1 in self.arm_geoms) == (g2 in self.arm_geoms):
                continue
            other = int(model.geom_bodyid[g2 if g1 in self.arm_geoms else g1])
            dist = float(contact.dist)
            if other == self.moved and not moved_near_misses:
                total += 10 * max(0.0, -dist)
                continue
            total += 0.2 * (NEAR_MISS - dist) + 10 * max(0.0, -dist)
        return total

    def end(self, points, offset, grips, moved_near_misses: bool, tilt: float, q_seed) -> list[_Option]:
        """The closings at which IK reaches every point, with their cost."""
        params, world = self.params, self.world
        step = np.deg2rad(params.closing_step_deg)
        options = []
        for k in range(int(round(2 * np.pi / step))):
            closing = k * step
            q, qs, total = np.array(q_seed, dtype=np.float64), [], 0.0
            for point, grip in zip(points, grips, strict=True):
                q, pos_err, rot_err = solve_ik(world, point, tilt, q, closing=closing, offset=offset)
                if pos_err > IK_POS_TOL or rot_err > IK_ROT_TOL:
                    break
                total += self.cost(q, grip, moved_near_misses)
                qs.append(q)
            else:
                frame = self.data.site_xmat[world.gripper_site].reshape(3, 3).copy()
                options.append(_Option(tilt, closing, total, qs, frame))
        return options


@dataclass
class _Option:
    tilt: float
    closing: float
    cost: float
    qs: list  # joint angles at each point of the descent
    frame: np.ndarray  # gripper site orientation at the lowest point


def _hold_offset(piece: _Piece, clearance: float) -> np.ndarray:
    """Grasp point in the site frame: the piece's axis, `clearance` off the fixed jaw."""
    return np.array([0.0, 0.0, FIXED_JAW_FACE + piece.radius + clearance])


def _ranked(src: list[_Option], dst: list[_Option], params: ExpertParams) -> list[tuple[_Option, _Option]]:
    """Pairs of grasp and release, cheapest first: intrusions, the tilt of the piece when let go, wrist roll.

    The piece keeps its orientation in the hand, so it lands upright only when the release frame
    turns the grasp frame about the vertical; anything else tilts it.
    """
    if not src or not dst:
        return []
    held_up = np.array([o.frame[2] for o in src])  # world up, in the hand frame at the grasp
    release_up = np.array([o.frame[2] for o in dst])  # world up, in the hand frame at the release
    tilt = np.arccos(np.clip(np.einsum("ik,jk->ij", release_up, held_up), -1.0, 1.0))  # (dst, src)
    roll_src = np.array([o.qs[-1][4] for o in src])[None, :]
    roll_dst = np.array([o.qs[-1][4] for o in dst])[:, None]
    roll = np.abs(roll_src - PARK_ROLL) + np.abs(roll_dst - roll_src)  # wrist travel: park, grasp, release
    total = (
        np.array([o.cost for o in dst])[:, None]
        + np.array([o.cost for o in src])[None, :]
        + params.carry_tilt_cost * tilt
        + params.roll_cost * roll
    )
    order = np.argsort(total, axis=None, kind="stable")[: 4 * params.rehearsals]
    return [(src[j], dst[i]) for i, j in zip(*np.unravel_index(order, total.shape), strict=True)]


@dataclass
class _Waypoints:
    piece: _Piece
    grasp: np.ndarray  # grasp point: on the piece's axis, `grasp_frac` up it
    pick_top: np.ndarray
    pick_lift: np.ndarray
    release: np.ndarray
    place_top: np.ndarray
    open_offset: np.ndarray  # grasp point in the site frame with the jaws open, ...
    hover_offset: np.ndarray  # ... above the piece, where the gripper site is right over it ...
    hold_offset: np.ndarray  # ... and closed on it
    grip_open: float
    grip_held: float


def _waypoints(world: ChessWorld, transfer: Transfer, params: ExpertParams, planner: _Planner) -> _Waypoints:
    source, target = transfer
    cell = world.cell
    piece = _piece(world, source)
    lift = params.grasp_frac * piece.height
    grasp = piece.base + np.array([0.0, 0.0, lift])
    tallest = cell.board_thickness + max(p.height for p in cell.pieces.values())
    if target == BIN:
        drop = _bin_drop(world, piece)
    else:
        drop = np.array([*cell.location_xy(target), _surface_z(world, target) + params.release_above])
    release = drop + np.array([0.0, 0.0, lift])
    travel_z = max(tallest, drop[2] if target == BIN else 0.0) + params.travel + lift
    pick_top = grasp + np.array([0.0, 0.0, piece.height - lift + params.hover])
    far_edge = FIXED_JAW_FACE + params.grip_clearance + 2 * piece.radius
    return _Waypoints(
        piece=piece,
        grasp=grasp,
        pick_top=pick_top,
        pick_lift=np.array([grasp[0], grasp[1], max(travel_z, pick_top[2])]),
        release=release,
        place_top=np.array([release[0], release[1], max(travel_z, release[2] + params.hover)]),
        open_offset=_hold_offset(piece, params.grip_clearance),
        hover_offset=np.array([0.0, 0.0, -params.hover_site_offset]),
        hold_offset=_hold_offset(piece, 0.0),
        grip_open=planner.grip_for(far_edge + params.jaw_gap),
        grip_held=planner.grip_for(far_edge - params.grip_clearance),
    )


def _trajectory(
    world: ChessWorld, way: _Waypoints, params: ExpertParams, pick: _Option, place: _Option, shift
) -> tuple[list[np.ndarray], float]:
    """Actions for one grasp and release, with the release moved by `shift` (x, y), and the largest
    distance (m) by which IK left the planned path."""
    shift = np.array([shift[0], shift[1], 0.0])
    release, place_top = way.release + shift, way.place_top + shift
    actions: list[np.ndarray] = []
    state = {
        "q": np.deg2rad(np.asarray(world.joint_state()[:5], dtype=np.float64)),
        "grip": float(world.joint_state()[5]),
        "path_err": 0.0,
    }

    def joint_move(goal, steps: int, grip_goal: float | None = None) -> None:
        start, g0 = state["q"].copy(), state["grip"]
        g1 = g0 if grip_goal is None else grip_goal
        for i in range(1, steps + 1):
            s = 0.5 - 0.5 * np.cos(np.pi * i / steps)  # ease in and out
            actions.append(np.concatenate([(1 - s) * start + s * goal, [(1 - s) * g0 + s * g1]]))
        state["q"], state["grip"] = np.array(goal, dtype=np.float64), g1

    def hold(steps: int, grip_goal: float | None = None) -> None:
        joint_move(state["q"], steps, grip_goal)

    def line(a, b, offsets, roll_end: float, speed: float, tilts) -> None:
        """Straight line of the grasp point from `a` to `b`; roll, tilt and offset turn evenly to the end's."""
        a, b = np.asarray(a), np.asarray(b)
        (o0, o1), (t0, t1) = offsets, tilts
        steps = max(8, int(np.ceil(np.linalg.norm(b - a) / speed * 30)))
        q, roll0 = state["q"].copy(), state["q"][4]
        for i in range(1, steps + 1):
            s = 0.5 - 0.5 * np.cos(np.pi * i / steps)
            q[4] = (1 - s) * roll0 + s * roll_end
            q, pos_err, _ = solve_ik(
                world, a + s * (b - a), (1 - s) * t0 + s * t1, q, offset=(1 - s) * o0 + s * o1
            )
            state["path_err"] = max(state["path_err"], pos_err)
            actions.append(np.concatenate([q, [state["grip"]]]))
        state["q"] = q

    at_src, at_dst = (pick.tilt, pick.tilt), (place.tilt, place.tilt)
    held = (way.hold_offset, way.hold_offset)
    q_hover, hover_err, _ = solve_ik(
        world, way.pick_top, pick.tilt, pick.qs[0], closing=pick.closing, offset=way.hover_offset
    )
    state["path_err"] = hover_err
    reach = float(np.linalg.norm(world.data.site_xpos[world.gripper_site] - way.pick_top))
    joint_move(q_hover, max(15, int(np.ceil(reach / params.speed * 30))), way.grip_open)
    line(
        way.pick_top,
        way.grasp,
        (way.hover_offset, way.open_offset),
        pick.qs[-1][4],
        params.slow_speed,
        at_src,
    )
    hold(params.dwell_steps)
    hold(params.close_steps, GRIPPER_CLOSED)
    line(way.grasp, way.pick_lift, held, pick.qs[-1][4], params.slow_speed, at_src)
    line(way.pick_lift, place_top, held, place.qs[0][4], params.speed, (pick.tilt, place.tilt))
    line(place_top, release, held, place.qs[-1][4], params.slow_speed, at_dst)
    hold(params.dwell_steps)
    hold(params.open_steps, way.grip_open)
    hold(params.dwell_steps)
    _, close_axis = approach_frame(_radial_at(world, state["q"]), place.tilt, place.closing)
    # Off the fixed jaw before rising, so it does not drag the piece.
    backed = release + params.back_off * close_axis
    line(release, backed, held, state["q"][4], params.slow_speed, at_dst)
    line(
        backed,
        backed + np.array([0.0, 0.0, place_top[2] - release[2]]),
        held,
        state["q"][4],
        params.slow_speed,
        at_dst,
    )
    joint_move(np.deg2rad(np.asarray(PARK[:5], dtype=np.float64)), 45, PARK[5])
    hold(params.settle_steps)
    out = np.array(actions, dtype=np.float64)
    out[:, :5] = np.rad2deg(out[:, :5])
    return list(out.astype(np.float32)), float(state["path_err"])


def _rehearse(world: ChessWorld, transfer: Transfer, actions) -> tuple[Outcome, np.ndarray]:
    """Run `actions` on a copy of the scene: the outcome, and where the piece came to rest."""
    sim = copy.copy(world)
    sim.data, sim.renderer = copy.copy(world.data), None
    tracker = TransferTracker(sim, transfer)
    for action in actions:
        sim.step(action)
        tracker.update()
    return tracker.outcome(), sim.data.xpos[tracker.body][:2].copy()


def plan_transfer(world: ChessWorld, transfer: Transfer, params: ExpertParams) -> list[np.ndarray]:
    """Actions for `transfer`, rehearsed on a copy of the scene before they are returned.

    The rehearsal catches what the clearance check cannot: the piece creeps along the jaws while
    carried and shifts when let go. A landing off the square moves the release by the miss and
    tries again; a failed pair of grasp and release gives way to the next cheapest one, and a pair
    whose IK leaves the planned path by more than `PATH_TOL` anywhere is skipped unrehearsed.

    If every rehearsal fails, the plan that got furthest (by `judge.FAILURES` stage) is returned,
    so `run_expert` reports that failure. Raises ValueError when no pair is reachable at all.
    """
    source, target = transfer
    planner = _Planner(world, params, _piece(world, source).body)
    way = _waypoints(world, transfer, params, planner)
    q_start = np.deg2rad(np.asarray(world.joint_state()[:5], dtype=np.float64))
    descent = (0.0, 0.5, 0.8, 1.0)
    pick_points = [way.pick_top + f * (way.grasp - way.pick_top) for f in descent]
    place_points = [way.place_top + f * (way.release - way.place_top) for f in descent]
    src, dst = [], []
    for tilt in (float(np.deg2rad(t)) for t in params.tilts_deg):
        src += planner.end(pick_points, way.open_offset, [way.grip_open] * 4, False, tilt, q_start)
        dst += planner.end(
            place_points, way.hold_offset, [way.grip_held] * 3 + [way.grip_open], True, tilt, q_start
        )
    for options, location in ((src, source), (dst, target)):
        for option in options:
            option.cost += params.tilt_cost * abs(option.tilt - tilt_for(world.cell, location, params))
    pairs = _ranked(src, dst, params)
    if not pairs:
        raise ValueError(f"No reachable grasp and release for {transfer}")
    goal = np.array(world.cell.location_xy(target)) if target != BIN else way.release[:2]
    best, rehearsed = None, 0
    for pick, place in pairs:
        if rehearsed == params.rehearsals:
            break
        shift = np.zeros(2)
        for attempt in range(params.refine_steps + 1):
            actions, path_err = _trajectory(world, way, params, pick, place, shift)
            if path_err > PATH_TOL:
                break
            rehearsed += attempt == 0
            outcome, rest = _rehearse(world, transfer, actions)
            if outcome.success:
                return actions
            stage = FAILURES.index(outcome.failure) if outcome.failure in FAILURES else -1
            if best is None or stage > best[0]:
                best = (stage, actions)
            if target == BIN or outcome.failure not in ("place", "topple") or attempt == params.refine_steps:
                break
            shift -= np.clip(rest - goal, -params.max_shift, params.max_shift)
    if best is None:
        raise ValueError(f"No grasp and release for {transfer} stays on its planned path")
    return best[1]


def _radial_at(world: ChessWorld, q5) -> np.ndarray:
    data = mujoco.MjData(world.model)
    data.qpos[:] = world.data.qpos
    data.qpos[world.qadr[:5]] = q5
    mujoco.mj_kinematics(world.model, data)
    return _radial(world, data)


def run_expert(
    world: ChessWorld,
    transfer: Transfer,
    params: ExpertParams,
    on_step: Callable[[np.ndarray], None] | None = None,
) -> Outcome:
    """Execute the scripted transfer. `on_step(action)` runs before each step (recording)."""
    tracker = TransferTracker(world, transfer)
    for action in plan_transfer(world, transfer, params):
        if on_step is not None:
            on_step(action)
        world.step(action)
        tracker.update()
    return tracker.outcome()
