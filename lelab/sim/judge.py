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

"""Did a simulated transfer succeed, and if not, where did it fail first."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..chess.game import BIN, PARK_BLACK, PARK_WHITE, Transfer

if TYPE_CHECKING:
    from .world import ChessWorld

FAILURES = ("approach", "grasp", "drop", "place", "topple", "disturbed")
LIFT_M = 0.02
APPROACH_M = 0.01  # horizontal distance from the source within which the tip counts as having reached it
# Highest the tip may be above the piece's start height and still count as reached: about a piece
# height plus the clearance of the open jaws, so hovering high over the square does not count.
APPROACH_HEIGHT_M = 0.05
PLACE_TOLERANCE = 0.35  # of a square
UPRIGHT_COS = 0.9
DISTURB_M = 0.005  # a neighbour that ever strayed further than this (3D) was disturbed
REST_LINEAR_MPS = 0.01  # at rest: the piece's linear speed is below this ...
REST_ANGULAR_RADPS = 0.5  # ... and its angular speed below this
SURFACE_TOL_M = 0.004  # the piece's centre height must be within this of its resting height


@dataclass(frozen=True)
class Outcome:
    success: bool
    failure: str | None
    max_lift: float
    final_error: float


def classify(
    *,
    reached: bool,
    max_lift: float,
    final_xy,
    final_up: float,
    target_rect,
    others_moved: int,
    released_near_target: bool,
    square_size: float,
    is_bin: bool,
    at_rest: bool = True,
    on_surface: bool = True,
) -> Outcome:
    """First failure among `FAILURES`, or success.

    `at_rest` is false while the piece still moves or the arm still touches it; `on_surface` is
    false when the piece is held above (or sunk below) the surface of the target.
    """
    (x0, y0), (x1, y1) = target_rect
    centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    error = float(np.linalg.norm(np.asarray(final_xy) - centre))
    inside = x0 <= final_xy[0] <= x1 and y0 <= final_xy[1] <= y1
    on_target = inside if is_bin else error <= PLACE_TOLERANCE * square_size

    def fail(name: str) -> Outcome:
        return Outcome(False, name, max_lift, error)

    if not reached:
        return fail("approach")
    if max_lift < LIFT_M:
        return fail("grasp")
    if not on_target:
        return fail("place" if released_near_target else "drop")
    if not is_bin and final_up < UPRIGHT_COS:
        return fail("topple")
    if not (at_rest and on_surface):
        return fail("place")
    if others_moved:
        return fail("disturbed")
    return Outcome(True, None, max_lift, error)


class TransferTracker:
    """Follows the moved piece and its neighbours through one transfer."""

    def __init__(self, world: ChessWorld, transfer: Transfer) -> None:
        self.world = world
        self.transfer = transfer
        name = world.body_at(transfer[0])
        if name is None:
            raise ValueError(f"No piece on {transfer[0]}")
        model, data = world.model, world.data
        self.body = world.body_id(name)
        start = data.xpos[self.body].copy()
        self.start_z = float(start[2])
        self.source = start[:2]
        joint = int(model.body_jntadr[self.body])
        self._dof = int(model.jnt_dofadr[joint])
        # Half height of the piece's collision cylinder: its centre sits this far above what it stands on.
        self._half_height = float(model.geom_size[model.body_geomadr[self.body]][1])
        # Everything that is not a piece, the bin or the world is the arm.
        self._arm = {
            i
            for i in range(1, model.nbody)
            if not model.body(i).name.startswith("p_") and model.body(i).name != "bin"
        }
        # Pieces that started in the bin are captured: neither their tipping nor their shifting is board damage.
        bin_prefix = f"p_{BIN}_"
        self.others = {
            i: data.xpos[i].copy()
            for n, i in world.piece_bodies().items()
            if i != self.body and not n.startswith(bin_prefix)
        }
        self.peak_shift = dict.fromkeys(self.others, 0.0)
        self.min_up = dict.fromkeys(self.others, 1.0)
        self.max_lift = 0.0
        self.reached = False
        self.last_lifted_xy: np.ndarray | None = None

    def update(self) -> None:
        data = self.world.data
        piece = data.xpos[self.body]
        lift = float(piece[2]) - self.start_z
        self.max_lift = max(self.max_lift, lift)
        if lift > LIFT_M / 2:
            self.last_lifted_xy = piece[:2].copy()
        tip = data.site_xpos[self.world.gripper_site]
        if np.linalg.norm(tip[:2] - self.source) < APPROACH_M and tip[2] - self.start_z < APPROACH_HEIGHT_M:
            self.reached = True
        self._observe_others()

    def _observe_others(self) -> None:
        data = self.world.data
        for i, start in self.others.items():
            self.peak_shift[i] = max(self.peak_shift[i], float(np.linalg.norm(data.xpos[i] - start)))
            self.min_up[i] = min(self.min_up[i], float(data.xmat[i].reshape(3, 3)[2, 2]))

    def outcome(self) -> Outcome:
        data, cell = self.world.data, self.world.cell
        target = self.transfer[1]
        self._observe_others()
        up = float(data.xmat[self.body].reshape(3, 3)[2, 2])
        moved = sum(self.peak_shift[i] > DISTURB_M or self.min_up[i] < UPRIGHT_COS for i in self.others)
        near = self.last_lifted_xy is not None and (
            np.linalg.norm(self.last_lifted_xy - np.array(cell.location_xy(target))) < 2 * cell.square_size
        )
        return classify(
            reached=self.reached,
            max_lift=self.max_lift,
            final_xy=tuple(data.xpos[self.body][:2]),
            final_up=up,
            target_rect=cell.location_rect(target),
            others_moved=moved,
            released_near_target=bool(near),
            square_size=cell.square_size,
            is_bin=target == BIN,
            at_rest=self._at_rest(),
            on_surface=self._on_surface(target),
        )

    def _at_rest(self) -> bool:
        """Slow enough, and no longer held or pushed by the arm."""
        qvel = self.world.data.qvel
        linear = float(np.linalg.norm(qvel[self._dof : self._dof + 3]))
        angular = float(np.linalg.norm(qvel[self._dof + 3 : self._dof + 6]))
        return linear < REST_LINEAR_MPS and angular < REST_ANGULAR_RADPS and not self._touching_arm()

    def _touching_arm(self) -> bool:
        model, data = self.world.model, self.world.data
        for contact in data.contact[: data.ncon]:
            bodies = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
            if self.body in bodies and bodies & self._arm:
                return True
        return False

    def _surface_z(self, target: str) -> float:
        """Height of the top of the board square or parking pad `target` (read from the compiled scene)."""
        model = self.world.model
        name = f"{target}_pad" if target in (PARK_WHITE, PARK_BLACK) else "board"
        geom = model.geom(name)
        return float(geom.pos[2] + geom.size[2])

    def _on_surface(self, target: str) -> bool:
        z = float(self.world.data.xpos[self.body][2])
        if target == BIN:  # captured pieces may sit on others, so only the wall top bounds them
            return z < self.world.cell.bin_wall_height
        return abs(z - (self._surface_z(target) + self._half_height)) <= SURFACE_TOL_M
