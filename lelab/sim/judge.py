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

import numpy as np

from ..chess.game import BIN, Transfer

FAILURES = ("approach", "grasp", "drop", "place", "topple", "disturbed")
LIFT_M = 0.02
APPROACH_M = 0.01
PLACE_TOLERANCE = 0.35  # of a square
UPRIGHT_COS = 0.9
DISTURB_M = 0.005


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
) -> Outcome:
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
    if others_moved:
        return fail("disturbed")
    return Outcome(True, None, max_lift, error)


class TransferTracker:
    """Follows the moved piece and its neighbours through one transfer."""

    def __init__(self, world, transfer: Transfer) -> None:
        self.world = world
        self.transfer = transfer
        name = world.body_at(transfer[0])
        if name is None:
            raise ValueError(f"No piece on {transfer[0]}")
        self.body = world.body_id(name)
        self.others = {i: world.data.xpos[i].copy() for i in world.piece_bodies().values() if i != self.body}
        self.start_z = float(world.data.xpos[self.body][2])
        self.max_lift = 0.0
        self.reached = False
        self.last_lifted_xy: np.ndarray | None = None
        sx, sy = world.cell.location_xy(transfer[0])
        self.source = np.array([sx, sy])

    def update(self) -> None:
        data = self.world.data
        piece = data.xpos[self.body]
        lift = float(piece[2]) - self.start_z
        self.max_lift = max(self.max_lift, lift)
        if lift > LIFT_M / 2:
            self.last_lifted_xy = piece[:2].copy()
        tip = data.site_xpos[self.world.gripper_site]
        if np.linalg.norm(tip[:2] - self.source) < APPROACH_M and tip[2] - self.start_z < 0.05:
            self.reached = True

    def outcome(self) -> Outcome:
        data, cell = self.world.data, self.world.cell
        target = self.transfer[1]
        rect = cell.location_rect(target)
        up = float(data.xmat[self.body].reshape(3, 3)[2, 2])
        moved = 0
        for i, start in self.others.items():
            tipped = data.xmat[i].reshape(3, 3)[2, 2] < UPRIGHT_COS and not self._in_bin(i)
            if np.linalg.norm(data.xpos[i][:2] - start[:2]) > DISTURB_M or tipped:
                moved += 1
        near = self.last_lifted_xy is not None and (
            np.linalg.norm(self.last_lifted_xy - np.array(cell.location_xy(target))) < 2 * cell.square_size
        )
        return classify(
            reached=self.reached or self.max_lift >= LIFT_M,
            max_lift=self.max_lift,
            final_xy=tuple(data.xpos[self.body][:2]),
            final_up=up,
            target_rect=rect,
            others_moved=moved,
            released_near_target=bool(near),
            square_size=cell.square_size,
            is_bin=target == BIN,
        )

    def _in_bin(self, body: int) -> bool:
        (x0, y0), (x1, y1) = self.world.cell.location_rect(BIN)
        x, y = self.world.data.xpos[body][:2]
        return x0 <= x <= x1 and y0 <= y <= y1
