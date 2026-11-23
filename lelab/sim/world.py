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

"""One simulated chess cell: the compiled scene with the SO-101, stepped in LeRobot units.

State and actions use LeRobot's `so101_follower` units with `use_degrees=True`:
degrees for the five arm joints and 0-100 for the gripper, absolute position
targets held for one control period (1/fps s).
"""

from __future__ import annotations

import contextlib

import mujoco
import numpy as np

from ..chess.calibration import BoardCalibration
from ..chess.game import BIN, PARK_BLACK, PARK_WHITE
from .assets import arm_model_path, require_mujoco
from .cell import SimCell
from .scene import SimPosition, scene_xml

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
JOINT_NAMES = [f"{joint}.pos" for joint in JOINTS]
HOME = (0.0, -60.0, 60.0, 35.0, 0.0, 30.0)
# Folded back over the base, clear of the board, so `top` sees every square.
PARK = (0.0, -100.0, 20.0, 30.0, 0.0, 30.0)
GRIPPER_OPEN = 70.0
GRIPPER_CLOSED = 2.0
MOUNT_SITE = "so101_mount"
GRIPPER_SITE = "gripperframe"
# Dataset camera names -> MJCF camera names; the wrist camera comes with the Menagerie arm.
CAMERAS = {"top": "top", "side": "side", "wrist": "wrist_cam"}


def compile_scene(xml: str) -> mujoco.MjModel:
    require_mujoco()
    spec = mujoco.MjSpec.from_string(xml)
    arm = mujoco.MjSpec.from_file(str(arm_model_path()))
    # Attaching drops the arm's tuned solver options; the scene sets none, so keep the arm's.
    for name in dir(arm.option):
        if not name.startswith("_"):
            with contextlib.suppress(AttributeError, TypeError):
                setattr(spec.option, name, getattr(arm.option, name))
    spec.attach(arm, site=spec.site(MOUNT_SITE), prefix="")
    return spec.compile()


class ChessWorld:
    def __init__(
        self,
        cell: SimCell,
        position: SimPosition,
        *,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        render: bool = True,
        initial_joints=PARK,
    ) -> None:
        self.cell = cell
        self.position = position
        self.width, self.height = width, height
        self.model = compile_scene(scene_xml(cell, position))
        self.substeps = max(1, round(1 / (fps * self.model.opt.timestep)))
        self.model.opt.timestep = 1 / (fps * self.substeps)
        self.data = mujoco.MjData(self.model)
        self.qadr = np.array([self.model.joint(n).qposadr[0] for n in JOINTS])
        self.dofadr = np.array([self.model.joint(n).dofadr[0] for n in JOINTS])
        self.actuators = np.array([self.model.actuator(n).id for n in JOINTS])
        self.lo, self.hi = self.model.actuator_ctrlrange[self.actuators].T.copy()
        self.gripper_site = self.model.site(GRIPPER_SITE).id
        self.renderer = mujoco.Renderer(self.model, height, width) if render else None
        self.reset_joints(initial_joints)

    # -- units -------------------------------------------------------------
    def to_ctrl(self, values) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        ctrl = np.deg2rad(values)
        ctrl[5] = self.lo[5] + values[5] / 100 * (self.hi[5] - self.lo[5])
        return np.clip(ctrl, self.lo, self.hi)

    def from_q(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        state = np.rad2deg(q)
        state[5] = (q[5] - self.lo[5]) / (self.hi[5] - self.lo[5]) * 100
        return state.astype(np.float32)

    def joint_state(self) -> np.ndarray:
        return self.from_q(self.data.qpos[self.qadr])

    # -- stepping ----------------------------------------------------------
    def reset_joints(self, values) -> None:
        ctrl = self.to_ctrl(values)
        self.data.qpos[self.qadr] = ctrl
        self.data.qvel[self.dofadr] = 0
        self.data.ctrl[self.actuators] = ctrl
        mujoco.mj_forward(self.model, self.data)

    def step(self, action) -> None:
        self.data.ctrl[self.actuators] = self.to_ctrl(action)
        mujoco.mj_step(self.model, self.data, nstep=self.substeps)

    # -- bodies ------------------------------------------------------------
    def body_id(self, name: str) -> int:
        return self.model.body(name).id

    def piece_bodies(self) -> dict[str, int]:
        return {
            self.model.body(i).name: i
            for i in range(self.model.nbody)
            if self.model.body(i).name.startswith("p_")
        }

    def body_at(self, location: str) -> str | None:
        """Name of the piece body that started on `location` (the top one for the bin)."""
        prefix = f"p_{location}_"
        names = sorted(n for n in self.piece_bodies() if n.startswith(prefix))
        return names[-1] if names else None

    # -- cameras -----------------------------------------------------------
    def render(self, camera: str) -> np.ndarray:
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, self.height, self.width)
        self.renderer.update_scene(self.data, camera=CAMERAS.get(camera, camera))
        return self.renderer.render().copy()

    def project(self, camera: str, points) -> np.ndarray:
        """World points (N, 3) -> pixel coordinates (N, 2) in a `width`x`height` render."""
        cam = self.model.camera(CAMERAS.get(camera, camera)).id
        pos = self.data.cam_xpos[cam]
        rot = self.data.cam_xmat[cam].reshape(3, 3)
        local = (np.asarray(points, dtype=np.float64) - pos) @ rot  # camera looks along -z
        focal = 0.5 * self.height / np.tan(np.deg2rad(self.model.cam_fovy[cam]) / 2)
        u = self.width / 2 + focal * local[:, 0] / -local[:, 2]
        v = self.height / 2 - focal * local[:, 1] / -local[:, 2]
        return np.stack([u, v], axis=1)

    def board_calibration(self, camera: str = "top") -> BoardCalibration:
        cell = self.cell
        z = cell.board_thickness
        x0, x1, h = cell.board_x0, cell.board_x0 + 8 * cell.square_size, cell.half
        corners = [(x0, h, z), (x0, -h, z), (x1, -h, z), (x1, h, z)]  # a1, h1, h8, a8

        def rect(name: str) -> list[tuple[float, float]]:
            (ax, ay), (bx, by) = cell.location_rect(name)
            pts = self.project(camera, [(ax, ay, 0.0), (ax, by, 0.0), (bx, by, 0.0), (bx, ay, 0.0)])
            return [tuple(map(float, p)) for p in pts]

        return BoardCalibration(
            robot="sim",
            image_size=(self.width, self.height),
            board_corners=[tuple(map(float, p)) for p in self.project(camera, corners)],
            bin=rect(BIN),
            park_white=rect(PARK_WHITE),
            park_black=rect(PARK_BLACK),
        )

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
