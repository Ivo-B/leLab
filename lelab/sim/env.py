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

"""One chess transfer per episode, observed like the real cell.

Observations carry the overlaid `top` frame and raw `wrist`/`side` frames, so a
policy trained on sim data sees what it will see on the real arm.
"""

from __future__ import annotations

import random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..chess.game import Transfer
from ..chess.overlay import GoalOverlayStep
from .cell import SimCell, load_cell
from .judge import LIFT_M, TransferTracker
from .positions import prepare_position, random_position, sample_sim_transfer
from .world import JOINT_NAMES, PARK, ChessWorld

# Arm parked: shoulder_pan .. wrist_flex all within this many degrees of PARK.
PARK_TOLERANCE_DEG = 8.0
SETTLE_STEPS = 10  # control steps at PARK that let the pieces come to rest before the episode starts


class ChessTransferEnv(gym.Env):
    """Gymnasium env: one transfer per episode, judged by `TransferTracker`.

    Terminates when all three hold: the piece was lifted (`max_lift > judge.LIFT_M`), the first four
    joints are within `PARK_TOLERANCE_DEG` of PARK, and the piece is at rest (`TransferTracker.at_rest`),
    so the judge never rules on a piece that is still moving. Truncates at `episode_seconds`.
    `info["outcome"]` appears at the end.

    `cameras=()` builds the world without a renderer: no GL context is needed and `pixels` is empty.

    The gripper reading is not part of the test: the scripted expert opens the jaws only as far as
    the piece needs (about 15 on the 0-100 scale, against 8 while holding), far below `GRIPPER_OPEN`.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        cell: SimCell | None = None,
        *,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        episode_seconds: float = 30.0,
        cameras: tuple[str, ...] = ("top", "wrist", "side"),
    ) -> None:
        super().__init__()
        self.cell = cell or load_cell()
        self.fps, self.width, self.height = fps, width, height
        self.max_steps = int(episode_seconds * fps)
        self.cameras = cameras
        image = spaces.Box(0, 255, (height, width, 3), np.uint8)
        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(-360.0, 360.0, (6,), np.float32),
                "pixels": spaces.Dict(dict.fromkeys(cameras, image)),
            }
        )
        self.action_space = spaces.Box(-360.0, 360.0, (6,), np.float32)
        self.world: ChessWorld | None = None
        self.overlay: GoalOverlayStep | None = None
        self.tracker: TransferTracker | None = None
        self.transfer: Transfer | None = None
        self._steps = 0

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        # From the env's generator (gymnasium convention), so unseeded resets continue a seeded sequence.
        rng = random.Random(int(self.np_random.integers(2**63)))
        position = options.get("position") or random_position(rng)
        transfer: Transfer = tuple(options.get("transfer") or sample_sim_transfer(position, rng))  # type: ignore[assignment]
        position = prepare_position(position, transfer, rng)
        self.close()
        self.world = ChessWorld(
            self.cell,
            position,
            fps=self.fps,
            width=self.width,
            height=self.height,
            render=bool(self.cameras),
        )
        for _ in range(SETTLE_STEPS):
            self.world.step(np.array(PARK, dtype=np.float32))
        self._begin(transfer)
        return self._obs(), {"transfer": transfer, "is_success": False}

    def start(self, world: ChessWorld, transfer: Transfer) -> dict[str, Any]:
        """Begin an episode on a `world` the caller built and settled, and return its first observation.

        `reset` settles at PARK, which would erase a randomized start pose; a caller that wants the
        arm to keep it (`sim.evaluate`) settles the world itself and hands it over. The env owns
        `world` from here on and closes it on `close` or the next `reset`/`start`."""
        self.close()
        self.world = world
        self._begin(transfer)
        return self._obs()

    def _begin(self, transfer: Transfer) -> None:
        assert self.world is not None
        self.overlay = GoalOverlayStep(calibration=self.world.board_calibration("top"))
        self.overlay.set_goal(*transfer)
        self.tracker = TransferTracker(self.world, transfer)
        self.transfer = transfer
        self._steps = 0

    def _obs(self) -> dict[str, Any]:
        assert self.world is not None and self.overlay is not None
        pixels = {c: self.world.render(c) for c in self.cameras}  # `world.CAMERAS` maps wrist -> wrist_cam
        if "top" in pixels:
            pixels["top"] = self.overlay.observation({"top": pixels["top"]})["top"]
        return {"agent_pos": self.world.joint_state(), "pixels": pixels}

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        assert self.world is not None and self.tracker is not None
        self.world.step(np.asarray(action, dtype=np.float32))
        self.tracker.update()
        self._steps += 1
        state = self.world.joint_state()
        lifted = self.tracker.max_lift > LIFT_M
        near_park = bool(np.abs(state[:4] - np.array(PARK[:4])).max() < PARK_TOLERANCE_DEG)
        terminated = bool(lifted and near_park and self.tracker.at_rest())
        truncated = self._steps >= self.max_steps
        info: dict[str, Any] = {"transfer": self.transfer, "is_success": False}
        reward = 0.0
        if terminated or truncated:
            outcome = self.tracker.outcome()
            info["outcome"] = outcome
            info["is_success"] = outcome.success
            reward = 1.0 if outcome.success else 0.0
        return self._obs(), reward, terminated, truncated, info

    def observation_for_policy(self, obs: dict[str, Any]) -> dict[str, Any]:
        """The raw-robot observation LeRobot's `predict_action` expects: scalar joints plus HWC frames."""
        raw: dict[str, Any] = {name: float(v) for name, v in zip(JOINT_NAMES, obs["agent_pos"], strict=True)}
        raw.update(obs["pixels"])
        return raw

    def close(self) -> None:
        if self.world is not None:
            self.world.close()
            self.world = None
