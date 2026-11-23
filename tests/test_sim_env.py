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

import numpy as np
import pytest

from tests.sim_helpers import arm_assets_or_skip, requires_mujoco

pytestmark = requires_mujoco


def test_env_reset_overlays_markers_and_expert_actions_succeed() -> None:
    arm_assets_or_skip()
    import mujoco

    from lelab.chess.overlay import SOURCE_RGB, TARGET_RGB
    from lelab.sim.env import ChessTransferEnv
    from lelab.sim.expert import ExpertParams, plan_transfer

    env = ChessTransferEnv(width=320, height=240)
    try:
        try:
            obs, info = env.reset(seed=0, options={"transfer": ("e2", "e4")})
        except mujoco.FatalError as e:  # pragma: no cover
            pytest.skip(str(e))
        except Exception as e:
            if "GL" in str(e) or "CGL" in str(e) or "context" in str(e).lower():
                pytest.skip(f"no OpenGL context: {e}")
            raise
        assert info["transfer"] == ("e2", "e4")
        assert obs["agent_pos"].shape == (6,)
        top = obs["pixels"]["top"]
        assert top.shape == (240, 320, 3)
        assert np.all(top == np.array(SOURCE_RGB), axis=-1).any()
        assert np.all(top == np.array(TARGET_RGB), axis=-1).any()
        assert not np.all(obs["pixels"]["wrist"] == np.array(SOURCE_RGB), axis=-1).any()

        raw = env.observation_for_policy(obs)
        assert set(raw) >= {"shoulder_pan.pos", "gripper.pos", "top", "wrist", "side"}

        terminated = truncated = False
        assert env.world is not None
        for action in plan_transfer(env.world, ("e2", "e4"), ExpertParams()):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        assert terminated and reward == 1.0 and info["outcome"].success
    finally:
        env.close()
