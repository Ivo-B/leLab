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

from tests.sim_helpers import arm_assets_or_skip, is_gl_context_error, requires_mujoco

pytestmark = requires_mujoco


def _make_env(**kwargs):
    from lelab.sim.env import ChessTransferEnv

    arm_assets_or_skip()
    return ChessTransferEnv(width=160, height=120, cameras=(), **kwargs)


def test_holding_park_never_terminates_and_truncates_with_a_failed_outcome() -> None:
    env = _make_env(episode_seconds=1.0)
    try:
        env.reset(seed=0, options={"transfer": ("e2", "e4")})
        park = env.world.joint_state()
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            _, reward, terminated, truncated, info = env.step(park)
            steps += 1
        assert truncated and not terminated
        assert steps == env.max_steps == 30
        assert reward == 0.0
        assert not info["outcome"].success and not info["is_success"]
        assert info["outcome"].failure == "approach"
    finally:
        env.close()


def test_expert_terminates_in_the_final_park_move_with_success() -> None:
    from lelab.sim.expert import ExpertParams, plan_transfer

    params = ExpertParams()
    env = _make_env()
    try:
        obs, info = env.reset(seed=0, options={"transfer": ("e2", "e4")})
        assert obs["pixels"] == {} and info["transfer"] == ("e2", "e4")
        actions = plan_transfer(env.world, ("e2", "e4"), params)
        # The plan ends with a 45-step move back to PARK followed by `settle_steps` of holding it.
        park_move = range(len(actions) - params.settle_steps - 45, len(actions) - params.settle_steps)
        index = -1
        for action in actions:
            index += 1
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        assert terminated and not truncated
        assert index in park_move, (index, park_move)
        assert reward == 1.0 and info["is_success"] and info["outcome"].success
    finally:
        env.close()


def test_observation_for_policy_has_float_joints() -> None:
    from lelab.sim.world import JOINT_NAMES

    env = _make_env()
    try:
        obs, _ = env.reset(seed=0, options={"transfer": ("e2", "e4")})
        raw = env.observation_for_policy(obs)
        assert list(raw) == JOINT_NAMES
        assert all(type(v) is float for v in raw.values())
    finally:
        env.close()


def test_env_reset_overlays_markers_and_expert_actions_succeed() -> None:
    arm_assets_or_skip()
    from lelab.chess.overlay import SOURCE_RGB, TARGET_RGB
    from lelab.sim.env import ChessTransferEnv
    from lelab.sim.expert import ExpertParams, plan_transfer

    env = ChessTransferEnv(width=320, height=240)
    try:
        try:
            obs, info = env.reset(seed=0, options={"transfer": ("e2", "e4")})
        except Exception as e:
            # Skip only when the GL context for mujoco.Renderer could not be created.
            if is_gl_context_error(e):
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
