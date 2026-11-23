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

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from tests.sim_helpers import arm_assets_or_skip, is_gl_context_error, requires_mujoco

pytestmark = requires_mujoco


def test_features_match_the_real_follower() -> None:
    from lelab.sim.generate import dataset_features

    features = dataset_features(640, 480)
    assert features["action"]["names"] == [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]
    assert features["observation.state"]["shape"] == (6,)
    for cam in ("top", "wrist", "side"):
        assert features[f"observation.images.{cam}"]["shape"] == (480, 640, 3)
        assert features[f"observation.images.{cam}"]["dtype"] == "video"


def test_ordered_map_keeps_order_across_workers() -> None:
    from lelab.sim.generate import ordered_map

    args = [(2, i) for i in range(7)]
    assert list(ordered_map(pow, args, 1)) == list(ordered_map(pow, args, 3)) == [2**i for i in range(7)]


def test_gate_table_counts_classes_and_rate() -> None:
    from lelab.sim.__main__ import gate_table

    table, rate = gate_table(
        Counter({("near", "success"): 3, ("near", "disturbed"): 1, ("bin", "success"): 4})
    )
    assert "| near | 3/4 | disturbed 1 |" in table
    assert "| bin | 4/4 | — |" in table
    assert "**7/8**" in table
    assert rate == pytest.approx(7 / 8)


def test_physics_only_attempts_do_not_depend_on_the_worker_count() -> None:
    arm_assets_or_skip()
    from lelab.sim.cell import SimCell
    from lelab.sim.expert import ExpertParams
    from lelab.sim.generate import ordered_map, plan_attempt

    jobs = [(3, i, SimCell().to_dict(), asdict(ExpertParams())) for i in range(2)]
    serial = list(ordered_map(plan_attempt, jobs, 1))
    pooled = list(ordered_map(plan_attempt, jobs, 2))
    assert [a.index for a in serial] == [0, 1]
    for a, b in zip(serial, pooled, strict=True):
        assert (a.index, a.transfer, a.failure) == (b.index, b.transfer, b.failure)
        assert (a.actions is None) == (b.actions is None)
        if a.actions is not None:
            np.testing.assert_array_equal(a.actions, b.actions)


def test_generates_two_episodes_with_sidecar_and_metadata(
    tmp_lerobot_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arm_assets_or_skip()
    from lelab.chess.transfers import read_transfer_sidecar
    from lelab.sim import generate
    from lelab.sim.generate import generate_dataset
    from lelab.sim.judge import Outcome

    replay, recorded = generate._replay, []

    def first_recording_diverges(world, transfer, actions, on_step=None) -> Outcome:
        outcome = replay(world, transfer, actions, on_step)
        if on_step is not None:
            recorded.append(transfer)
            if len(recorded) == 1:
                return Outcome(False, "place", outcome.max_lift, outcome.final_error)
        return outcome

    monkeypatch.setattr(generate, "_replay", first_recording_diverges)
    root = tmp_path / "datasets" / "test-sim"
    try:
        report = generate_dataset(
            "local/test-sim", 2, root=root, seed=0, width=160, height=120, log_artifact=False
        )
    except Exception as e:
        if is_gl_context_error(e):
            pytest.skip(f"no OpenGL context: {e}")
        raise
    assert report.root == root
    assert report.saved == 2
    entries = read_transfer_sidecar(report.root)
    assert [e["episode_index"] for e in entries] == [0, 1]
    # The diverged recording was discarded; the saved episodes are the next two successful transfers.
    assert report.dropped.get("nondeterministic") == 1
    assert [(e["source"], e["target"]) for e in entries] == [tuple(t) for t in recorded[1:]]
    meta = json.loads((report.root / "meta" / "garage_sim.json").read_text())
    assert meta["source"] == "sim" and meta["seed"] == 0 and "expert" in meta and "cell" in meta
    assert meta["attempts"] == 2 + sum(meta["dropped"].values())
    info = json.loads((report.root / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2
    assert info["robot_type"] == "so101_follower"
    assert info["features"]["observation.images.top"]["shape"] == [120, 160, 3]
    videos = sorted((report.root / "videos").rglob("*.mp4"))
    assert len(videos) == 3
    import av

    for video in videos:
        with av.open(str(video)) as container:
            assert container.streams.video[0].codec_context.name == "h264", video
            assert sum(1 for _ in container.decode(video=0)) == info["total_frames"], video
    assert meta["interrupted"] is False and meta["error"] is None and meta["vcodec"] == "h264"

    from lerobot.datasets import LeRobotDataset

    loaded = LeRobotDataset("local/test-sim", root=root)
    last = loaded[len(loaded) - 1]
    for cam in ("top", "wrist", "side"):
        assert tuple(last[f"observation.images.{cam}"].shape) == (3, 120, 160)


def test_gives_up_after_max_attempts_and_marks_the_dataset_interrupted(
    tmp_lerobot_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lelab.sim import generate
    from lelab.sim.generate import Attempt, generate_dataset

    def always_unplannable(seed: int, index: int, cell: dict, params: dict) -> Attempt:
        return Attempt(index, ("e2", "e4"), None, "unplannable")

    monkeypatch.setattr(generate, "plan_attempt", always_unplannable)
    root = tmp_path / "datasets" / "never"
    calls: list[tuple[int, int, int]] = []
    with pytest.raises(RuntimeError, match=r"Saved 0/2 episodes in 5 attempts.*unplannable"):
        generate_dataset(
            "local/never",
            2,
            root=root,
            width=32,
            height=24,
            log_artifact=False,
            max_attempts=5,
            progress=lambda *a: calls.append(a),
        )
    assert calls == [(0, 2, n) for n in range(1, 6)]
    meta = json.loads((root / "meta" / "garage_sim.json").read_text())
    assert meta["source"] == "sim" and meta["interrupted"] is True and meta["error"] == "RuntimeError"
    assert meta["attempts"] == 5 and meta["dropped"] == {"unplannable": 5}


def test_a_finalize_failure_does_not_mask_the_error_that_stopped_the_run(
    tmp_lerobot_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lelab.sim import generate
    from lelab.sim.generate import Attempt, generate_dataset
    from lerobot.datasets import LeRobotDataset

    def stop(seed: int, index: int, cell: dict, params: dict) -> Attempt:
        raise KeyboardInterrupt

    def broken_finalize(self) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(generate, "plan_attempt", stop)
    monkeypatch.setattr(LeRobotDataset, "finalize", broken_finalize)
    root = tmp_path / "datasets" / "stopped"
    with pytest.raises(KeyboardInterrupt):
        generate_dataset("local/stopped", 1, root=root, width=32, height=24, log_artifact=False)
    meta = json.loads((root / "meta" / "garage_sim.json").read_text())
    assert meta["interrupted"] is True and meta["error"] == "KeyboardInterrupt"
