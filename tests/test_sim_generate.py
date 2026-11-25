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
            "local/test-sim", 2, root=root, seed=0, width=160, height=120, log_artifact=False, randomize=False
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
    assert meta["randomization"] == {"enabled": False, "seed": 0} and meta["visuals"] == "staunton"
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
            randomize=False,
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

    cleared: list[bool] = []
    monkeypatch.setattr(generate, "plan_attempt", stop)
    monkeypatch.setattr(LeRobotDataset, "finalize", broken_finalize)
    monkeypatch.setattr(LeRobotDataset, "clear_episode_buffer", lambda self: cleared.append(True))
    root = tmp_path / "datasets" / "stopped"
    with pytest.raises(KeyboardInterrupt):
        generate_dataset(
            "local/stopped", 1, root=root, width=32, height=24, log_artifact=False, randomize=False
        )
    meta = json.loads((root / "meta" / "garage_sim.json").read_text())
    assert meta["interrupted"] is True and meta["error"] == "KeyboardInterrupt"
    assert cleared == [True]


def test_an_interrupted_episode_leaves_no_frame_images(
    tmp_lerobot_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lelab.sim import generate
    from lelab.sim.generate import CAMERAS, Attempt, generate_dataset
    from lerobot.datasets import LeRobotDataset

    def one_frame_then_stop(seed: int, index: int, cell: dict, params: dict) -> Attempt:
        dataset = created[0]
        frame = {
            "observation.state": np.zeros(6, np.float32),
            "action": np.zeros(6, np.float32),
            "task": "t",
            **{f"observation.images.{c}": np.zeros((24, 32, 3), np.uint8) for c in CAMERAS},
        }
        dataset.add_frame(frame)
        raise KeyboardInterrupt

    create, created = LeRobotDataset.create, []

    def recording_create(*args, **kwargs):
        created.append(create(*args, **kwargs))
        return created[-1]

    monkeypatch.setattr(LeRobotDataset, "create", recording_create)
    monkeypatch.setattr(generate, "plan_attempt", one_frame_then_stop)
    root = tmp_path / "datasets" / "cut"
    with pytest.raises(KeyboardInterrupt):
        generate_dataset("local/cut", 1, root=root, width=32, height=24, log_artifact=False, randomize=False)
    assert not list(root.rglob("*.png"))


def test_planning_and_recording_build_the_same_randomized_scene(
    tmp_lerobot_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import random

    from lelab.sim import generate
    from lelab.sim.cell import SimCell
    from lelab.sim.generate import Attempt, generate_dataset
    from lelab.sim.randomize import SimVariation, randomize_cell, randomize_variation
    from lelab.sim.world import PARK

    planned: list[tuple[int, dict, dict | None]] = []
    recorded: list[tuple[SimCell, SimVariation | None]] = []

    def plan(seed: int, index: int, cell: dict, params: dict, variation: dict | None = None) -> Attempt:
        planned.append((index, cell, variation))
        return Attempt(index, ("e2", "e4"), np.zeros((2, 6), np.float32), None)

    class SceneBuiltError(Exception):
        pass

    def settled(cell: SimCell, position, *, variation=None, **kwargs):
        recorded.append((cell, variation))
        raise SceneBuiltError

    monkeypatch.setattr(generate, "plan_attempt", plan)
    monkeypatch.setattr(generate, "settled_world", settled)
    base, seed = SimCell(), 7
    with pytest.raises(SceneBuiltError):
        generate_dataset(
            "local/rand",
            1,
            root=tmp_path / "rand",
            seed=seed,
            cell=base,
            width=32,
            height=24,
            log_artifact=False,
        )

    # What the attempt's seed draws, in the order the record promises: the cell, then the variation.
    rng = random.Random(generate._episode_seed(seed, 0))
    expected_cell = randomize_cell(base, rng)
    expected_variation = randomize_variation(rng, PARK)
    assert expected_cell != base
    (index, cell_dict, variation_dict), (rec_cell, rec_variation) = planned[0], recorded[0]
    assert index == 0
    assert SimCell.from_dict(cell_dict) == rec_cell == expected_cell
    assert variation_dict is not None
    assert SimVariation.from_dict(variation_dict) == rec_variation == expected_variation
    meta = json.loads((tmp_path / "rand" / "meta" / "garage_sim.json").read_text())
    assert meta["randomization"] == {"enabled": True, "seed": seed} and meta["visuals"] == "staunton"
    assert meta["cell"] == json.loads(json.dumps(base.to_dict()))  # the nominal cell, not one attempt's draw


def test_no_randomize_plans_and_records_the_nominal_scene(
    tmp_lerobot_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lelab.sim import generate
    from lelab.sim.cell import SimCell
    from lelab.sim.generate import Attempt, generate_dataset

    planned: list[tuple] = []

    def plan(*args) -> Attempt:
        planned.append(args)
        return Attempt(args[1], ("e2", "e4"), None, "unplannable")

    monkeypatch.setattr(generate, "plan_attempt", plan)
    base = SimCell()
    with pytest.raises(RuntimeError):
        generate_dataset(
            "local/nominal",
            1,
            root=tmp_path / "nominal",
            cell=base,
            width=32,
            height=24,
            log_artifact=False,
            max_attempts=2,
            randomize=False,
        )
    assert [len(a) for a in planned] == [4, 4] and planned[0][2] == base.to_dict()
    meta = json.loads((tmp_path / "nominal" / "meta" / "garage_sim.json").read_text())
    assert meta["randomization"] == {"enabled": False, "seed": 0}


def test_randomize_is_the_default_and_the_cli_can_turn_it_off(monkeypatch: pytest.MonkeyPatch) -> None:
    import inspect

    from lelab.sim import __main__ as cli, generate

    assert inspect.signature(generate.generate_dataset).parameters["randomize"].default is True
    calls: list[dict] = []

    class Report:
        saved, root, dropped = 0, Path("x"), {}

    def fake(*args, **kwargs):
        calls.append(kwargs)
        return Report()

    monkeypatch.setattr(generate, "generate_dataset", fake)
    assert cli.main(["generate", "--repo-id", "local/x", "--episodes", "1", "--no-artifact"]) == 0
    assert cli.main(["generate", "--repo-id", "local/x", "--episodes", "1", "--no-randomize"]) == 0
    assert [c["randomize"] for c in calls] == [True, False]


def test_the_generator_uses_the_camera_contract() -> None:
    import inspect

    from lelab.chess.transfers import DATASET_CAMERAS, DATASET_FPS, DATASET_SIZE
    from lelab.sim import generate

    assert generate.CAMERAS == DATASET_CAMERAS
    assert generate.FPS == DATASET_FPS
    params = inspect.signature(generate.generate_dataset).parameters
    assert (params["width"].default, params["height"].default) == DATASET_SIZE
