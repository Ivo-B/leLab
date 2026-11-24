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
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from tests.sim_helpers import arm_assets_or_skip, requires_mujoco


def _park() -> np.ndarray:
    from lelab.sim.world import PARK

    return np.array(PARK, dtype=np.float32)


def _env():
    from lelab.sim.env import ChessTransferEnv

    arm_assets_or_skip()
    return ChessTransferEnv(width=160, height=120, cameras=(), episode_seconds=1.0)


def test_summary_counts_outcomes_per_transfer_class() -> None:
    from lelab.sim.evaluate import Trial, summarize

    trials = [
        Trial(("a1", "c3"), 0, "success", 2.0),
        Trial(("a1", "c3"), 1, "place", 3.0),
        Trial(("c3", "bin"), 0, "approach", 1.0),
    ]
    assert summarize(trials) == {"near": {"success": 1, "place": 1}, "bin": {"approach": 1}}


@requires_mujoco
def test_run_plan_plays_every_transfer_once_per_seed() -> None:
    from lelab.chess.transfers import TRANSFER_PLAN
    from lelab.sim.evaluate import run_plan

    env = _env()
    resets = []
    try:
        report = run_plan(env, lambda raw: _park(), 2, on_reset=lambda: resets.append(1))
    finally:
        env.close()
    assert len(report.trials) == len(TRANSFER_PLAN) * 2 == len(resets)
    assert [t.transfer for t in report.trials] == list(TRANSFER_PLAN) * 2
    assert sorted({t.seed for t in report.trials}) == [10_000, 10_001]
    assert sum(sum(counts.values()) for counts in report.summary.values()) == len(TRANSFER_PLAN) * 2
    # Holding PARK never lifts the piece, so no trial succeeds and each ends at the 1 s truncation.
    assert {t.outcome for t in report.trials} == {"approach"}
    assert {t.duration_s for t in report.trials} == {1.0}
    assert report.success_rate == 0.0


@requires_mujoco
def test_run_plan_keeps_the_randomized_start_pose_and_the_scene_variation() -> None:
    from lelab.sim.evaluate import EVAL_SEED_BASE, run_plan
    from lelab.sim.generate import episode_variation
    from lelab.sim.world import PARK

    env = _env()
    starts: list[np.ndarray] = []

    def predict(raw: dict) -> np.ndarray:
        if not starts:  # the first observation of trial 0
            starts.append(np.array([raw[k] for k in raw if k.endswith(".pos")], dtype=np.float32))
        return _park()

    try:
        run_plan(env, predict, 1)
        base_cell = env.cell
    finally:
        env.close()
    expected = np.array(episode_variation(EVAL_SEED_BASE, 0, base_cell, True)[1].initial_joints)
    # shoulder_lift (index 1) starts at its joint limit, so the world clamps it; the rest hold the draw.
    free = [0, 2, 3, 4, 5]
    assert np.abs(starts[0] - expected)[free].max() < 0.1, "settling must hold the start pose"
    assert np.abs(expected - np.array(PARK))[free].max() > 0.5, (
        "the draw must differ from PARK to mean anything"
    )


def test_log_to_weave_writes_one_prediction_per_trial_with_its_video() -> None:
    from lelab.sim.evaluate import EvalReport, Trial, log_to_weave

    class Prediction:
        def __init__(self) -> None:
            self.scores: dict[str, object] = {}
            self.finished = False

        def log_score(self, name: str, value: object) -> None:
            self.scores[name] = value

        def finish(self) -> None:
            self.finished = True

    class Logger:
        def __init__(self) -> None:
            self.predictions: list[tuple[dict, dict, Prediction]] = []
            self.summary: dict | None = None

        def log_prediction(self, inputs: dict, output: dict) -> Prediction:
            self.predictions.append((inputs, output, Prediction()))
            return self.predictions[-1][2]

        def log_summary(self, summary: dict) -> None:
            self.summary = summary

    report = EvalReport.from_trials(
        [Trial(("a1", "c3"), 10_000, "success", 4.0, "/tmp/a.mp4"), Trial(("c3", "a1"), 10_000, "place", 5.0)]
    )
    logger = Logger()
    log_to_weave(report, logger, as_video=lambda path: f"video:{path}")
    (inputs, output, prediction), (_, second_output, second) = logger.predictions
    assert inputs == {"transfer": "a1-c3", "class": "near", "seed": 10_000}
    assert output["top_video"] == "video:/tmp/a.mp4" and second_output["top_video"] is None
    assert (prediction.scores, second.scores) == ({"success": 1}, {"success": 0})
    assert prediction.finished and second.finished
    assert logger.summary == {"success_rate": 0.5, "by_class": report.summary}


@requires_mujoco
def test_evaluate_checkpoint_writes_the_report_and_skips_weave_when_forge_is_not_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lelab.sim import evaluate
    from lelab.sim.env import ChessTransferEnv

    arm_assets_or_skip()
    for name in ("WANDB_API_KEY", "WANDB_ENTITY"):
        monkeypatch.delenv(name, raising=False)

    class Predictor:
        def __call__(self, raw: dict) -> np.ndarray:
            return _park()

        def reset(self) -> None:
            pass

    loaded = evaluate.LoadedPolicy(Predictor(), (), (160, 120))
    loaded_from: list[Path] = []
    weave_calls: list[Path] = []
    monkeypatch.setattr(evaluate, "load_policy", lambda path: loaded_from.append(path) or loaded)
    monkeypatch.setattr(evaluate, "_weave_logger", lambda cfg, path: weave_calls.append(path))
    monkeypatch.setattr(evaluate, "ChessTransferEnv", partial(ChessTransferEnv, episode_seconds=1.0))

    checkpoint = tmp_path / "run" / "checkpoints" / "000100" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    report = evaluate.evaluate_checkpoint(checkpoint, seeds=1)

    assert loaded_from == [checkpoint] and weave_calls == []
    data = json.loads((checkpoint.parent / "eval_report.json").read_text())
    assert len(data["trials"]) == len(report.trials) == 20
    assert data["summary"] == report.summary and data["name"] == "so101-chess-sim-eval"

    out = tmp_path / "elsewhere"
    evaluate.evaluate_checkpoint(checkpoint, seeds=1, out=out)
    assert (out / "eval_report.json").is_file()
