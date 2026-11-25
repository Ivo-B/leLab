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

"""Run a trained checkpoint on `TRANSFER_PLAN` in the simulated cell and count how each transfer ends.

Every trial is a randomized scene (cameras, physics, lighting, colours, starting pose), drawn from
seeds that lie far from the generator's, so no trial repeats a training scene. The arm keeps its
randomized start pose through the settle steps: trials start from `generate.settled_world`, not from
`ChessTransferEnv.reset`, which settles at PARK.
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..chess.game import Transfer
from ..chess.transfers import CHESS_TASK, EVAL_NAME, TRANSFER_PLAN, transfer_class
from ..utils.forge_config import forge_configured, load_forge_config
from .cell import SimCell
from .env import ChessTransferEnv
from .generate import episode_variation, settled_world
from .positions import prepare_position, random_position

EVAL_SEED_BASE = (
    10_000  # generator seeds are small; `episode_variation` mixes the seed in as `seed * 1_000_003`
)
SUCCESS = "success"
REPORT_NAME = "eval_report.json"
VIDEO_DIR = "eval_videos"
DEFAULT_CAMERAS = ("top", "side", "wrist")
DEFAULT_SIZE = (640, 480)

logger = logging.getLogger(__name__)

Predict = Callable[[dict[str, Any]], np.ndarray]


@dataclass(frozen=True)
class Trial:
    transfer: Transfer
    seed: int
    outcome: str  # `success`, or the judge's failure name
    duration_s: float  # simulated seconds until the env terminated or truncated
    top_video: str | None = None

    @property
    def transfer_class(self) -> str:
        return transfer_class(self.transfer)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transfer": list(self.transfer),
            "class": self.transfer_class,
            "seed": self.seed,
            "outcome": self.outcome,
            "duration_s": self.duration_s,
            "top_video": self.top_video,
        }


def summarize(trials: Iterable[Trial]) -> dict[str, dict[str, int]]:
    """Counts of `success` and of each failure name, per transfer class."""
    summary: dict[str, Counter[str]] = {}
    for trial in trials:
        summary.setdefault(trial.transfer_class, Counter())[trial.outcome] += 1
    return {cls: dict(counts) for cls, counts in summary.items()}


@dataclass(frozen=True)
class EvalReport:
    trials: list[Trial]
    summary: dict[str, dict[str, int]]
    cameras: tuple[str, ...] = ()  # the cameras the policy read

    @classmethod
    def from_trials(cls, trials: list[Trial]) -> EvalReport:
        return cls(trials, summarize(trials))

    @property
    def success_rate(self) -> float:
        wins = sum(t.outcome == SUCCESS for t in self.trials)
        return wins / len(self.trials) if self.trials else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": EVAL_NAME,
            "success_rate": self.success_rate,
            "summary": self.summary,
            "cameras": list(self.cameras),
            "trials": [t.to_dict() for t in self.trials],
        }


class Mp4Writer:
    """Encodes frames to h264 as they arrive, so a 30 s trial is never held in memory."""

    def __init__(self, path: Path, width: int, height: int, fps: int) -> None:
        import av

        path.parent.mkdir(parents=True, exist_ok=True)
        self._container = av.open(str(path), "w")
        self._stream = self._container.add_stream("h264", rate=fps)
        self._stream.width, self._stream.height, self._stream.pix_fmt = width, height, "yuv420p"

    def write(self, frame: np.ndarray) -> None:
        import av

        for packet in self._stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            self._container.mux(packet)

    def close(self) -> None:
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()


def _start_trial(env: ChessTransferEnv, transfer: Transfer, seed: int, index: int) -> dict[str, Any]:
    """Build trial `index` of eval seed `seed` and return its first observation.

    The scene is randomized and the arm settles holding its randomized start pose (`settled_world`)."""
    rng = random.Random(f"{seed}:{index}")
    position = prepare_position(random_position(rng), transfer, rng)
    scene_cell, variation = episode_variation(seed, index, env.cell, True)
    world = settled_world(
        scene_cell,
        position,
        render=bool(env.cameras),
        width=env.width,
        height=env.height,
        variation=variation,
    )
    try:
        return env.start(world, transfer)
    except BaseException:
        world.close()
        raise


def run_plan(
    env: ChessTransferEnv,
    predict: Predict,
    seeds: int,
    *,
    on_reset: Callable[[], None] | None = None,
    video_dir: Path | None = None,
) -> EvalReport:
    """Every transfer of `TRANSFER_PLAN` once per seed, in randomized scenes, driven by `predict`.

    `predict` maps the raw observation (`ChessTransferEnv.observation_for_policy`) to a joint
    action; `on_reset` runs before each trial, to clear the policy's action queue. With `video_dir`
    (and a `top` camera in `env`) each trial's top view is written to `<video_dir>/<seed>-<n>.mp4`.
    A trial ends when the env terminates or truncates; its outcome is the judge's verdict.
    """
    if seeds < 1:
        raise ValueError(f"seeds must be at least 1, got {seeds}")
    record = video_dir is not None and "top" in env.cameras
    trials: list[Trial] = []
    for seed in range(seeds):
        eval_seed = EVAL_SEED_BASE + seed
        for index, transfer in enumerate(TRANSFER_PLAN):
            obs = _start_trial(env, transfer, eval_seed, index)
            if on_reset is not None:
                on_reset()
            writer = None
            path = None
            if record:
                assert video_dir is not None
                path = video_dir / f"{seed}-{index:02d}-{transfer[0]}-{transfer[1]}.mp4"
                writer = Mp4Writer(path, env.width, env.height, env.fps)
            steps = 0
            try:
                while True:
                    if writer is not None:
                        writer.write(obs["pixels"]["top"])
                    action = predict(env.observation_for_policy(obs))
                    obs, _, terminated, truncated, info = env.step(action)
                    steps += 1
                    if terminated or truncated:
                        break
            finally:
                if writer is not None:
                    writer.close()
            outcome = info["outcome"]
            trials.append(
                Trial(
                    transfer,
                    eval_seed,
                    SUCCESS if outcome.success else outcome.failure or "unknown",
                    steps / env.fps,
                    str(path) if path is not None else None,
                )
            )
    return EvalReport.from_trials(trials)


@dataclass(frozen=True)
class LoadedPolicy:
    predict: PolicyPredictor
    cameras: tuple[str, ...]  # the cameras the policy reads
    size: tuple[int, int]  # (width, height) of its images


class PolicyPredictor:
    """A LeRobot policy and its checkpoint's processors, as a `Predict` callable."""

    def __init__(self, policy, preprocessor, postprocessor, features: dict[str, dict]) -> None:
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.features = features

    def reset(self) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def __call__(self, raw: dict[str, Any]) -> np.ndarray:
        import torch

        from lerobot.common.control_utils import predict_action
        from lerobot.utils.constants import OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame

        frame = build_dataset_frame(self.features, raw, prefix=OBS_STR)
        action = predict_action(
            frame,
            self.policy,
            torch.device(self.policy.config.device),
            self.preprocessor,
            self.postprocessor,
            bool(self.policy.config.use_amp),
            task=CHESS_TASK,
            robot_type="so101_follower",
        )
        return action.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _pretrained_dir(checkpoint: Path) -> Path:
    """`checkpoint` itself, or its `pretrained_model` folder when it is a step directory."""
    inner = checkpoint / "pretrained_model"
    return inner if inner.is_dir() else checkpoint


def load_policy(checkpoint: Path) -> LoadedPolicy:
    """Policy, processors and image geometry from a `pretrained_model` directory.

    LeRobot's API: `PreTrainedConfig.from_pretrained` reads `config.json` (its `__post_init__` swaps
    in an available device), `get_policy_class(type).from_pretrained(path, config=cfg)` loads the
    weights in eval mode, and `make_pre_post_processors(cfg, pretrained_path=path)` loads the
    saved processors with their dataset statistics."""
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import get_policy_class, make_pre_post_processors
    from lerobot.processor import bind_relative_anchor

    from .generate import dataset_features

    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.pretrained_path = checkpoint
    policy = get_policy_class(cfg.type).from_pretrained(checkpoint, config=cfg)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(cfg.device)}},
    )
    bind_relative_anchor(policy, preprocessor)

    prefix = "observation.images."
    images = {k.removeprefix(prefix): v for k, v in cfg.image_features.items()}
    unknown = sorted(set(images) - set(DEFAULT_CAMERAS))
    if unknown:
        raise ValueError(f"The checkpoint reads cameras the sim does not have: {unknown}")
    cameras = tuple(images) or DEFAULT_CAMERAS
    height, width = next(iter(cfg.image_features.values())).shape[1:] if images else DEFAULT_SIZE[::-1]
    wanted = set(cfg.input_features)
    features = {k: v for k, v in dataset_features(width, height).items() if k in wanted}
    return LoadedPolicy(
        PolicyPredictor(policy, preprocessor, postprocessor, features), cameras, (width, height)
    )


def _model_name(checkpoint: Path) -> str:
    parts = checkpoint.resolve().parts[-4:]
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", "-".join(parts)).strip("-") or "checkpoint"


def log_to_weave(report: EvalReport, eval_logger, as_video: Callable[[str], Any] | None = None) -> None:
    """One prediction per trial on `eval_logger` (a `weave.EvaluationLogger`), then the summary."""
    for trial in report.trials:
        video = trial.top_video
        prediction = eval_logger.log_prediction(
            inputs={
                "transfer": "-".join(trial.transfer),
                "class": trial.transfer_class,
                "domain": "sim",
                "seed": trial.seed,
                "cameras": ",".join(report.cameras),
            },
            output={
                "outcome": trial.outcome,
                "duration_s": trial.duration_s,
                "top_video": as_video(video) if video and as_video else video,
            },
        )
        prediction.log_score("success", 1 if trial.outcome == SUCCESS else 0)
        prediction.finish()
    eval_logger.log_summary(
        {"success_rate": report.success_rate, "by_class": report.summary, "cameras": list(report.cameras)}
    )


def _weave_logger(cfg, checkpoint: Path):
    import weave

    weave.init(f"{cfg.entity}/{cfg.project}")
    return weave.EvaluationLogger(name=EVAL_NAME, model=_model_name(checkpoint)), weave.Content.from_path


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    seeds: int = 3,
    cell: SimCell | None = None,
    out: Path | None = None,
) -> EvalReport:
    """Run the checkpoint on `TRANSFER_PLAN` x `seeds`, write `eval_report.json`, and (when Forge is
    configured) log a Weave evaluation `so101-chess-sim-eval` with one `top` video per trial.

    The report goes to `out`, or next to the checkpoint's parent: `checkpoint.parent`."""
    checkpoint = Path(checkpoint)
    out = Path(out) if out is not None else checkpoint.parent
    forge = load_forge_config()
    log_weave = forge_configured(forge)
    loaded = load_policy(_pretrained_dir(checkpoint))
    cameras = loaded.cameras if not log_weave or "top" in loaded.cameras else ("top", *loaded.cameras)
    env = ChessTransferEnv(cell, width=loaded.size[0], height=loaded.size[1], cameras=cameras)
    try:
        report = run_plan(
            env,
            loaded.predict,
            seeds,
            on_reset=loaded.predict.reset,
            video_dir=out / VIDEO_DIR if log_weave else None,
        )
    finally:
        env.close()
    report = replace(report, cameras=tuple(loaded.cameras))
    out.mkdir(parents=True, exist_ok=True)
    (out / REPORT_NAME).write_text(json.dumps(report.to_dict(), indent=2))
    if log_weave:
        try:
            eval_logger, as_video = _weave_logger(forge, checkpoint)
            log_to_weave(report, eval_logger, as_video)
        except Exception as exc:  # the report is already on disk; never lose it to a logging failure
            logger.warning("Weave evaluation was not logged: %s", type(exc).__name__)
    return report
