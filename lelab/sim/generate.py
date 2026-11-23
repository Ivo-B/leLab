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

"""Sim demonstrations -> LeRobotDataset, with the real cell's feature names.

Each attempt is first planned and run without rendering. Only successful transfers are
rendered and recorded, by replaying the planned actions on a fresh world (MuJoCo is
deterministic), so failed attempts cost no rendering or video encoding. The physics-only
passes can run in a process pool; their results are consumed in attempt order, so a seed
gives the same dataset for any number of workers.
"""

from __future__ import annotations

import itertools
import json
import logging
import multiprocessing
import os
import random
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..chess.game import Transfer
from ..chess.transfers import CHESS_TASK, TRANSFER_PLAN, transfer_class, write_transfer_sidecar
from .assets import ASSETS_REPO, ASSETS_REVISION
from .cell import SimCell, load_cell
from .env import SETTLE_STEPS
from .expert import ExpertParams, plan_transfer
from .judge import Outcome, TransferTracker
from .positions import prepare_position, random_position, sample_sim_transfer
from .scene import SimPosition
from .world import JOINT_NAMES, PARK, ChessWorld

CAMERAS = ("top", "wrist", "side")
FPS = 30
UNPLANNABLE = "unplannable"
NONDETERMINISTIC = "nondeterministic"

logger = logging.getLogger(__name__)


@dataclass
class GenerationReport:
    root: Path
    saved: int = 0
    dropped: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Attempt:
    """Result of one physics-only pass: the planned actions, and why the transfer failed (if it did)."""

    index: int
    transfer: Transfer
    actions: np.ndarray | None
    failure: str | None


def default_workers() -> int:
    return min(8, os.cpu_count() or 1)


def dataset_features(width: int, height: int) -> dict[str, dict[str, Any]]:
    from lerobot.utils.feature_utils import hw_to_dataset_features

    joints: dict[str, type | tuple[int, ...]] = dict.fromkeys(JOINT_NAMES, float)
    cameras: dict[str, type | tuple[int, ...]] = dict.fromkeys(CAMERAS, (height, width, 3))
    return {
        **hw_to_dataset_features(joints, "action", True),
        **hw_to_dataset_features({**joints, **cameras}, "observation", True),
    }


def ordered_map[T](fn: Callable[..., T], args: Iterable[tuple], workers: int) -> Iterator[T]:
    """`fn(*a)` for each `a` in `args`, lazily and in order; in a spawn process pool when `workers > 1`.

    At most `2 * workers` calls are in flight, so stopping early wastes little work.
    """
    if workers <= 1:
        for a in args:
            yield fn(*a)
        return
    pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
    pending: deque[Future[T]] = deque()
    try:
        it = iter(args)
        for a in itertools.islice(it, 2 * workers):
            pending.append(pool.submit(fn, *a))
        while pending:
            result = pending.popleft().result()
            for a in itertools.islice(it, 1):
                pending.append(pool.submit(fn, *a))
            yield result
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _episode_seed(seed: int, index: int) -> int:
    return seed * 1_000_003 + index


def episode_setup(seed: int, index: int) -> tuple[SimPosition, Transfer]:
    """Position and transfer of attempt `index`; the transfer's rank band cycles with `index`."""
    rng = random.Random(_episode_seed(seed, index))
    position = random_position(rng)
    transfer = sample_sim_transfer(position, rng, band=index)
    return prepare_position(position, transfer, rng), transfer


def settled_world(cell: SimCell, position: SimPosition, **kwargs: Any) -> ChessWorld:
    world = ChessWorld(cell, position, **kwargs)
    for _ in range(SETTLE_STEPS):
        world.step(np.array(PARK, dtype=np.float32))
    return world


def _replay(
    world: ChessWorld,
    transfer: Transfer,
    actions: Iterable[np.ndarray],
    on_step: Callable[[np.ndarray], None] | None = None,
) -> Outcome:
    tracker = TransferTracker(world, transfer)
    for action in actions:
        if on_step is not None:
            on_step(action)
        world.step(action)
        tracker.update()
    return tracker.outcome()


def plan_attempt(seed: int, index: int, cell: dict[str, Any], params: dict[str, Any]) -> Attempt:
    """Plan and run attempt `index` without rendering (picklable inputs, for the process pool)."""
    sim_cell, expert = SimCell.from_dict(cell), ExpertParams(**params)
    position, transfer = episode_setup(seed, index)
    world = settled_world(sim_cell, position, render=False)
    try:
        try:
            actions = np.asarray(plan_transfer(world, transfer, expert), dtype=np.float32)
        except ValueError:
            return Attempt(index, transfer, None, UNPLANNABLE)
        outcome = _replay(world, transfer, actions)
    finally:
        world.close()
    return Attempt(index, transfer, actions, None if outcome.success else outcome.failure or "unknown")


def gate_seed(seed: int, cell: dict[str, Any], params: dict[str, Any]) -> list[tuple[str, str]]:
    """(class, result) for every transfer of `TRANSFER_PLAN` on positions drawn from `random.Random(seed)`."""
    sim_cell, expert = SimCell.from_dict(cell), ExpertParams(**params)
    rng = random.Random(seed)
    results = []
    for transfer in TRANSFER_PLAN:
        world = settled_world(sim_cell, prepare_position(random_position(rng), transfer, rng), render=False)
        try:
            result = _replay(world, transfer, plan_transfer(world, transfer, expert)).failure or "success"
        except ValueError:
            result = UNPLANNABLE
        finally:
            world.close()
        results.append((transfer_class(transfer), result))
    return results


def run_gate(
    seeds: int, *, cell: SimCell | None = None, params: ExpertParams | None = None, workers: int = 1
) -> Counter[tuple[str, str]]:
    cell_dict, params_dict = (cell or load_cell()).to_dict(), asdict(params or ExpertParams())
    results: Counter[tuple[str, str]] = Counter()
    for seed_results in ordered_map(gate_seed, ((s, cell_dict, params_dict) for s in range(seeds)), workers):
        results.update(seed_results)
    return results


def generate_dataset(
    repo_id: str,
    episodes: int,
    *,
    root: Path | None = None,
    seed: int = 0,
    cell: SimCell | None = None,
    params: ExpertParams | None = None,
    width: int = 640,
    height: int = 480,
    log_artifact: bool = True,
    progress: Callable[[int, int, int], None] | None = None,
    workers: int = 1,
    max_attempts: int | None = None,
) -> GenerationReport:
    """Record `episodes` successful expert transfers.

    `progress(saved, episodes, attempts)` runs after every attempt, saved or dropped. Raises
    RuntimeError after `max_attempts` attempts (default `10 * episodes + 100`) without enough
    successes; the episodes saved so far are kept, and `meta/garage_sim.json` marks the
    dataset as interrupted whenever the run ends with an exception.
    """
    from lerobot.datasets import LeRobotDataset
    from lerobot.utils import constants

    from ..chess.overlay import GoalOverlayStep

    cell = cell or load_cell()
    params = params or ExpertParams()
    max_attempts = 10 * episodes + 100 if max_attempts is None else max_attempts
    root = Path(root) if root else Path(constants.HF_LEROBOT_HOME) / repo_id
    dataset = LeRobotDataset.create(
        repo_id,
        FPS,
        dataset_features(width, height),
        root=root,
        robot_type="so101_follower",
        use_videos=True,
        image_writer_threads=4 * len(CAMERAS),
    )
    report = GenerationReport(root=root)
    dropped: Counter[str] = Counter()
    attempts = 0
    error: BaseException | None = None
    jobs = ((seed, i, cell.to_dict(), asdict(params)) for i in itertools.count())
    results = ordered_map(plan_attempt, jobs, workers)
    try:
        while report.saved < episodes:
            if attempts >= max_attempts:
                raise RuntimeError(
                    f"Saved {report.saved}/{episodes} episodes in {attempts} attempts "
                    f"(max_attempts={max_attempts}); dropped {dict(dropped)}"
                )
            attempt = next(results)
            attempts += 1
            if attempt.failure is not None or attempt.actions is None:
                dropped[attempt.failure or "unknown"] += 1
                if progress:
                    progress(report.saved, episodes, attempts)
                continue

            position, transfer = episode_setup(seed, attempt.index)
            world = settled_world(cell, position, fps=FPS, width=width, height=height)
            try:
                overlay = GoalOverlayStep(calibration=world.board_calibration("top"))
                overlay.set_goal(*transfer)

                def record(
                    action: np.ndarray, world: ChessWorld = world, overlay: GoalOverlayStep = overlay
                ) -> None:
                    frame: dict[str, Any] = {
                        "observation.state": world.joint_state(),
                        "action": np.asarray(action, dtype=np.float32),
                        "task": CHESS_TASK,
                    }
                    for cam in CAMERAS:
                        image = world.render(cam)
                        if cam == "top":
                            image = overlay.observation({"top": image})["top"]
                        frame[f"observation.images.{cam}"] = image
                    dataset.add_frame(frame)

                outcome = _replay(world, transfer, attempt.actions, on_step=record)
            finally:
                world.close()
            if not outcome.success:  # the replay diverged from the physics-only pass; never record a failure
                dataset.clear_episode_buffer()
                dropped[NONDETERMINISTIC] += 1
            else:
                dataset.save_episode()
                write_transfer_sidecar(root, dataset.num_episodes - 1, transfer)
                report.saved += 1
            if progress:
                progress(report.saved, episodes, attempts)
    except BaseException as e:
        error = e
        raise
    finally:
        results.close()
        finalize_error: Exception | None = None
        try:
            dataset.finalize()
        except Exception as e:
            if error is None:
                finalize_error = error = e
            else:  # keep the exception that stopped the run; this one is a consequence
                logger.exception("Could not finalize %s after %s", root, type(error).__name__)
        report.dropped = dict(dropped)
        meta = {
            "source": "sim",
            "seed": seed,
            "episodes": report.saved,
            "attempts": attempts,
            "dropped": report.dropped,
            "interrupted": error is not None,
            "error": type(error).__name__ if error is not None else None,
            "cell": cell.to_dict(),
            "expert": asdict(params),
            "assets": {"repo": ASSETS_REPO, "revision": ASSETS_REVISION},
            "resolution": [width, height],
            "fps": FPS,
        }
        (root / "meta" / "garage_sim.json").write_text(json.dumps(meta, indent=2))
        if finalize_error is not None:
            raise finalize_error

    if log_artifact:
        from ..forge.tracking import log_dataset_if_configured

        log_dataset_if_configured(root, repo_id)
    return report
