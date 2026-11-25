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

"""Is a local dataset chess transfer data, and can two datasets be trained together?

Reads only metadata (`meta/info.json`, the episode index and `meta/garage_transfers.json`), so it is cheap enough
to run when a dataset is picked. The red/green markers are not detected in pixels: an episode has them exactly when
its transfer is in the sidecar, because the overlay and the sidecar come from the same recording step.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import episode_media
from .chess.transfers import CHESS_TASK, DATASET_CAMERAS, POLICY_CAMERAS, SIDECAR_NAME, read_transfer_sidecar

CHESS_JOINTS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


def chess_problems(dataset_dir: Path) -> list[str]:
    info = episode_media.read_info(dataset_dir)
    features = info.get("features") or {}
    problems: list[str] = []
    for key in ("action", "observation.state"):
        names = list((features.get(key) or {}).get("names") or [])
        if names != list(CHESS_JOINTS):
            problems.append(f"{key} joints are {names}, expected {list(CHESS_JOINTS)}")
    for camera in POLICY_CAMERAS:
        key = f"{episode_media.VIDEO_KEY_PREFIX}{camera}"
        if (features.get(key) or {}).get("dtype") != "video":
            problems.append(f"no {key} video")
    rows = episode_media.read_episode_index(dataset_dir)
    total = int(info.get("total_episodes") or 0)
    if total == 0:
        problems.append("the dataset has no episodes")
    elif not rows:
        problems.append("no episode index in meta/episodes (the recording did not finish writing)")
    other_tasks = sorted({t for row in rows for t in row.tasks} - {CHESS_TASK})
    if other_tasks:
        problems.append(f"episodes with a task other than the chess task: {other_tasks}")
    try:
        covered = {entry.get("episode_index") for entry in read_transfer_sidecar(dataset_dir)}
    except (OSError, ValueError, TypeError, AttributeError):
        problems.append(f"meta/{SIDECAR_NAME} is unreadable")
        return problems
    missing = sum(1 for index in range(total) if index not in covered)
    if missing:
        problems.append(
            f"{missing} of {total} episodes have no transfer in meta/{SIDECAR_NAME} "
            "(recorded without the red/green markers)"
        )
    return problems


def _feature_signature(feature: dict[str, Any]) -> tuple[Any, Any, Any]:
    return feature.get("dtype"), list(feature.get("shape") or []), feature.get("names")


def dataset_cameras(dataset_dir: Path) -> list[str]:
    """The dataset's video cameras: the contract's cameras first, in contract order, then any others."""
    features = episode_media.read_info(dataset_dir).get("features") or {}
    prefix = episode_media.VIDEO_KEY_PREFIX
    found = [
        k[len(prefix) :] for k, v in features.items() if k.startswith(prefix) and v.get("dtype") == "video"
    ]
    ordered = [c for c in DATASET_CAMERAS if c in found]
    return ordered + sorted(set(found) - set(ordered))


def _compared_keys(keys: set[str], cameras: Sequence[str] | None) -> list[str]:
    """The feature keys to compare: all of them, or everything but the cameras outside `cameras`."""
    if cameras is None:
        return sorted(keys)
    prefix = episode_media.VIDEO_KEY_PREFIX
    wanted = {f"{prefix}{camera}" for camera in cameras}
    return sorted({k for k in keys if not k.startswith(prefix)} | wanted)


def mismatches(a: Path, b: Path, cameras: Sequence[str] | None = None) -> list[str]:
    """How two datasets differ. With `cameras`, only those cameras are compared; a policy that reads
    top and wrist can be trained on data that also has side."""
    info_a, info_b = episode_media.read_info(a), episode_media.read_info(b)
    found: list[str] = []
    if info_a.get("fps") != info_b.get("fps"):
        found.append(f"fps {info_a.get('fps')} vs {info_b.get('fps')}")
    features_a, features_b = info_a.get("features") or {}, info_b.get("features") or {}
    for key in _compared_keys(set(features_a) | set(features_b), cameras):
        if key not in features_a or key not in features_b:
            if key not in features_a and key not in features_b:
                found.append(f"{key} is in neither dataset")
            else:
                found.append(f"{key} only in {'the first' if key in features_a else 'the second'} dataset")
            continue
        sig_a, sig_b = _feature_signature(features_a[key]), _feature_signature(features_b[key])
        for label, value_a, value_b in zip(("dtype", "shape", "names"), sig_a, sig_b, strict=True):
            if value_a != value_b:
                found.append(f"{key} {label} {value_a} vs {value_b}")
    return found


def handle_chess_check(
    repo_id: str, against: str | None = None, cameras: Sequence[str] | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "checked": False,
        "repo_id": repo_id,
        "problems": [],
        "mismatches": [],
        "cameras": [],
        "shared_cameras": [],
        "message": None,
    }
    try:
        dataset_dir = episode_media.resolve_dataset_dir(repo_id)
        result["problems"] = chess_problems(dataset_dir)
    except episode_media.DatasetNotFoundError as exc:
        result["message"] = f"{exc}. Only local datasets can be checked."
        return result
    result["checked"] = True
    result["cameras"] = dataset_cameras(dataset_dir)
    if against:
        try:
            other = episode_media.resolve_dataset_dir(against)
            result["mismatches"] = mismatches(dataset_dir, other, cameras)
            result["shared_cameras"] = [c for c in result["cameras"] if c in dataset_cameras(other)]
        except episode_media.DatasetNotFoundError as exc:
            result["message"] = (
                f"{exc}. The comparison dataset is not local or not readable, so it was not compared."
            )
    return result
