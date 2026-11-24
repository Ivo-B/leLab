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

import json
from pathlib import Path
from typing import Any

from . import episode_media
from .chess.transfers import CHESS_TASK, SIDECAR_NAME, read_transfer_sidecar

CHESS_JOINTS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
POLICY_CAMERAS = ("top", "wrist")


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
    other_tasks = sorted(
        {t for row in episode_media.read_episode_index(dataset_dir) for t in row.tasks} - {CHESS_TASK}
    )
    if other_tasks:
        problems.append(f"episodes with a task other than the chess task: {other_tasks}")
    total = int(info.get("total_episodes") or 0)
    try:
        covered = {entry.get("episode_index") for entry in read_transfer_sidecar(dataset_dir)}
    except (OSError, json.JSONDecodeError, AttributeError):
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


def mismatches(a: Path, b: Path) -> list[str]:
    info_a, info_b = episode_media.read_info(a), episode_media.read_info(b)
    found: list[str] = []
    if info_a.get("fps") != info_b.get("fps"):
        found.append(f"fps {info_a.get('fps')} vs {info_b.get('fps')}")
    features_a, features_b = info_a.get("features") or {}, info_b.get("features") or {}
    for key in sorted(set(features_a) | set(features_b)):
        if key not in features_a or key not in features_b:
            found.append(f"{key} only in {'the first' if key in features_a else 'the second'} dataset")
            continue
        sig_a, sig_b = _feature_signature(features_a[key]), _feature_signature(features_b[key])
        for label, value_a, value_b in zip(("dtype", "shape", "names"), sig_a, sig_b, strict=True):
            if value_a != value_b:
                found.append(f"{key} {label} {value_a} vs {value_b}")
    return found


def handle_chess_check(repo_id: str, against: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "checked": False,
        "repo_id": repo_id,
        "problems": [],
        "mismatches": [],
        "message": None,
    }
    try:
        dataset_dir = episode_media.resolve_dataset_dir(repo_id)
        other_dir = episode_media.resolve_dataset_dir(against) if against else None
    except episode_media.DatasetNotFoundError as exc:
        result["message"] = f"{exc}. Only local datasets can be checked."
        return result
    result["checked"] = True
    result["problems"] = chess_problems(dataset_dir)
    if other_dir is not None:
        result["mismatches"] = mismatches(dataset_dir, other_dir)
    return result
