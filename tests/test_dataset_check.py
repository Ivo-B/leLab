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

"""chess_problems and mismatches over hand-written LeRobot v3.0 metadata. No torch, no videos."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from lelab import dataset_check, server
from lelab.chess.transfers import CHESS_TASK, write_transfer_sidecar

JOINTS = list(dataset_check.CHESS_JOINTS)


def make_dataset(
    root: Path,
    *,
    episodes: int = 2,
    task: str = CHESS_TASK,
    sidecar: bool = True,
    cameras: tuple[str, ...] = ("top", "wrist", "side"),
    joints: list[str] = JOINTS,
    size: tuple[int, int] = (480, 640),
    fps: int = 30,
) -> Path:
    meta = root / "meta"
    (meta / "episodes" / "chunk-000").mkdir(parents=True)
    vector = {"dtype": "float32", "shape": [len(joints)], "names": joints}
    features = {"action": vector, "observation.state": vector}
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [*size, 3],
            "names": ["height", "width", "channels"],
        }
    (meta / "info.json").write_text(
        json.dumps({"fps": fps, "total_episodes": episodes, "features": features})
    )
    pd.DataFrame(
        {
            "episode_index": list(range(episodes)),
            "length": [10] * episodes,
            "tasks": [[task]] * episodes,
        }
    ).to_parquet(meta / "episodes" / "chunk-000" / "file-000.parquet")
    if sidecar:
        for i in range(episodes):
            write_transfer_sidecar(root, i, ("e2", "e4"))
    return root


def test_a_chess_transfer_dataset_has_no_problems(tmp_path: Path) -> None:
    assert dataset_check.chess_problems(make_dataset(tmp_path / "d")) == []


def test_a_plain_teleop_dataset_lists_every_gap(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "d", task="pick the bolt", sidecar=False, cameras=("wrist",))
    problems = dataset_check.chess_problems(root)
    assert any("observation.images.top" in p for p in problems)
    assert any("pick the bolt" in p for p in problems)
    assert any("2 of 2 episodes have no transfer" in p for p in problems)


def test_partial_sidecar_and_wrong_joints(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "d", episodes=3, sidecar=False, joints=["a", "b"])
    write_transfer_sidecar(root, 0, ("e2", "e4"))
    problems = dataset_check.chess_problems(root)
    assert any("2 of 3 episodes have no transfer" in p for p in problems)
    assert any(p.startswith("action joints") for p in problems)


def test_unreadable_sidecar_is_a_problem_not_a_crash(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "d", sidecar=False)
    (root / "meta" / "garage_transfers.json").write_text("{not json")
    assert any("garage_transfers.json" in p for p in dataset_check.chess_problems(root))


def test_mismatches_compare_fps_cameras_and_shapes(tmp_path: Path) -> None:
    sim = make_dataset(tmp_path / "sim")
    real = make_dataset(tmp_path / "real", cameras=("top", "wrist"), size=(1080, 1920), fps=15)
    found = dataset_check.mismatches(sim, real)
    assert any(p.startswith("fps") for p in found)
    assert any("observation.images.side" in p for p in found)
    assert any("observation.images.top" in p and "shape" in p for p in found)
    assert dataset_check.mismatches(sim, make_dataset(tmp_path / "same")) == []


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    return tmp_path


def test_route_checks_a_local_dataset_against_another(cache: Path) -> None:
    make_dataset(cache / "local" / "sim")
    make_dataset(cache / "local" / "real", sidecar=False)
    body = (
        TestClient(server.app)
        .get("/datasets/chess-check", params={"repo_id": "local/real", "against": "local/sim"})
        .json()
    )
    assert body["checked"] is True
    assert body["repo_id"] == "local/real"
    assert any("no transfer" in p for p in body["problems"])
    assert body["mismatches"] == []


def test_route_does_not_fail_for_a_hub_only_dataset(cache: Path) -> None:
    response = TestClient(server.app).get("/datasets/chess-check", params={"repo_id": "someone/remote"})
    assert response.status_code == 200
    assert response.json()["checked"] is False
    assert response.json()["message"]
