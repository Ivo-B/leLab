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

"""The fixed task string, the evaluation transfer plan and the per-episode sidecar."""

from __future__ import annotations

import json
from pathlib import Path

from lelab.utils import config

from .game import BIN, PARK_BLACK, PARK_WHITE, Transfer

CHESS_TASK = "move the piece on the red square to the green square"
SIDECAR_NAME = "garage_transfers.json"

# The same 20 transfers run in simulation and on the real arm, so the sim-vs-real
# gap is one table: corners, centre, near/mid/far rows, the bin and both parking pads.
TRANSFER_PLAN: tuple[Transfer, ...] = (
    ("a1", "c3"),
    ("h1", "f3"),
    ("a8", "c6"),
    ("h8", "f6"),
    ("c3", "a1"),
    ("f3", "h1"),
    ("c6", "a8"),
    ("f6", "h8"),
    ("d4", "e5"),
    ("e5", "d4"),
    ("b2", "g7"),
    ("g7", "b2"),
    ("d1", "d8"),
    ("e8", "e1"),
    ("a4", "h5"),
    ("e2", "e3"),
    ("e4", BIN),
    ("b7", BIN),
    (PARK_WHITE, "c8"),
    (PARK_BLACK, "f1"),
)


def transfer_class(transfer: Transfer) -> str:
    """`bin`, `park`, or the row band of the farther square: near (1-3), mid (4-5), far (6-8)."""
    source, target = transfer
    if target == BIN:
        return "bin"
    if source in (PARK_WHITE, PARK_BLACK):
        return "park"
    rank = max(int(source[1]), int(target[1]))
    return "near" if rank <= 3 else "mid" if rank <= 5 else "far"


def _sidecar_path(dataset_root: Path) -> Path:
    return Path(dataset_root) / "meta" / SIDECAR_NAME


def read_transfer_sidecar(dataset_root: Path) -> list[dict]:
    path = _sidecar_path(dataset_root)
    if not path.is_file():
        return []
    return json.loads(path.read_text())


def write_transfer_sidecar(dataset_root: Path, episode_index: int, transfer: Transfer) -> Path:
    """Record which transfer episode `episode_index` shows. Re-writes an existing index."""
    path = _sidecar_path(dataset_root)
    entries = [e for e in read_transfer_sidecar(dataset_root) if e.get("episode_index") != episode_index]
    entries.append({"episode_index": episode_index, "source": transfer[0], "target": transfer[1]})
    entries.sort(key=lambda e: e["episode_index"])
    config._atomic_write_text(str(path), json.dumps(entries, indent=2))
    return path
