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

"""`python -m lelab.sim generate | gate | preview`."""

from __future__ import annotations

import argparse
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

GATE_RATE = 0.9
CLASSES = ("near", "mid", "far", "bin", "park")


def gate_table(results: Counter[tuple[str, str]]) -> tuple[str, float]:
    """Markdown table of successes and failures per transfer class, and the overall success rate."""
    lines = ["| class | success | failures |", "|---|---|---|"]
    classes = [c for c in CLASSES if any(k[0] == c for k in results)]
    classes += sorted({k[0] for k in results} - set(classes))
    for cls in classes:
        rows = {result: n for (c, result), n in results.items() if c == cls}
        failures = ", ".join(f"{r} {n}" for r, n in sorted(rows.items()) if r != "success") or "—"
        lines.append(f"| {cls} | {rows.get('success', 0)}/{sum(rows.values())} | {failures} |")
    total = sum(results.values())
    wins = sum(n for (_, r), n in results.items() if r == "success")
    lines.append(f"| **total** | **{wins}/{total}** | |")
    return "\n".join(lines), wins / total if total else 0.0


def _gate(seeds: int, workers: int) -> int:
    from .generate import run_gate

    table, rate = gate_table(run_gate(seeds, workers=workers))
    print(table)
    print(f"success rate {rate:.1%}")
    return 0 if rate >= GATE_RATE else 1


def _preview(source: str, target: str, out: str) -> int:
    from PIL import Image

    from .env import ChessTransferEnv

    env = ChessTransferEnv()
    try:
        obs, _ = env.reset(seed=0, options={"transfer": (source, target)})
    finally:
        env.close()
    Image.fromarray(np.concatenate([obs["pixels"][c] for c in ("top", "side", "wrist")], axis=1)).save(out)
    print(out)
    return 0


def main(argv: list[str] | None = None) -> int:
    from .generate import default_workers

    parser = argparse.ArgumentParser(prog="python -m lelab.sim")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="record expert transfers into a LeRobotDataset")
    gen.add_argument("--repo-id", required=True)
    gen.add_argument("--episodes", type=int, default=1000)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--width", type=int, default=640)
    gen.add_argument("--height", type=int, default=480)
    gen.add_argument("--root", default=None, help="dataset directory (default: HF_LEROBOT_HOME/<repo-id>)")
    gen.add_argument("--no-artifact", action="store_true", help="do not log the dataset to W&B")
    gen.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="give up after this many attempts (default 10 * episodes + 100)",
    )
    gen.add_argument(
        "--workers", type=int, default=default_workers(), help="processes for the physics-only passes"
    )
    gate = sub.add_parser("gate", help="expert success on TRANSFER_PLAN x seeds")
    gate.add_argument("--seeds", type=int, default=5)
    gate.add_argument("--workers", type=int, default=default_workers())
    preview = sub.add_parser("preview", help="save top|side|wrist after reset")
    preview.add_argument("--transfer", nargs=2, default=["e2", "e4"], metavar=("SOURCE", "TARGET"))
    preview.add_argument("--out", default=str(Path(tempfile.gettempdir()) / "garage_sim_preview.png"))
    args = parser.parse_args(argv)

    if args.command == "gate":
        return _gate(args.seeds, args.workers)
    if args.command == "preview":
        return _preview(*args.transfer, args.out)
    from .generate import generate_dataset

    report = generate_dataset(
        args.repo_id,
        args.episodes,
        root=Path(args.root) if args.root else None,
        seed=args.seed,
        width=args.width,
        height=args.height,
        log_artifact=not args.no_artifact,
        progress=lambda saved, total, attempts: print(
            f"\rsaved {saved}/{total} after {attempts} attempts", end="", flush=True
        ),
        workers=args.workers,
        max_attempts=args.max_attempts,
    )
    print(f"\nsaved {report.saved} episodes to {report.root}; dropped {report.dropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
