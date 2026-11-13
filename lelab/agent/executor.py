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

from collections.abc import Callable
from typing import Any

from lelab.agent.loop import ArmBusy
from lelab.forge.registry import collection_name


def _needs_review(*_args: object, **_kwargs: object) -> dict[str, Any]:
    return {"stage": None, "duration_s": 0.0, "verdict_source": "needs_review"}


class ToolExecutor:
    def __init__(
        self,
        capture: Callable[[], dict[str, Any]] | None = None,
        run_skill: Callable[..., dict[str, Any]] | None = None,
        count_tray: Callable[[], Any] | None = None,
        ask: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self._capture = capture
        self._run_skill = run_skill or _needs_review
        self._count_tray = count_tray
        self._ask = ask

    def look(self) -> dict[str, Any]:
        frames = self._capture() if self._capture is not None else {}
        if not isinstance(frames, dict):
            frames = {}
        return {
            "top": frames.get("top"),
            "side": frames.get("side"),
            "wrist": frames.get("wrist"),
        }

    def pick(self, item: str, policy_family: str, policy_host: str, timeout_s: float) -> dict[str, Any]:
        import lelab.record as record
        import lelab.replay as replay
        import lelab.rollout as rollout
        import lelab.teleoperate as teleoperate

        if (
            record.recording_active
            or teleoperate.teleoperation_active
            or rollout.inference_active
            or replay.replay_active
        ):
            raise ArmBusy()

        if policy_family == "smolvla":
            collection = collection_name("smolvla", None)
        elif policy_family == "act":
            collection = collection_name("act", item)
        else:
            collection = collection_name(policy_family, item)

        result = dict(self._run_skill(item, policy_family, policy_host, timeout_s))
        result["collection"] = collection
        result["alias"] = "production"
        result["policy_host"] = policy_host
        return result

    def verify_tray(self) -> dict[str, Any]:
        counts = None if self._count_tray is None else self._count_tray()
        if not isinstance(counts, dict):
            return {"counts": None}
        return {"counts": counts}

    def ask_operator(self, message: str) -> dict[str, Any]:
        if self._ask is None:
            return {"message": message}
        return self._ask(message)
