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

import re
import threading
from dataclasses import dataclass, field

SKILLS = ("bolt", "bracket", "gear")
ROLLOUT_TIMEOUT_S = 30
_MAX_FAILED_PICKS = 3
_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_CANONICAL = {
    "bolt": "bolt",
    "bolts": "bolt",
    "bracket": "bracket",
    "brackets": "bracket",
    "gear": "gear",
    "gears": "gear",
}


class ArmBusy(Exception):  # noqa: N818
    pass


@dataclass
class Action:
    kind: str
    name: str = ""
    arguments: dict = field(default_factory=dict)
    detail: str = ""


@dataclass
class OrderResult:
    status: str
    reason: str | None
    counts: dict[str, int]
    steps: list[dict]
    rollouts: int


def item_counts(text: str) -> dict[str, int]:
    counts = dict.fromkeys(SKILLS, 0)
    pending: int | None = None
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if token.isdigit():
            pending = int(token)
            continue
        if token in _WORDS:
            pending = _WORDS[token]
            continue
        skill = _CANONICAL.get(token)
        if skill is None:
            continue
        counts[skill] += 1 if pending is None else pending
        pending = None
    return {skill: count for skill, count in counts.items() if count}


def rollout_budget(counts: dict[str, int]) -> int:
    return 2 * sum(counts.values()) + 2


def _matches(reported: object, wanted: dict[str, int]) -> bool:
    if not isinstance(reported, dict):
        return False
    return all(reported.get(skill, 0) == wanted.get(skill, 0) for skill in SKILLS)


class OrderLoop:
    def __init__(self, planner, executor, stop: threading.Event | None = None) -> None:
        self.planner = planner
        self.executor = executor
        self.stop = stop or threading.Event()
        self.failures: dict[str, int] = {}
        self.rollouts = 0
        self.steps: list[dict] = []

    def run(self, text: str, policy_family: str, policy_host: str) -> OrderResult:
        counts = item_counts(text)
        if not counts:
            return OrderResult("aborted", "no_items", {}, [], 0)
        budget = rollout_budget(counts)
        transcript: list[dict] = [{"role": "user", "content": text}]
        while True:
            if self.stop.is_set():
                return self._finish("stopped", None, counts)
            try:
                action = self.planner.next_action(transcript)
            except Exception:
                return self._finish("aborted", "planner_error", counts)
            if not isinstance(action, Action) or action.kind in {"done", "error"}:
                reason = (
                    "planner_error" if isinstance(action, Action) and action.kind == "error" else "incomplete"
                )
                if not isinstance(action, Action):
                    reason = "planner_error"
                return self._finish("aborted", reason, counts)
            if action.kind != "tool":
                return self._finish("aborted", "invalid_tool", counts)
            ended = self._tool(action, counts, budget, policy_family, policy_host)
            if ended is not None:
                return ended
            transcript.append({"role": "tool", "name": action.name, "content": self.steps[-1]["result"]})

    def _finish(self, status: str, reason: str | None, counts: dict[str, int]) -> OrderResult:
        return OrderResult(status, reason, counts, list(self.steps), self.rollouts)

    def _tool(
        self, action: Action, counts: dict[str, int], budget: int, policy_family: str, policy_host: str
    ) -> OrderResult | None:
        if action.name not in {"look", "pick", "verify_tray", "ask_operator"}:
            return self._finish("aborted", "invalid_tool", counts)
        if action.name == "pick":
            item = str(action.arguments.get("item", ""))
            if item not in SKILLS:
                return self._finish("aborted", "unknown_item", counts)
            if self.failures.get(item, 0) >= _MAX_FAILED_PICKS:
                return self._finish("aborted", "retry_cap", counts)
            if self.rollouts >= budget:
                return self._finish("aborted", "rollout_cap", counts)
            try:
                result = self.executor.pick(item, policy_family, policy_host, ROLLOUT_TIMEOUT_S)
            except ArmBusy:
                return self._finish("aborted", "arm_busy", counts)
            self.rollouts += 1
            if result.get("stage") != "delivered":
                self.failures[item] = self.failures.get(item, 0) + 1
            self.steps.append({"tool": "pick", "arguments": {"item": item}, "result": result})
            return None
        if action.name == "look":
            result = self.executor.look()
        elif action.name == "verify_tray":
            result = self.executor.verify_tray()
            self.steps.append({"tool": "verify_tray", "arguments": {}, "result": result})
            if _matches(result.get("counts"), counts):
                return self._finish("fulfilled", None, counts)
            return None
        else:
            result = self.executor.ask_operator(str(action.arguments.get("message", "")))
            if self.stop.is_set() or result.get("stopped"):
                self.steps.append({"tool": "ask_operator", "arguments": action.arguments, "result": result})
                return self._finish("stopped", None, counts)
        self.steps.append({"tool": action.name, "arguments": action.arguments, "result": result})
        return None
