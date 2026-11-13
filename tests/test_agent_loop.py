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

from lelab.agent.loop import (
    ROLLOUT_TIMEOUT_S,
    Action,
    ArmBusy,
    OrderLoop,
    item_counts,
    rollout_budget,
)


class ScriptedPlanner:
    def __init__(self, actions: list[Action]) -> None:
        self.actions = list(actions)

    def next_action(self, transcript: list[dict]) -> Action:
        del transcript
        if not self.actions:
            return Action("done")
        return self.actions.pop(0)


class FakeExecutor:
    def __init__(self) -> None:
        self.picks: list[tuple] = []
        self.calls: list[str] = []
        self.stage = "grasped"
        self.tray: dict[str, int] | None = None

    def look(self) -> dict:
        self.calls.append("look")
        return {"top": None, "side": None, "wrist": None}

    def pick(self, item: str, policy_family: str, policy_host: str, timeout_s: float) -> dict:
        self.calls.append("pick")
        self.picks.append((item, policy_family, policy_host, timeout_s))
        if self.stage == "busy":
            raise ArmBusy()
        return {"stage": self.stage, "duration_s": 1.0, "verdict_source": "judge"}

    def verify_tray(self) -> dict:
        self.calls.append("verify_tray")
        return {"counts": self.tray}

    def ask_operator(self, message: str) -> dict:
        self.calls.append("ask_operator")
        return {"answer": "ready", "message": message}


def test_item_counts_reads_the_kit_sentence() -> None:
    assert item_counts("Kit 7: two brackets and one bolt") == {"bracket": 2, "bolt": 1}
    assert item_counts("no parts here") == {}
    assert rollout_budget({"bracket": 2, "bolt": 1}) == 8


def test_unknown_item_aborts_without_a_pick() -> None:
    executor = FakeExecutor()
    result = OrderLoop(ScriptedPlanner([Action("tool", "pick", {"item": "washer"})]), executor).run(
        "one bolt", "smolvla", "local"
    )
    assert result.status == "aborted"
    assert result.reason == "unknown_item"
    assert executor.picks == []


def test_invalid_tool_aborts_without_calling_the_executor() -> None:
    executor = FakeExecutor()
    result = OrderLoop(ScriptedPlanner([Action("tool", "dance", {})]), executor).run(
        "one bolt", "smolvla", "local"
    )
    assert result.status == "aborted"
    assert result.reason == "invalid_tool"
    assert executor.calls == []


def test_two_retries_then_the_next_pick_is_refused() -> None:
    executor = FakeExecutor()
    picks = [Action("tool", "pick", {"item": "bolt"}) for _ in range(4)]
    result = OrderLoop(ScriptedPlanner(picks), executor).run("one bolt", "act", "local")
    assert result.status == "aborted"
    assert result.reason == "retry_cap"
    assert len(executor.picks) == 3
    assert executor.picks[0] == ("bolt", "act", "local", ROLLOUT_TIMEOUT_S)


def test_rollout_budget_stops_the_next_pick() -> None:
    executor = FakeExecutor()
    executor.stage = "delivered"
    picks = [Action("tool", "pick", {"item": "bolt"}) for _ in range(5)]
    result = OrderLoop(ScriptedPlanner(picks), executor).run("one bolt", "smolvla", "local")
    assert result.reason == "rollout_cap"
    assert len(executor.picks) == 4
    assert result.rollouts == 4


def test_arm_busy_aborts_and_does_not_retry() -> None:
    executor = FakeExecutor()
    executor.stage = "busy"
    result = OrderLoop(
        ScriptedPlanner([Action("tool", "pick", {"item": "bolt"}), Action("tool", "pick", {"item": "bolt"})]),
        executor,
    ).run("one bolt", "smolvla", "local")
    assert result.status == "aborted"
    assert result.reason == "arm_busy"
    assert executor.calls == ["pick"]


def test_stop_ends_the_loop_before_the_next_tool() -> None:
    executor = FakeExecutor()
    loop = OrderLoop(ScriptedPlanner([Action("tool", "look", {})]), executor)
    loop.stop.set()
    result = loop.run("one gear", "smolvla", "local")
    assert result.status == "stopped"
    assert executor.calls == []


def test_matching_tray_fulfills_the_order() -> None:
    executor = FakeExecutor()
    executor.tray = {"bolt": 1, "bracket": 0, "gear": 0}
    result = OrderLoop(ScriptedPlanner([Action("tool", "verify_tray", {})]), executor).run(
        "one bolt", "smolvla", "local"
    )
    assert result.status == "fulfilled"
    assert result.reason is None


def test_planner_exception_aborts_and_keeps_the_result() -> None:
    class Boom:
        def next_action(self, transcript: list[dict]) -> Action:
            del transcript
            raise RuntimeError("model down")

    result = OrderLoop(Boom(), FakeExecutor()).run("one bolt", "smolvla", "local")
    assert result.status == "aborted"
    assert result.reason == "planner_error"
    assert result.counts == {"bolt": 1}


def test_order_with_no_skill_does_not_call_the_planner() -> None:
    class Boom:
        def next_action(self, transcript: list[dict]) -> Action:
            raise AssertionError("planner should not be called")

    result = OrderLoop(Boom(), FakeExecutor()).run("hello", "smolvla", "local")
    assert result.reason == "no_items"
