# LeLab Forge M5 — Order Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A free-text kitting order is parsed, fulfilled by a tool-calling planner under hard guardrails, traced for Agent Lens, and shown on the Orders page. The arm does not move in a unit test.

**Architecture:** `item_counts` reads the order. `OrderLoop` asks a planner for one action at a time and calls `look`, `pick`, `verify_tray`, or `ask_operator`. The loop, not the model, enforces known skills, two retries per item, the rollout budget, the arm mutex, and stop. `InferencePlanner` calls Serverless Inference. `ToolExecutor` resolves the `production` alias and calls replaceable functions; its default skill runner returns `needs_review` and imports no robot code. `export_trace` lazy-imports the Agent Lens SDK. One background session serves the Orders page.

**Tech Stack:** Python 3.12, FastAPI, `openai` against `https://api.inference.wandb.ai/v1`, `coreweave` Agent Lens SDK, React + Vite.

## Global Constraints

- Work on branch `forge/m5-orders`. Push only to `origin` (`https://github.com/Ivo-B/leLab.git`). Do not push `upstream`.
- Do not start M6. No Automations, ARIA, curation notebook, or README storyline.
- Do not start the robot arm in a unit test. Do not import `cwsandbox` from `lelab/agent/`. The default skill runner returns `{"stage": None, "duration_s": 0.0, "verdict_source": "needs_review"}` and does not call a rollout.
- Skills are `bolt`, `bracket`, and `gear`. A pick whose stage is not `delivered` is a failed pick. Two retries means three failed picks may run; the fourth pick of that item aborts before the executor.
- Rollout budget is `2 * total_items + 2`. Timeout passed to every pick is `30` seconds.
- Planner model is `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B`, override `FORGE_PLANNER_MODEL`. No image-byte limit is published; drop a frame larger than `MAX_FRAME_BYTES` (`200_000`) before it is sent. Never default an order to fulfilled.
- Agent Lens uses `coreweave.forge.agentlens.tracing` with `autopatch_integrations=False`. Do not invent a trace URL. `trace_url` stays `None` unless the SDK returns one.
- `WANDB_API_KEY` and `CWSANDBOX_API_KEY` never appear in a response, a log line, or a logger call.
- New Python modules start with the Apache-2.0 header in `tests/test_forge_config.py` lines 1–13, then a blank line.
- Pytest is not on `PATH`. Run `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest`.
- Commit only the files the task lists. Do not skip hooks.

Checked on 2026-11-13: the Forge catalogue page for `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B` lists vision, tool calling, and structured outputs. The Agent Lens docs say a custom agent should call `tracing.init`, `start_conversation`, `start_turn`, and `start_tool`; raw OTLP is the lower-level path and is not used here.

---

### Task 1: Stop the loop at the guardrails

**Files:**

- Create: `lelab/agent/__init__.py`
- Create: `lelab/agent/loop.py`
- Create: `tests/test_agent_loop.py`
- Modify: `docs/superpowers/plans/2026-11-13-lelab-forge-m5-orders.md` (this plan, if it is still untracked)

**Interfaces:**

- Consumes: nothing from later tasks.
- Produces: `SKILLS`, `ROLLOUT_TIMEOUT_S = 30`, `item_counts(text: str) -> dict[str, int]`, `rollout_budget(counts: dict[str, int]) -> int`, `Action`, `ArmBusy`, `OrderResult`, `OrderLoop`. `OrderLoop.run(text, policy_family, policy_host) -> OrderResult`. `OrderResult` fields are `status`, `reason`, `counts`, `steps`, `rollouts`. `status` is `fulfilled`, `aborted`, or `stopped`.

- [x] **Step 1: Write the failing test**

```python
import pytest

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
```

- [x] **Step 2: Run the test to verify it fails**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_agent_loop.py -q`

Expected: FAIL with `ModuleNotFoundError`

- [x] **Step 3: Write the minimal implementation**

`lelab/agent/__init__.py` is only the Apache-2.0 header.

`lelab/agent/loop.py`:

```python
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


class ArmBusy(Exception):
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
    counts = {skill: 0 for skill in SKILLS}
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
                reason = "planner_error" if isinstance(action, Action) and action.kind == "error" else "incomplete"
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

    def _tool(self, action: Action, counts: dict[str, int], budget: int, policy_family: str, policy_host: str) -> OrderResult | None:
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
```

- [x] **Step 4: Run the test to verify it passes**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_agent_loop.py -q`

Expected: all passed.

- [x] **Step 5: Commit**

```bash
git add lelab/agent/__init__.py lelab/agent/loop.py tests/test_agent_loop.py docs/superpowers/plans/2026-11-13-lelab-forge-m5-orders.md
git commit -m "$(cat <<'EOF'
feat: stop an order at the guardrails

EOF
)"
```

---

### Task 2: Ask the planner model for the next tool

**Files:**

- Create: `lelab/agent/planner.py`
- Create: `tests/test_agent_planner.py`
- Modify: `lelab/utils/forge_config.py`
- Modify: `tests/test_forge_config.py` only if the existing equality assertion fails because of the new field

**Interfaces:**

- Consumes: `Action` from `lelab.agent.loop`. `ForgeConfig.judge_model` stays as it is.
- Produces: `ForgeConfig.planner_model: str`, loaded from `FORGE_PLANNER_MODEL` and otherwise `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B`. `action_from_message(message) -> Action`. `InferencePlanner(client, cfg).next_action(transcript) -> Action`. `TOOLS` is a list of four OpenAI function tools named `look`, `pick`, `verify_tray`, and `ask_operator`. `pick` requires `item` with enum `bolt`, `bracket`, `gear`. `ask_operator` requires `message`.

- [ ] **Step 1: Write the failing test**

```python
import json

from lelab.agent.planner import TOOLS, InferencePlanner, action_from_message
from lelab.utils.forge_config import ForgeConfig


class FakeFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class FakeCall:
    def __init__(self, name: str, arguments: str) -> None:
        self.function = FakeFunction(name, arguments)


class FakeMessage:
    def __init__(self, calls) -> None:
        self.tool_calls = calls
        self.content = None


def test_a_pick_tool_call_becomes_an_action() -> None:
    message = FakeMessage([FakeCall("pick", json.dumps({"item": "gear"}))])
    action = action_from_message(message)
    assert action.kind == "tool"
    assert action.name == "pick"
    assert action.arguments == {"item": "gear"}


def test_no_tool_call_is_done() -> None:
    assert action_from_message(FakeMessage(None)).kind == "done"


def test_broken_arguments_are_an_error() -> None:
    assert action_from_message(FakeMessage([FakeCall("look", "{")]) ).kind == "error"


def test_planner_sends_the_catalogue_model_and_drops_a_large_frame() -> None:
    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)

            class Choice:
                message = FakeMessage([FakeCall("look", "{}")])

            class Response:
                choices = [Choice()]

            return Response()

    class Client:
        chat = type("Chat", (), {"completions": Completions()})()

    cfg = ForgeConfig(api_key="secret", entity="team", project="so101-kitting")
    transcript = [
        {"role": "user", "content": "one bolt"},
        {"role": "tool", "name": "look", "content": {"top": b"a" * 10, "side": b"b" * 200_001}},
    ]
    action = InferencePlanner(Client(), cfg).next_action(transcript)
    assert action.name == "look"
    assert captured["model"] == "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B"
    assert captured["tools"] == TOOLS
    encoded = json.dumps(captured["messages"])
    assert "secret" not in encoded
    assert "aaaa" in encoded
    assert "bbbb" not in encoded
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_agent_planner.py -q`

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the minimal implementation**

Add `planner_model: str = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B"` to `ForgeConfig` after `judge_model`. In `load_forge_config`, set `planner_model` from `FORGE_PLANNER_MODEL` with the same fallback string used for `judge_model`.

`action_from_message` reads `message.tool_calls`. `None` or an empty list returns `Action("done")`. `json.loads` of the first call's arguments returns `Action("error")` on `JSONDecodeError` or when the value is not a dict. Otherwise it returns `Action("tool", name, arguments)`.

`InferencePlanner.next_action` copies the transcript. A tool content dict drops any `bytes` value whose length is greater than `200_000`, and replaces a smaller `bytes` value with `base64.b64encode(value).decode("ascii")`. It calls `client.chat.completions.create(model=cfg.planner_model, messages=messages, tools=TOOLS)` and returns `action_from_message` of the first choice. It does not log `cfg.api_key`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_agent_planner.py tests/test_forge_config.py -q`

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add lelab/agent/planner.py tests/test_agent_planner.py lelab/utils/forge_config.py tests/test_forge_config.py
git commit -m "$(cat <<'EOF'
feat: ask the planner model for the next tool

EOF
)"
```

---

### Task 3: Bind the tools without moving the arm

**Files:**

- Create: `lelab/agent/executor.py`
- Create: `tests/test_agent_executor.py`

**Interfaces:**

- Consumes: `ArmBusy` and `SKILLS` from `lelab.agent.loop`. `collection_name` from `lelab.forge.registry`.
- Produces: `ToolExecutor(capture=None, run_skill=None, count_tray=None, ask=None)`. `look()` returns `{"top", "side", "wrist"}` from `capture()`, defaulting each to `None`. `pick(item, policy_family, policy_host, timeout_s)` raises `ArmBusy` when `recording_active`, `teleoperation_active`, `inference_active`, or `replay_active` is true, then returns the skill-runner dict plus `collection`, `alias="production"`, and `policy_host`. SmolVLA uses `collection_name("smolvla", None)`. ACT uses `collection_name("act", item)`. `verify_tray()` returns `{"counts": None}` when `count_tray` returns a non-dict. `ask_operator(message)` returns the `ask` callable's dict.

- [ ] **Step 1: Write the failing test**

```python
import lelab.record as record
import lelab.replay as replay
import lelab.rollout as rollout
import lelab.teleoperate as teleoperate
from lelab.agent.executor import ToolExecutor
from lelab.agent.loop import ArmBusy


def _idle(monkeypatch) -> None:
    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(teleoperate, "teleoperation_active", False)
    monkeypatch.setattr(rollout, "inference_active", False)
    monkeypatch.setattr(replay, "replay_active", False)


def test_pick_records_the_production_collection_and_does_not_move_the_arm(monkeypatch) -> None:
    _idle(monkeypatch)
    result = ToolExecutor().pick("bracket", "act", "local", 30)
    assert result["collection"] == "so101-kitting-act-bracket"
    assert result["alias"] == "production"
    assert result["stage"] is None
    assert result["verdict_source"] == "needs_review"
    smolvla = ToolExecutor().pick("bracket", "smolvla", "coreweave", 30)
    assert smolvla["collection"] == "so101-kitting-smolvla"
    assert smolvla["policy_host"] == "coreweave"


def test_pick_refuses_while_inference_holds_the_arm(monkeypatch) -> None:
    _idle(monkeypatch)
    monkeypatch.setattr(rollout, "inference_active", True)
    try:
        ToolExecutor().pick("bolt", "smolvla", "local", 30)
    except ArmBusy:
        return
    raise AssertionError("expected ArmBusy")


def test_verify_without_counts_is_not_a_match() -> None:
    assert ToolExecutor().verify_tray() == {"counts": None}
    assert ToolExecutor(capture=lambda: {"top": b"top", "side": None, "wrist": None}).look()["top"] == b"top"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_agent_executor.py -q`

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the minimal implementation**

Implement `ToolExecutor` as the interface describes. Read the four `*_active` flags inside `pick`, not at import time. The default `run_skill` ignores its arguments and returns `{"stage": None, "duration_s": 0.0, "verdict_source": "needs_review"}`. Do not import a robot class, `cwsandbox`, or `openai`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_agent_executor.py -q`

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add lelab/agent/executor.py tests/test_agent_executor.py
git commit -m "$(cat <<'EOF'
feat: bind order tools without moving the arm

EOF
)"
```

---

### Task 4: Keep an Agent Lens trace for the order

**Files:**

- Create: `lelab/agent/tracing.py`
- Create: `tests/test_agent_tracing.py`
- Modify: `pyproject.toml`

**Interfaces:**

- Consumes: `OrderResult` from `lelab.agent.loop`. `ForgeConfig` from `lelab.utils.forge_config`.
- Produces: `export_trace(result: OrderResult, cfg: ForgeConfig, order_text: str) -> str | None`. A missing `coreweave` import returns `None` and does not raise. When the SDK imports, call `tracing.init(f"{cfg.entity}/{cfg.project}", api_key=cfg.api_key, autopatch_integrations=False)`, then `start_conversation(agent_name="so101-kitting", conversation_name="order")`, `start_turn(user_message=order_text, agent_name="so101-kitting", model=cfg.planner_model)`, and `start_tool(name=step["tool"], arguments=json.dumps(step["arguments"]))` for each step. Set `tool.result` to `json.dumps` of the step result with any `bytes` replaced by `{"omitted_bytes": len(value)}`. End each context manager. Return `None`. Do not build a URL.

- [ ] **Step 1: Write the failing test**

```python
import json

from lelab.agent.loop import OrderResult
from lelab.agent.tracing import export_trace
from lelab.utils.forge_config import ForgeConfig


def test_missing_sdk_returns_no_url(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "coreweave" or name.startswith("coreweave."):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    result = OrderResult("aborted", "planner_error", {"bolt": 1}, [], 0)
    cfg = ForgeConfig(api_key="secret", entity="team", project="so101-kitting")
    assert export_trace(result, cfg, "one bolt") is None


def test_export_opens_a_conversation_and_one_tool_span(monkeypatch) -> None:
    opened = []

    class Span:
        def __init__(self, kind: str, **kwargs) -> None:
            self.kind = kind
            self.kwargs = kwargs
            self.result = None

        def __enter__(self):
            opened.append(self)
            return self

        def __exit__(self, *args):
            return False

    class FakeTracing:
        @staticmethod
        def init(project_name, api_key=None, autopatch_integrations=True):
            opened.append(("init", project_name, api_key, autopatch_integrations))

        @staticmethod
        def start_conversation(**kwargs):
            return Span("conversation", **kwargs)

        @staticmethod
        def start_turn(**kwargs):
            return Span("turn", **kwargs)

        @staticmethod
        def start_tool(**kwargs):
            return Span("tool", **kwargs)

    monkeypatch.setattr("lelab.agent.tracing._load_tracing", lambda: FakeTracing)
    result = OrderResult(
        "fulfilled",
        None,
        {"bolt": 1},
        [{"tool": "look", "arguments": {}, "result": {"top": b"abc"}}],
        0,
    )
    cfg = ForgeConfig(api_key="secret", entity="team", project="so101-kitting")
    assert export_trace(result, cfg, "one bolt") is None
    assert opened[0] == ("init", "team/so101-kitting", "secret", False)
    assert opened[1].kwargs["agent_name"] == "so101-kitting"
    tool = opened[3]
    assert tool.kwargs["name"] == "look"
    assert "abc" not in tool.result
    assert json.loads(tool.result)["top"] == {"omitted_bytes": 3}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_agent_tracing.py -q`

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the minimal implementation**

`_load_tracing` imports `coreweave.forge.agentlens.tracing` and returns that module. `export_trace` calls it inside `try/except Exception` and returns `None` on failure. Add `"coreweave>=0.1.0b0"` to `pyproject.toml` dependencies after `openai`. Install it:

`uv pip install --python /Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python 'coreweave>=0.1.0b0'`

Do not call `tracing.init` from the test. Do not log the API key.

- [ ] **Step 4: Run the test to verify it passes**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_agent_tracing.py -q`

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add lelab/agent/tracing.py tests/test_agent_tracing.py pyproject.toml
git commit -m "$(cat <<'EOF'
feat: keep an Agent Lens trace for an order

EOF
)"
```

---

### Task 5: Run one order from the API

**Files:**

- Create: `lelab/agent/session.py`
- Modify: `lelab/server.py`
- Create: `tests/test_agent_session.py`

**Interfaces:**

- Consumes: `OrderLoop`, `Action`, `OrderResult`, `InferencePlanner`, `ToolExecutor`, `export_trace`, `load_forge_config`, `forge_configured`.
- Produces: `OrderRequest` with `text: str`, `policy_family: Literal["smolvla", "act"]`, `policy_host: Literal["local", "coreweave"]`. `handle_start_order(request) -> dict`, `handle_order_status() -> dict`, `handle_stop_order() -> dict`, `handle_answer(message: str) -> dict`, `reset_session() -> None`. `OrderBusyError`. Routes: `POST /orders`, `GET /orders/current`, `POST /orders/current/stop`, `POST /orders/current/answer` with body `{"message": str}`. ValueError is HTTP 400. `OrderBusyError` is HTTP 409. The worker thread runs the loop. The start response is `{"status": "running", "counts": ...}` and returns before the loop finishes. `session.py` imports `OrderLoop` at module level so tests can replace `lelab.agent.session.OrderLoop`.

- [ ] **Step 1: Write the failing test**

```python
import time

import pytest
from fastapi.testclient import TestClient

from lelab.agent.loop import OrderResult
from lelab.agent.session import reset_session
from lelab.server import app

client = TestClient(app)


class BlockingLoop:
    def __init__(self, planner, executor, stop=None) -> None:
        del planner, executor
        self.stop = stop

    def run(self, text, policy_family, policy_host):
        del text, policy_family, policy_host
        assert self.stop is not None
        self.stop.wait(timeout=2)
        return OrderResult("stopped", None, {"bolt": 1}, [], 0)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.setattr("lelab.agent.session.OrderLoop", BlockingLoop)
    monkeypatch.setattr("lelab.agent.session.export_trace", lambda *args, **kwargs: None)
    reset_session()
    yield
    reset_session()


def test_forge_off_is_400() -> None:
    response = client.post("/orders", json={"text": "one bolt", "policy_family": "smolvla", "policy_host": "local"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Orders need Weights & Biases"


def test_second_order_is_409_and_stop_finishes_the_worker(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "local-test-key")
    monkeypatch.setenv("WANDB_ENTITY", "team")
    body = {"text": "one bolt", "policy_family": "smolvla", "policy_host": "local"}
    first = client.post("/orders", json=body)
    assert first.status_code == 200
    assert first.json()["status"] == "running"
    assert first.json()["counts"] == {"bolt": 1}
    second = client.post("/orders", json=body)
    assert second.status_code == 409
    stopped = client.post("/orders/current/stop")
    assert stopped.status_code == 200
    status = {}
    for _ in range(20):
        status = client.get("/orders/current").json()
        if status["status"] == "stopped":
            break
        time.sleep(0.05)
    assert status["status"] == "stopped"
    assert client.post("/orders/current/answer", json={"message": "go"}).status_code == 400
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_agent_session.py -q`

Expected: FAIL with 404 or `ImportError`.

- [ ] **Step 3: Write the minimal implementation**

Hold `_lock` only while checking and storing `_session`. Start a daemon thread for `OrderLoop.run`. `handle_stop_order` sets the session stop event. `handle_answer` sets the pending answer and its event, or raises `ValueError` when nothing is waiting. After `run` returns, call `export_trace` and store `trace_url`. A second start while `status == "running"` raises `OrderBusyError("An order is already running")`. Forge off raises `ValueError("Orders need Weights & Biases")`. Do not call the real planner in this task's tests. Do not log the API key.

- [ ] **Step 4: Run the test to verify it passes**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_agent_session.py tests/test_agent_loop.py -q`

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add lelab/agent/session.py lelab/server.py tests/test_agent_session.py
git commit -m "$(cat <<'EOF'
feat: run one order session from the API

EOF
)"
```

---

### Task 6: Show the order on the Orders page

**Files:**

- Create: `frontend/src/pages/Orders.tsx`
- Create: `frontend/src/pages/Orders.test.tsx`
- Modify: `frontend/src/App.tsx`

**Interfaces:**

- Consumes: `POST /orders`, `GET /orders/current`, `POST /orders/current/stop`. `forge_configured` and `sandbox_configured` from `GET /forge-status`, using the same strict `=== true` checks as `Evaluate.tsx`.
- Produces: Forge off renders the existing sentence `The order-picking agent arrives in a later milestone.` Forge on renders a text field, policy family `smolvla`/`act`, policy host `local`/`coreweave`, and one primary button `Start order`. CoreWeave is disabled unless `sandbox_configured === true`, with the hint `Add CWSANDBOX_API_KEY and CWSANDBOX_TRAIN_IMAGE to the backend environment.` After start, the page shows each count (`bolt: 1`), the status word, and `Stop`. `Open trace in Agent Lens` is an anchor only when `trace_url` is a non-empty string. Three tiles are labeled `Wrist`, `Top`, and `Side`, each saying `No frame yet`. Empty state before start: `Type an order, then the agent picks each part into the tray.`

- [ ] **Step 1: Write the failing test**

Follow `frontend/src/pages/Evaluate.test.tsx`: `stubFetch`, the local `userEvent` helper, and `ApiProvider`. Cover the placeholder, the empty state, a disabled CoreWeave option, a start that posts `{text, policy_family: "smolvla", policy_host: "local"}`, the checklist text `bolt: 1`, a `Stop` click posting to `/orders/current/stop`, the trace link when `trace_url` is `https://example.test/trace`, and no trace link when `trace_url` is null. Assert the tiles say `No frame yet`.

- [ ] **Step 2: Run the test to verify it fails**

Run from `frontend/`: `npm test -- src/pages/Orders.test.tsx`

Expected: FAIL because the page module does not exist, or the placeholder route has no `Start order` button.

- [ ] **Step 3: Write the minimal implementation**

Replace the `/orders` placeholder in `App.tsx` with `<Orders />`. Use `Button`, `Input`, and `useApi` as `Evaluate.tsx` does. Poll `GET /orders/current` only while status is `running`. Do not add a font, a teal color, or a status that is color without a word. Do not request camera bytes.

- [ ] **Step 4: Run the test to verify it passes**

Run from `frontend/`: `npm test -- src/pages/Orders.test.tsx`

Expected: the file passes.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/Orders.tsx frontend/src/pages/Orders.test.tsx frontend/src/App.tsx
git commit -m "$(cat <<'EOF'
feat: show the order on the Orders page

EOF
)"
```
