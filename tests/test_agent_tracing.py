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
