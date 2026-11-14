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

import base64
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
    assert action_from_message(FakeMessage([FakeCall("look", "{")])).kind == "error"


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
    assert base64.b64encode(b"a" * 10).decode("ascii") in encoded
    assert "side" not in captured["messages"][1]["content"]
    assert isinstance(captured["messages"][1]["content"], str)
    assert "bbbb" not in encoded
