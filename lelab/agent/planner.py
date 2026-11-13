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

from lelab.agent.loop import Action
from lelab.utils.forge_config import ForgeConfig

MAX_FRAME_BYTES = 200_000

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "look",
            "description": "Capture the top, side, and wrist cameras.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pick",
            "description": "Pick one item and place it in the tray.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {"type": "string", "enum": ["bolt", "bracket", "gear"]},
                },
                "required": ["item"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_tray",
            "description": "Count the items currently in the tray.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_operator",
            "description": "Ask the operator for help.",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
                "additionalProperties": False,
            },
        },
    },
]


def action_from_message(message) -> Action:
    calls = message.tool_calls
    if not calls:
        return Action("done")
    try:
        arguments = json.loads(calls[0].function.arguments)
    except json.JSONDecodeError:
        return Action("error")
    if not isinstance(arguments, dict):
        return Action("error")
    return Action("tool", calls[0].function.name, arguments)


def _encode_tool_content(content: dict) -> dict:
    encoded = {}
    for key, value in content.items():
        if not isinstance(value, bytes):
            encoded[key] = value
            continue
        if len(value) > MAX_FRAME_BYTES:
            continue
        encoded[key] = base64.b64encode(value).decode("ascii")
    return encoded


def _message_for_planner(message: dict) -> dict:
    copied = dict(message)
    content = copied.get("content")
    if copied.get("role") == "tool" and isinstance(content, dict):
        copied["content"] = _encode_tool_content(content)
    return copied


class InferencePlanner:
    def __init__(self, client, cfg: ForgeConfig) -> None:
        self.client = client
        self.cfg = cfg

    def next_action(self, transcript: list[dict]) -> Action:
        messages = [_message_for_planner(message) for message in transcript]
        response = self.client.chat.completions.create(
            model=self.cfg.planner_model,
            messages=messages,
            tools=TOOLS,
        )
        return action_from_message(response.choices[0].message)
