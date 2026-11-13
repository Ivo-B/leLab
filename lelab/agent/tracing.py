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
from lelab.utils.forge_config import ForgeConfig


def _load_tracing():
    import coreweave.forge.agentlens.tracing as tracing

    return tracing


def _omit_bytes(value: object) -> object:
    if isinstance(value, bytes):
        return {"omitted_bytes": len(value)}
    if isinstance(value, dict):
        return {key: _omit_bytes(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_omit_bytes(item) for item in value]
    return value


def export_trace(result: OrderResult, cfg: ForgeConfig, order_text: str) -> str | None:
    try:
        tracing = _load_tracing()
        tracing.init(
            f"{cfg.entity}/{cfg.project}",
            api_key=cfg.api_key,
            autopatch_integrations=False,
        )
        with tracing.start_conversation(agent_name="so101-kitting", conversation_name="order"):
            with tracing.start_turn(
                user_message=order_text,
                agent_name="so101-kitting",
                model=cfg.planner_model,
            ):
                for step in result.steps:
                    with tracing.start_tool(
                        name=step["tool"],
                        arguments=json.dumps(step["arguments"]),
                    ) as tool:
                        tool.result = json.dumps(_omit_bytes(step["result"]))
    except Exception:
        return None
    return None
