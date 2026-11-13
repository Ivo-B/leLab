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

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from lelab.agent.executor import ToolExecutor
from lelab.agent.loop import OrderLoop, OrderResult, item_counts
from lelab.agent.planner import InferencePlanner
from lelab.agent.tracing import export_trace
from lelab.utils.forge_config import ForgeConfig, forge_configured, load_forge_config

logger = logging.getLogger(__name__)

_ANSWER_POLL_S = 0.1


class OrderBusyError(Exception):
    pass


class OrderRequest(BaseModel):
    text: str
    policy_family: Literal["smolvla", "act"]
    policy_host: Literal["local", "coreweave"]


@dataclass
class _Session:
    text: str
    counts: dict[str, int]
    status: str = "running"
    reason: str | None = None
    steps: list[dict] = field(default_factory=list)
    rollouts: int = 0
    trace_url: str | None = None
    question: str | None = None
    answer: str | None = None
    stop: threading.Event = field(default_factory=threading.Event)
    answered: threading.Event = field(default_factory=threading.Event)


_lock = threading.Lock()
_session: _Session | None = None


def _omit_bytes(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"omitted_bytes": len(value)}
    if isinstance(value, dict):
        return {key: _omit_bytes(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_omit_bytes(item) for item in value]
    return value


def _ask(session: _Session, message: str) -> dict[str, Any]:
    session.answer = None
    session.answered.clear()
    session.question = message
    while not session.answered.wait(timeout=_ANSWER_POLL_S):
        if session.stop.is_set():
            session.question = None
            return {"stopped": True}
    answer = session.answer
    session.question = None
    session.answer = None
    return {"message": message, "answer": answer}


def _build_loop(session: _Session, cfg: ForgeConfig) -> OrderLoop:
    import openai

    client = openai.OpenAI(base_url="https://api.inference.wandb.ai/v1", api_key=cfg.api_key)
    executor = ToolExecutor(ask=lambda message: _ask(session, message))
    return OrderLoop(InferencePlanner(client, cfg), executor, stop=session.stop)


def _work(session: _Session, cfg: ForgeConfig, request: OrderRequest) -> None:
    try:
        result = _build_loop(session, cfg).run(request.text, request.policy_family, request.policy_host)
    except Exception:
        logger.exception("Order run failed")
        result = OrderResult("aborted", "error", session.counts, [], 0)
    try:
        session.trace_url = export_trace(result, cfg, request.text)
    except Exception:
        logger.exception("Order trace export failed")
    session.steps = result.steps
    session.rollouts = result.rollouts
    session.reason = result.reason
    session.question = None
    session.status = result.status


def handle_start_order(request: OrderRequest) -> dict[str, Any]:
    global _session
    cfg = load_forge_config()
    if not forge_configured(cfg):
        raise ValueError("Orders need Weights & Biases")
    counts = item_counts(request.text)
    session = _Session(text=request.text, counts=counts)
    with _lock:
        if _session is not None and _session.status == "running":
            raise OrderBusyError("An order is already running")
        _session = session
    threading.Thread(target=_work, args=(session, cfg, request), name="order-worker", daemon=True).start()
    return {"status": "running", "counts": counts}


def handle_order_status() -> dict[str, Any]:
    with _lock:
        session = _session
    if session is None:
        return {"status": "idle"}
    return {
        "status": session.status,
        "reason": session.reason,
        "counts": session.counts,
        "steps": _omit_bytes(session.steps),
        "rollouts": session.rollouts,
        "trace_url": session.trace_url,
        "question": session.question,
    }


def handle_stop_order() -> dict[str, Any]:
    with _lock:
        session = _session
    if session is None:
        raise ValueError("No order to stop")
    session.stop.set()
    return {"status": session.status}


def handle_answer(message: str) -> dict[str, Any]:
    with _lock:
        session = _session
    if session is None or session.status != "running" or session.question is None:
        raise ValueError("No operator question is waiting")
    session.answer = message
    session.answered.set()
    return {"status": session.status}


def reset_session() -> None:
    global _session
    with _lock:
        session = _session
        _session = None
    if session is not None:
        session.stop.set()
