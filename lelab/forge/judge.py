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
from dataclasses import dataclass

STAGES = ("started", "reached", "grasped", "delivered")
CONFIDENCE_FLOOR = 0.6
MAX_FRAME_BYTES = 200_000


@dataclass(frozen=True)
class JudgeVerdict:
    stage: str | None
    confidence: float | None
    status: str


def parse_judge_payload(payload: dict) -> JudgeVerdict:
    stage = payload.get("stage")
    raw = payload.get("confidence")
    if stage not in STAGES or isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return JudgeVerdict(None, None, "needs_review")
    confidence = float(raw)
    if confidence < 0 or confidence > 1:
        return JudgeVerdict(None, None, "needs_review")
    status = "scored" if confidence >= CONFIDENCE_FLOOR else "needs_review"
    return JudgeVerdict(stage, confidence, status)


def frame_data_url(name: str, jpeg: bytes) -> str:
    if len(jpeg) > MAX_FRAME_BYTES:
        raise ValueError(f"{name} frame exceeds {MAX_FRAME_BYTES} bytes")
    encoded = base64.b64encode(jpeg).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
