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

import pytest

from lelab.forge.judge import (
    CONFIDENCE_FLOOR,
    MAX_FRAME_BYTES,
    STAGES,
    frame_data_url,
    parse_judge_payload,
)


def test_stage_ladder_and_floor() -> None:
    assert STAGES == ("started", "reached", "grasped", "delivered")
    assert CONFIDENCE_FLOOR == 0.6
    assert MAX_FRAME_BYTES == 200_000


def test_delivered_above_the_floor_is_scored() -> None:
    verdict = parse_judge_payload({"stage": "delivered", "confidence": 0.91})
    assert verdict.status == "scored"
    assert verdict.stage == "delivered"
    assert verdict.confidence == 0.91


def test_low_confidence_needs_review() -> None:
    verdict = parse_judge_payload({"stage": "grasped", "confidence": 0.59})
    assert verdict.status == "needs_review"
    assert verdict.stage == "grasped"


def test_unknown_stage_or_missing_confidence_needs_review() -> None:
    assert parse_judge_payload({"stage": "success", "confidence": 1}).status == "needs_review"
    assert parse_judge_payload({"stage": "delivered"}).status == "needs_review"
    assert parse_judge_payload({"stage": "delivered", "confidence": 1.2}).stage is None


def test_frame_data_url_rejects_an_oversized_jpeg() -> None:
    url = frame_data_url("top", b"\xff\xd8\xff")
    assert url.startswith("data:image/jpeg;base64,")
    with pytest.raises(ValueError, match="top"):
        frame_data_url("top", b"x" * (MAX_FRAME_BYTES + 1))
