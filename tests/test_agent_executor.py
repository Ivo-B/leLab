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
