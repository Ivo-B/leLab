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

from lelab.forge.weave_eval import TrialRow, WeaveTrialLog


class _Prediction:
    def __init__(self) -> None:
        self.scores: dict[str, object] = {}
        self.finished = False

    def log_score(self, name: str, value: object) -> None:
        self.scores[name] = value

    def finish(self) -> None:
        self.finished = True


class _Logger:
    def __init__(self) -> None:
        self.predictions: list[tuple[dict, dict, _Prediction]] = []
        self.summary: dict | None = None

    def log_prediction(self, inputs: dict, output: dict) -> _Prediction:
        prediction = _Prediction()
        self.predictions.append((inputs, output, prediction))
        return prediction

    def log_summary(self, summary: dict) -> None:
        self.summary = summary


def test_one_row_records_the_spec_fields_and_the_forge_url() -> None:
    logger = _Logger()
    log = WeaveTrialLog(logger, entity="team", project="so101-kitting")
    log.log_trial(
        TrialRow(
            skill="bracket",
            stage="grasped",
            verdict_source="judge",
            duration_s=12.5,
            chunk_latency_s=None,
            policy_host="local",
            top_video="/tmp/top.mp4",
            side_video="/tmp/side.mp4",
        )
    )
    inputs, output, prediction = logger.predictions[0]
    assert inputs["skill"] == "bracket"
    assert output["stage"] == "grasped"
    assert output["verdict_source"] == "judge"
    assert output["policy_host"] == "local"
    assert prediction.scores["needs_review"] == 0
    assert prediction.finished is True
    assert log.finish() == "https://wandb.ai/team/so101-kitting/weave"
    assert logger.summary == {"trials": 1}


def test_needs_review_score_is_one_when_the_stage_is_missing() -> None:
    logger = _Logger()
    log = WeaveTrialLog(logger, entity="team", project="so101-kitting")
    log.log_trial(
        TrialRow("bolt", None, "judge", 4.0, None, "coreweave", None, None)
    )
    assert logger.predictions[0][2].scores["needs_review"] == 1
