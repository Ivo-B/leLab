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


def test_a_real_trial_is_a_chess_row_with_domain_real() -> None:
    logger = _Logger()
    log = WeaveTrialLog(logger, entity="team", project="so101-chess")
    log.log_trial(TrialRow(("e2", "e3"), "near", "success", "operator", 3.5, "local", "local"))
    inputs, output, prediction = logger.predictions[0]
    assert inputs == {"transfer": "e2-e3", "class": "near", "domain": "real", "policy_host": "local"}
    assert output == {
        "outcome": "success",
        "verdict_source": "operator",
        "duration_s": 3.5,
        "policy_host": "local",
        "requested_policy_host": "local",
    }
    assert prediction.scores == {"success": 1}
    assert prediction.finished is True
    assert log.finish() == "https://wandb.ai/team/so101-chess/weave"
    assert logger.summary == {"trials": 1}


def test_a_failed_trial_scores_zero() -> None:
    logger = _Logger()
    log = WeaveTrialLog(logger, entity="team", project="so101-chess")
    log.log_trial(TrialRow(("e4", "bin"), "bin", "drop", "override", 4.0, "coreweave"))
    inputs, output, prediction = logger.predictions[0]
    assert inputs["policy_host"] == "coreweave"
    assert output["outcome"] == "drop"
    assert output["requested_policy_host"] is None
    assert prediction.scores == {"success": 0}
