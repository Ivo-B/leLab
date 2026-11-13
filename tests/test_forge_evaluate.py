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
from fastapi.testclient import TestClient

from lelab import evaluate
from lelab.evaluate import EvaluationRequest, EvaluationSession
from lelab.forge.judge import JudgeVerdict
from lelab.forge.weave_eval import WeaveTrialLog
from lelab.server import app


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


def _session(logger: _Logger, family: str = "smolvla", skill: str | None = None, trials: int = 2):
    request = EvaluationRequest(
        policy_family=family,
        version="candidate",
        policy_host="local",
        trials_per_skill=trials,
        skill=skill,
    )
    return EvaluationSession(request, WeaveTrialLog(logger, entity="team", project="so101-kitting"))


@pytest.fixture(autouse=True)
def _reset_session():
    evaluate.reset_session()
    yield
    evaluate.reset_session()


@pytest.fixture
def forge_on(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "local-test-key")
    monkeypatch.setenv("WANDB_ENTITY", "team")
    monkeypatch.delenv("WANDB_PROJECT", raising=False)


@pytest.fixture
def fake_logger(monkeypatch, forge_on):
    logger = _Logger()

    def factory(request, cfg):
        return _session_from(request, logger)

    def _session_from(request, logger):
        return EvaluationSession(request, WeaveTrialLog(logger, entity="team", project="so101-kitting"))

    monkeypatch.setattr(evaluate, "create_session", factory)
    return logger


BODY = {
    "policy_family": "smolvla",
    "version": "candidate",
    "policy_host": "local",
    "trials_per_skill": 2,
}


def test_smolvla_session_covers_three_skills_repeated() -> None:
    session = _session(_Logger(), trials=2)
    assert session.skills == ["bolt", "bolt", "bracket", "bracket", "gear", "gear"]


def test_act_session_contains_only_the_requested_skill() -> None:
    session = _session(_Logger(), family="act", skill="gear", trials=3)
    assert session.skills == ["gear", "gear", "gear"]


def test_act_without_a_skill_is_rejected() -> None:
    with pytest.raises(ValueError, match="skill"):
        _session(_Logger(), family="act", skill=None)


def test_start_trial_does_not_need_the_arm() -> None:
    session = _session(_Logger())
    assert session.start_trial("bolt") == {"skill": "bolt", "index": 1, "status": "running"}


def test_scored_verdict_logs_a_row_with_needs_review_zero() -> None:
    logger = _Logger()
    session = _session(logger)
    session.start_trial("bolt")
    result = session.finish_trial(JudgeVerdict("delivered", 0.9, "scored"), duration_s=12.5)
    assert result["stage"] == "delivered"
    assert result["verdict_source"] == "judge"
    assert logger.predictions == []
    session.close()
    assert len(logger.predictions) == 1
    inputs, output, prediction = logger.predictions[0]
    assert inputs == {"skill": "bolt", "policy_host": "local"}
    assert output["stage"] == "delivered"
    assert output["verdict_source"] == "judge"
    assert output["duration_s"] == 12.5
    assert output["chunk_latency_s"] is None
    assert output["top_video"] is None
    assert output["side_video"] is None
    assert prediction.scores["needs_review"] == 0


def test_needs_review_verdict_logs_needs_review_one() -> None:
    logger = _Logger()
    session = _session(logger)
    session.start_trial("bolt")
    session.finish_trial(JudgeVerdict(None, None, "needs_review"), duration_s=3.0)
    assert logger.predictions == []
    session.close()
    assert len(logger.predictions) == 1
    _, output, prediction = logger.predictions[0]
    assert output["stage"] is None
    assert output["verdict_source"] == "judge"
    assert prediction.scores["needs_review"] == 1


def test_override_replaces_the_stage_and_source() -> None:
    logger = _Logger()
    session = _session(logger)
    session.start_trial("bolt")
    session.finish_trial(JudgeVerdict("reached", 0.8, "scored"), duration_s=5.0)
    result = session.override("grasped")
    assert result["stage"] == "grasped"
    assert result["verdict_source"] == "operator"
    assert logger.predictions == []
    session.close()
    assert len(logger.predictions) == 1
    _, output, prediction = logger.predictions[0]
    assert output["stage"] == "grasped"
    assert output["verdict_source"] == "operator"
    assert prediction.scores["needs_review"] == 0


def test_override_to_null_leaves_the_stage_empty_for_review() -> None:
    logger = _Logger()
    session = _session(logger)
    session.start_trial("bolt")
    session.finish_trial(JudgeVerdict("reached", 0.8, "scored"), duration_s=5.0)
    result = session.override(None)
    assert result["stage"] is None
    assert result["verdict_source"] == "operator"
    assert logger.predictions == []
    session.close()
    assert len(logger.predictions) == 1
    _, output, prediction = logger.predictions[0]
    assert output["stage"] is None
    assert output["verdict_source"] == "operator"
    assert prediction.scores["needs_review"] == 1


def test_override_with_an_unknown_stage_is_rejected() -> None:
    session = _session(_Logger())
    session.start_trial("bolt")
    session.finish_trial(JudgeVerdict("reached", 0.8, "scored"), duration_s=5.0)
    with pytest.raises(ValueError, match="stage"):
        session.override("flying")


def test_override_before_any_trial_is_rejected() -> None:
    with pytest.raises(ValueError, match="trial"):
        _session(_Logger()).override("grasped")


def test_start_trial_rejects_a_skill_outside_the_session() -> None:
    session = _session(_Logger(), family="act", skill="gear")
    with pytest.raises(ValueError, match="bolt"):
        session.start_trial("bolt")


def test_close_returns_the_weave_url() -> None:
    logger = _Logger()
    session = _session(logger)
    session.start_trial("bolt")
    assert session.close() == {"weave_url": "https://wandb.ai/team/so101-kitting/weave"}
    assert logger.predictions == []
    assert logger.summary == {"trials": 0}


def test_close_logs_one_prediction_per_finished_trial() -> None:
    logger = _Logger()
    session = _session(logger)
    session.start_trial("bolt")
    session.finish_trial(JudgeVerdict("delivered", 0.9, "scored"), duration_s=12.5)
    session.start_trial("bracket")
    session.finish_trial(JudgeVerdict("reached", 0.8, "scored"), duration_s=4.0)
    session.override("grasped")
    session.start_trial("gear")
    session.close()
    assert len(logger.predictions) == 2
    first_output, first_prediction = logger.predictions[0][1], logger.predictions[0][2]
    assert first_output["stage"] == "delivered"
    assert first_output["verdict_source"] == "judge"
    assert first_output["duration_s"] == 12.5
    assert first_prediction.scores["needs_review"] == 0
    second_output, second_prediction = logger.predictions[1][1], logger.predictions[1][2]
    assert second_output["stage"] == "grasped"
    assert second_output["verdict_source"] == "operator"
    assert second_output["duration_s"] == 4.0
    assert second_prediction.scores["needs_review"] == 0
    assert logger.summary == {"trials": 2}


def test_forge_off_is_400(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    response = TestClient(app).post("/evaluations", json=BODY)
    assert response.status_code == 400
    assert response.json()["detail"] == "Evaluation needs Weights & Biases"


@pytest.mark.parametrize("trials", [0, 51])
def test_trials_per_skill_outside_range_is_400(fake_logger, trials) -> None:
    response = TestClient(app).post("/evaluations", json={**BODY, "trials_per_skill": trials})
    assert response.status_code == 400


def test_act_without_a_skill_is_400(fake_logger) -> None:
    response = TestClient(app).post("/evaluations", json={**BODY, "policy_family": "act"})
    assert response.status_code == 400


def test_second_session_is_409(fake_logger) -> None:
    client = TestClient(app)
    assert client.post("/evaluations", json=BODY).status_code == 200
    assert client.post("/evaluations", json=BODY).status_code == 409


def test_route_override_and_close(fake_logger) -> None:
    client = TestClient(app)
    assert client.post("/evaluations", json=BODY).status_code == 200
    session = evaluate.current_session()
    session.start_trial("bolt")
    session.finish_trial(JudgeVerdict("reached", 0.8, "scored"), duration_s=5.0)

    bad = client.post("/evaluations/current/override", json={"stage": "flying"})
    assert bad.status_code == 400

    good = client.post("/evaluations/current/override", json={"stage": "grasped"})
    assert good.status_code == 200
    assert good.json()["stage"] == "grasped"
    assert good.json()["verdict_source"] == "operator"

    empty = client.post("/evaluations/current/override", json={"stage": None})
    assert empty.status_code == 200
    assert empty.json()["stage"] is None
    assert fake_logger.predictions == []

    closed = client.post("/evaluations/current/close")
    assert closed.status_code == 200
    assert closed.json() == {"weave_url": "https://wandb.ai/team/so101-kitting/weave"}
    assert len(fake_logger.predictions) == 1
    _, output, prediction = fake_logger.predictions[0]
    assert output["stage"] is None
    assert output["verdict_source"] == "operator"
    assert prediction.scores["needs_review"] == 1
    assert fake_logger.summary is not None
    assert client.post("/evaluations", json=BODY).status_code == 200


def test_override_without_a_session_is_400(forge_on) -> None:
    response = TestClient(app).post("/evaluations/current/override", json={"stage": "grasped"})
    assert response.status_code == 400


def test_second_close_does_not_log_the_trials_again(fake_logger) -> None:
    summaries: list[dict] = []
    original = fake_logger.log_summary
    fake_logger.log_summary = lambda summary: (summaries.append(summary), original(summary))[1]
    client = TestClient(app)
    assert client.post("/evaluations", json=BODY).status_code == 200
    session = evaluate.current_session()
    session.start_trial("bolt")
    session.finish_trial(JudgeVerdict("reached", 0.8, "scored"), duration_s=5.0)

    first = client.post("/evaluations/current/close")
    assert first.status_code == 200
    assert first.json() == {"weave_url": "https://wandb.ai/team/so101-kitting/weave"}

    second = client.post("/evaluations/current/close")
    assert second.status_code == 400
    assert len(fake_logger.predictions) == 1
    assert summaries == [{"trials": 1}]


def test_override_without_a_stage_key_is_rejected_and_keeps_the_stage(fake_logger) -> None:
    client = TestClient(app)
    assert client.post("/evaluations", json=BODY).status_code == 200
    session = evaluate.current_session()
    session.start_trial("bolt")
    session.finish_trial(JudgeVerdict("reached", 0.8, "scored"), duration_s=5.0)

    response = client.post("/evaluations/current/override", json={})
    assert 400 <= response.status_code < 500
    assert session._last is not None
    assert session._last["stage"] == "reached"
    assert session._last["verdict_source"] == "judge"
