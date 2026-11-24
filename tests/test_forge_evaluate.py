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
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lelab import evaluate
from lelab.evaluate import EvaluationRequest, EvaluationSession
from lelab.forge.judge import JudgeVerdict
from lelab.forge.registry import RegistryStore
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


CHECKPOINT_CONFIG = {
    "input_features": {
        "observation.images.top": {"type": "VISUAL", "shape": [3, 480, 640]},
        "observation.state": {"type": "STATE", "shape": [6]},
    },
    "output_features": {"action": {"type": "ACTION", "shape": [6]}},
}
ARTIFACT_PATH = "team/wandb-registry-model/so101-chess-smolvla:v3"
ENDPOINT = "tls/sandbox.example:7447"
SETUP_FAILURE = (
    "The CoreWeave policy host could not be set up ({kind}). "
    "Close the evaluation and reopen it with the local policy host."
)


class _FakeHost:
    """Stands in for PolicyHost; each phase blocks on an event the test controls."""

    def __init__(self) -> None:
        self.start_args: tuple | None = None
        self.ready_timeout: float | None = None
        self.stops = 0
        self.start_error: Exception | None = None
        self.ready_error: Exception | None = None
        self.stop_error: Exception | None = None
        self.start_entered = threading.Event()
        self.start_release = threading.Event()
        self.ready_entered = threading.Event()
        self.ready_release = threading.Event()

    def start(self, artifact_path, model_subdir, features, action) -> str:
        self.start_args = (artifact_path, model_subdir, features, action)
        self.start_entered.set()
        assert self.start_release.wait(5)
        if self.start_error is not None:
            raise self.start_error
        return ENDPOINT

    def wait_until_ready(self, timeout_s: float) -> None:
        self.ready_timeout = timeout_s
        self.ready_entered.set()
        assert self.ready_release.wait(5)
        if self.ready_error is not None:
            raise self.ready_error

    def stop(self) -> None:
        self.stops += 1
        if self.stop_error is not None:
            raise self.stop_error


@pytest.fixture
def fake_host(monkeypatch, fake_logger):
    """A CoreWeave-capable config and a controllable policy host; no Sandbox is ever created."""
    monkeypatch.setenv("CWSANDBOX_API_KEY", "sandbox-test-key")
    monkeypatch.setenv("CWSANDBOX_TRAIN_IMAGE", "registry.example/train:latest")
    host = _FakeHost()
    monkeypatch.setattr(evaluate, "make_policy_host", lambda cfg: host)
    monkeypatch.setattr(evaluate, "artifact_path_for", lambda family, skill, alias: ARTIFACT_PATH)
    return host


COREWEAVE_BODY = {
    "policy_family": "smolvla",
    "version": "candidate",
    "policy_host": "coreweave",
    "trials_per_skill": 1,
}


def _open_coreweave(host: _FakeHost) -> TestClient:
    """Open a CoreWeave session and leave it in `starting` (start blocked)."""
    client = TestClient(app)
    assert client.post("/evaluations", json=COREWEAVE_BODY).status_code == 200
    assert host.start_entered.wait(5)
    return client


def _join_host_thread() -> None:
    thread = evaluate.current_session()._host_thread
    assert thread is not None
    thread.join(5)
    assert not thread.is_alive()


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
def registry_downloads(monkeypatch, tmp_path):
    """Replace the Registry download so no test reaches W&B; records each call."""
    calls: list[dict] = []

    def fake_download(self, entity, collection, alias, root=None):
        calls.append({"entity": entity, "collection": collection, "alias": alias, "root": root})
        pretrained_model = tmp_path / "registry" / collection / alias / "pretrained_model"
        pretrained_model.mkdir(parents=True, exist_ok=True)
        (pretrained_model / "config.json").write_text(json.dumps(CHECKPOINT_CONFIG))
        return pretrained_model

    monkeypatch.setattr(RegistryStore, "download", fake_download)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return calls


@pytest.fixture
def fake_logger(monkeypatch, forge_on, registry_downloads):
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


def test_a_finished_trial_is_logged_when_the_next_one_starts() -> None:
    logger = _Logger()
    session = _session(logger)
    session.start_trial("bolt")
    session.finish_trial(JudgeVerdict("delivered", 0.9, "scored"), duration_s=1.0)
    assert logger.predictions == []
    session.start_trial("bolt")
    assert [p[1]["stage"] for p in logger.predictions] == ["delivered"]
    session.finish_trial(JudgeVerdict("reached", 0.8, "scored"), duration_s=2.0)
    session.override("grasped")
    assert len(logger.predictions) == 1
    session.close()
    assert [p[1]["stage"] for p in logger.predictions] == ["delivered", "grasped"]
    assert logger.summary == {"trials": 2}


def test_blank_act_skill_is_400(fake_logger) -> None:
    for skill in (" ", ""):
        response = TestClient(app).post("/evaluations", json={**BODY, "policy_family": "act", "skill": skill})
        assert response.status_code == 400


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


def _trial_body(**overrides) -> dict:
    body = {
        "skill": "bolt",
        "top_jpeg_base64": base64.b64encode(b"top-frame").decode(),
        "side_jpeg_base64": base64.b64encode(b"side-frame").decode(),
        "duration_s": 1.5,
    }
    return {**body, **overrides}


def test_trial_without_a_session_is_400(forge_on) -> None:
    response = TestClient(app).post("/evaluations/current/trial", json=_trial_body())
    assert response.status_code == 400


def test_trial_returns_the_judged_stage(fake_logger, monkeypatch) -> None:
    seen: list[tuple[bytes, bytes]] = []

    def fake_judge(client, cfg, top, side):
        seen.append((top, side))
        return JudgeVerdict("grasped", 0.9, "scored")

    monkeypatch.setattr(evaluate, "judge_frames", fake_judge)
    client = TestClient(app)
    assert client.post("/evaluations", json=BODY).status_code == 200

    response = client.post("/evaluations/current/trial", json=_trial_body())

    assert response.status_code == 200
    assert response.json() == {"skill": "bolt", "stage": "grasped", "verdict_source": "judge"}
    assert seen == [(b"top-frame", b"side-frame")]
    assert evaluate.current_session()._running is False


def test_trial_with_invalid_base64_finishes_as_needs_review(fake_logger, monkeypatch) -> None:
    def fail_judge(*args, **kwargs):
        raise AssertionError("judge_frames must not run for undecodable frames")

    monkeypatch.setattr(evaluate, "judge_frames", fail_judge)
    client = TestClient(app)
    assert client.post("/evaluations", json=BODY).status_code == 200

    response = client.post("/evaluations/current/trial", json=_trial_body(top_jpeg_base64="@@"))

    assert response.status_code == 200
    assert response.json() == {"skill": "bolt", "stage": None, "verdict_source": "judge"}
    assert evaluate.current_session()._running is False


def test_trial_with_an_empty_frame_finishes_as_needs_review(fake_logger, monkeypatch) -> None:
    def fail_judge(*args, **kwargs):
        raise AssertionError("judge_frames must not run for empty frames")

    monkeypatch.setattr(evaluate, "judge_frames", fail_judge)
    client = TestClient(app)
    assert client.post("/evaluations", json=BODY).status_code == 200

    response = client.post("/evaluations/current/trial", json=_trial_body(side_jpeg_base64=""))

    assert response.status_code == 200
    assert response.json()["stage"] is None


def test_evaluation_resolves_candidate_through_the_registry(
    fake_logger, registry_downloads, tmp_path
) -> None:
    response = TestClient(app).post("/evaluations", json=BODY)

    assert response.status_code == 200
    root = tmp_path / ".cache" / "huggingface" / "lerobot" / "forge-registry"
    assert registry_downloads == [
        {
            "entity": "team",
            "collection": "so101-chess-smolvla",
            "alias": "candidate",
            "root": root / "so101-chess-smolvla" / "candidate",
        }
    ]
    expected = tmp_path / "registry" / "so101-chess-smolvla" / "candidate" / "pretrained_model"
    assert evaluate.current_session().policy_dir == expected
    assert response.json()["policy_path"] == str(expected)


def test_evaluation_resolves_production_for_an_act_skill(fake_logger, registry_downloads) -> None:
    body = {**BODY, "policy_family": "act", "skill": "gear", "version": "production"}

    assert TestClient(app).post("/evaluations", json=body).status_code == 200

    assert [(c["collection"], c["alias"]) for c in registry_downloads] == [("so101-chess-act", "production")]
    assert evaluate.current_session().policy_dir is not None


def test_missing_registry_alias_is_400_and_opens_no_session(fake_logger, monkeypatch) -> None:
    def missing(self, entity, collection, alias, root=None):
        raise ValueError(f"Registry artifact is missing: {entity}/wandb-registry-model/{collection}:{alias}")

    monkeypatch.setattr(RegistryStore, "download", missing)
    client = TestClient(app)

    response = client.post("/evaluations", json={**BODY, "version": "production"})

    assert response.status_code == 400
    assert "so101-chess-smolvla:production" in response.json()["detail"]
    assert "local-test-key" not in response.text
    with pytest.raises(ValueError, match="No evaluation session"):
        evaluate.current_session()


def test_checkpoint_dir_for_ignores_the_skill_for_smolvla(forge_on, registry_downloads) -> None:
    from lelab.forge.registry import checkpoint_dir_for

    checkpoint_dir_for("smolvla", "gear", "production")

    assert registry_downloads[0]["collection"] == "so101-chess-smolvla"


def test_current_evaluation_is_closed_without_a_session() -> None:
    response = TestClient(app).get("/evaluations/current")

    assert response.status_code == 200
    assert response.json() == {"open": False}


def test_current_evaluation_reports_the_open_session(fake_logger) -> None:
    client = TestClient(app)
    assert client.post("/evaluations", json=BODY).status_code == 200
    client.post("/evaluations/current/trial", json=_trial_body(top_jpeg_base64="", side_jpeg_base64=""))

    response = client.get("/evaluations/current")

    assert response.json() == {
        "open": True,
        "policy_family": "smolvla",
        "version": "candidate",
        "policy_host": "local",
        "policy_runs_on": "this machine",
        "policy_host_state": "off",
        "policy_host_error": None,
        "trials": 6,
        "finished": 1,
        "running": False,
    }


def test_current_evaluation_closes_with_the_session(fake_logger) -> None:
    client = TestClient(app)
    client.post("/evaluations", json=BODY)
    client.post("/evaluations/current/close", json={})

    assert client.get("/evaluations/current").json() == {"open": False}


def test_a_coreweave_session_logs_coreweave_as_the_host(fake_logger, fake_host) -> None:
    fake_host.start_release.set()
    fake_host.ready_release.set()
    client = TestClient(app)
    assert client.post("/evaluations", json=COREWEAVE_BODY).status_code == 200
    _join_host_thread()
    client.post(
        "/evaluations/current/trial",
        json=_trial_body(top_jpeg_base64="", side_jpeg_base64=""),
    )
    client.post("/evaluations/current/close", json={})

    inputs, output, _ = fake_logger.predictions[0]

    assert inputs["policy_host"] == "coreweave"
    assert output["policy_host"] == "coreweave"
    assert output["requested_policy_host"] == "coreweave"


def test_a_local_session_logs_local_as_the_host(fake_logger) -> None:
    client = TestClient(app)
    assert client.post("/evaluations", json=BODY).status_code == 200
    client.post("/evaluations/current/trial", json=_trial_body(top_jpeg_base64="", side_jpeg_base64=""))
    client.post("/evaluations/current/close", json={})

    inputs, output, _ = fake_logger.predictions[0]

    assert inputs["policy_host"] == "local"
    assert output["policy_host"] == "local"


def test_coreweave_without_sandbox_config_is_400_and_opens_nothing(
    fake_logger, registry_downloads, monkeypatch
) -> None:
    monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
    monkeypatch.delenv("CWSANDBOX_TRAIN_IMAGE", raising=False)
    made: list[object] = []
    monkeypatch.setattr(evaluate, "make_policy_host", lambda cfg: made.append(cfg))

    response = TestClient(app).post("/evaluations", json=COREWEAVE_BODY)

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "The CoreWeave policy host needs CWSANDBOX_API_KEY and CWSANDBOX_TRAIN_IMAGE"
    )
    assert registry_downloads == []
    assert made == []
    with pytest.raises(ValueError, match="No evaluation session"):
        evaluate.current_session()


def test_coreweave_with_an_unusable_checkpoint_config_is_400_and_opens_nothing(
    fake_host, registry_downloads, monkeypatch, tmp_path
) -> None:
    def download(self, entity, collection, alias, root=None):
        pretrained_model = tmp_path / "bad" / "pretrained_model"
        pretrained_model.mkdir(parents=True, exist_ok=True)
        (pretrained_model / "config.json").write_text("{}")
        return pretrained_model

    monkeypatch.setattr(RegistryStore, "download", download)

    response = TestClient(app).post("/evaluations", json=COREWEAVE_BODY)

    assert response.status_code == 400
    assert "input_features" in response.json()["detail"] or "image" in response.json()["detail"]
    assert not fake_host.start_entered.is_set()
    with pytest.raises(ValueError, match="No evaluation session"):
        evaluate.current_session()


def test_the_host_goes_from_starting_to_ready(fake_host) -> None:
    client = _open_coreweave(fake_host)

    starting = client.get("/evaluations/current").json()
    assert starting["policy_runs_on"] == "CoreWeave"
    assert starting["policy_host_state"] == "starting"
    assert starting["policy_host_error"] is None

    fake_host.start_release.set()
    assert fake_host.ready_entered.wait(5)
    assert client.get("/evaluations/current").json()["policy_host_state"] == "starting"

    fake_host.ready_release.set()
    _join_host_thread()

    ready = client.get("/evaluations/current").json()
    assert ready["policy_host_state"] == "ready"
    assert ready["policy_host_error"] is None
    assert ready["policy_runs_on"] == "CoreWeave"
    assert fake_host.ready_timeout == evaluate.READY_TIMEOUT_S == 20 * 60


def test_the_host_starts_on_the_session_checkpoint(fake_host, tmp_path) -> None:
    fake_host.start_release.set()
    fake_host.ready_release.set()
    client = _open_coreweave(fake_host)
    _join_host_thread()

    artifact_path, model_subdir, features, action = fake_host.start_args
    assert artifact_path == ARTIFACT_PATH
    assert model_subdir == "pretrained_model"
    by_name = {feature["name"]: feature for feature in features}
    assert by_name["observation.images.top"]["shape"] == [480, 640, 3]
    assert by_name["observation.state"]["shape"] == [6]
    assert action["name"] == "action"
    assert client.get("/evaluations/current").json()["policy_host_state"] == "ready"


def test_a_checkpoint_without_a_pretrained_model_dir_uses_an_empty_subdir(
    fake_host, monkeypatch, tmp_path
) -> None:
    def download(self, entity, collection, alias, root=None):
        flat = tmp_path / "flat" / "candidate"
        flat.mkdir(parents=True, exist_ok=True)
        (flat / "config.json").write_text(json.dumps(CHECKPOINT_CONFIG))
        return flat

    monkeypatch.setattr(RegistryStore, "download", download)
    fake_host.start_release.set()
    fake_host.ready_release.set()
    _open_coreweave(fake_host)
    _join_host_thread()

    assert fake_host.start_args[1] == ""


def test_the_status_stays_answerable_while_the_host_is_starting(fake_host) -> None:
    client = _open_coreweave(fake_host)
    # start() is blocked inside the host; neither status route may wait for it.
    assert client.get("/evaluations/current").json()["policy_host_state"] == "starting"
    assert evaluate.handle_evaluation_status()["policy_host_state"] == "starting"
    fake_host.start_release.set()
    fake_host.ready_release.set()
    _join_host_thread()


def test_a_start_failure_leaves_the_redacted_message(fake_host) -> None:
    message = "Could not start the CoreWeave policy server: the download failed [redacted]."
    fake_host.start_error = RuntimeError(message)
    fake_host.start_release.set()
    client = _open_coreweave(fake_host)
    _join_host_thread()

    body = client.get("/evaluations/current").json()

    assert body["policy_host_state"] == "failed"
    assert body["policy_host_error"] == message
    assert not fake_host.ready_entered.is_set()


def test_a_readiness_failure_leaves_the_redacted_message(fake_host) -> None:
    message = "Could not start the CoreWeave policy server: the policy server stopped before it was ready."
    fake_host.ready_error = RuntimeError(message)
    fake_host.start_release.set()
    fake_host.ready_release.set()
    client = _open_coreweave(fake_host)
    _join_host_thread()

    body = client.get("/evaluations/current").json()

    assert body["policy_host_state"] == "failed"
    assert body["policy_host_error"] == message


def test_an_unexpected_error_is_never_echoed_into_the_summary(fake_host) -> None:
    fake_host.start_error = ValueError("token=sandbox-test-key")
    fake_host.start_release.set()
    client = _open_coreweave(fake_host)
    _join_host_thread()

    body = client.get("/evaluations/current").json()

    assert body["policy_host_state"] == "failed"
    assert body["policy_host_error"] == SETUP_FAILURE.format(kind="ValueError")
    assert "sandbox-test-key" not in json.dumps(body)


def test_a_factory_failure_is_a_fixed_sentence(fake_logger, monkeypatch) -> None:
    monkeypatch.setenv("CWSANDBOX_API_KEY", "sandbox-test-key")
    monkeypatch.setenv("CWSANDBOX_TRAIN_IMAGE", "registry.example/train:latest")

    def broken(cfg):
        raise OSError("no cwsandbox credentials: sandbox-test-key")

    monkeypatch.setattr(evaluate, "make_policy_host", broken)
    monkeypatch.setattr(evaluate, "artifact_path_for", lambda family, skill, alias: ARTIFACT_PATH)
    client = TestClient(app)
    assert client.post("/evaluations", json=COREWEAVE_BODY).status_code == 200
    _join_host_thread()

    body = client.get("/evaluations/current").json()

    assert body["policy_host_state"] == "failed"
    assert body["policy_host_error"] == SETUP_FAILURE.format(kind="OSError")
    assert "sandbox-test-key" not in json.dumps(body)


def test_closing_a_ready_session_stops_the_host(fake_host) -> None:
    fake_host.start_release.set()
    fake_host.ready_release.set()
    client = _open_coreweave(fake_host)
    _join_host_thread()

    assert client.post("/evaluations/current/close", json={}).status_code == 200

    assert fake_host.stops == 1


def test_closing_while_the_host_is_starting_stops_it_once_start_returns(fake_host) -> None:
    client = _open_coreweave(fake_host)
    session = evaluate.current_session()

    assert client.post("/evaluations/current/close", json={}).status_code == 200
    assert fake_host.stops == 0  # nothing to stop until start() hands the host back

    fake_host.start_release.set()
    assert session._host_thread is not None
    session._host_thread.join(5)

    assert fake_host.stops == 1
    assert not fake_host.ready_entered.is_set()
    assert session.summary()["policy_host_state"] == "off"


def test_closing_while_waiting_for_readiness_stops_the_host_and_never_marks_it_ready(fake_host) -> None:
    fake_host.start_release.set()
    client = _open_coreweave(fake_host)
    session = evaluate.current_session()
    assert fake_host.ready_entered.wait(5)

    assert client.post("/evaluations/current/close", json={}).status_code == 200
    assert fake_host.stops == 1

    fake_host.ready_release.set()
    assert session._host_thread is not None
    session._host_thread.join(5)

    assert session.summary()["policy_host_state"] == "off"
    assert fake_host.stops == 1


def test_stop_policy_host_is_quiet_when_the_host_fails_to_stop(fake_host) -> None:
    fake_host.start_release.set()
    fake_host.ready_release.set()
    _open_coreweave(fake_host)
    _join_host_thread()
    fake_host.stop_error = RuntimeError("sandbox-test-key leaked")

    evaluate.stop_policy_host()
    evaluate.stop_policy_host()

    assert fake_host.stops == 1
    assert evaluate.current_session().summary()["policy_host_state"] == "off"


def test_stop_policy_host_without_a_session_does_nothing() -> None:
    evaluate.stop_policy_host()


def test_a_local_session_has_no_host_to_stop(fake_logger) -> None:
    client = TestClient(app)
    assert client.post("/evaluations", json=BODY).status_code == 200
    evaluate.stop_policy_host()

    assert client.get("/evaluations/current").json()["policy_host_state"] == "off"


def test_rollout_target_without_a_session_is_none() -> None:
    assert evaluate.rollout_target() is None


def test_rollout_target_for_a_local_session_is_the_checkpoint(fake_logger) -> None:
    assert TestClient(app).post("/evaluations", json=BODY).status_code == 200
    policy_dir = evaluate.current_session().policy_dir

    assert evaluate.rollout_target() == evaluate.RolloutTarget("local", str(policy_dir))


def test_rollout_target_refuses_while_the_host_is_starting(fake_host) -> None:
    _open_coreweave(fake_host)

    with pytest.raises(evaluate.PolicyHostNotReady) as info:
        evaluate.rollout_target()

    assert str(info.value) == "The CoreWeave policy host is still starting. Try again in a minute."
    fake_host.start_release.set()
    fake_host.ready_release.set()
    _join_host_thread()


def test_rollout_target_refuses_with_the_stored_error_when_the_host_failed(fake_host) -> None:
    message = "Could not start the CoreWeave policy server: boom. Close the evaluation and reopen it."
    fake_host.start_error = RuntimeError(message)
    fake_host.start_release.set()
    _open_coreweave(fake_host)
    _join_host_thread()

    with pytest.raises(evaluate.PolicyHostNotReady) as info:
        evaluate.rollout_target()

    assert str(info.value) == message


def test_rollout_target_for_a_ready_host_is_its_endpoint(fake_host) -> None:
    fake_host.start_release.set()
    fake_host.ready_release.set()
    _open_coreweave(fake_host)
    _join_host_thread()

    assert evaluate.rollout_target() == evaluate.RolloutTarget("remote", ENDPOINT)


def test_rollout_target_refuses_after_the_host_was_stopped(fake_host) -> None:
    fake_host.start_release.set()
    fake_host.ready_release.set()
    _open_coreweave(fake_host)
    _join_host_thread()
    evaluate.stop_policy_host()

    with pytest.raises(evaluate.PolicyHostNotReady, match="not running"):
        evaluate.rollout_target()


def test_active_policy_path_is_still_the_session_checkpoint(fake_logger) -> None:
    assert TestClient(app).post("/evaluations", json=BODY).status_code == 200

    assert evaluate.active_policy_path() == str(evaluate.current_session().policy_dir)
