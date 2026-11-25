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
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lelab import evaluate
from lelab.chess.transfers import OUTCOMES, TRANSFER_PLAN, transfer_class
from lelab.evaluate import EvaluationRequest, EvaluationSession, session_plan
from lelab.forge.registry import RegistryStore
from lelab.forge.weave_eval import WeaveTrialLog
from lelab.server import app

WEAVE_URL = "https://wandb.ai/team/so101-chess/weave"


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


class _FakeLog:
    def __init__(self) -> None:
        self.rows: list = []

    def log_trial(self, row) -> None:
        self.rows.append(row)

    def finish(self) -> str:
        return WEAVE_URL


@pytest.fixture
def fake_log() -> _FakeLog:
    return _FakeLog()


def _request(**kw) -> EvaluationRequest:
    return EvaluationRequest(
        policy_family=kw.get("policy_family", "act"),
        version="candidate",
        policy_host="local",
        rounds=kw.get("rounds", 1),
    )


def _session(logger: _Logger, rounds: int = 1) -> EvaluationSession:
    return EvaluationSession(
        _request(rounds=rounds), WeaveTrialLog(logger, entity="team", project="so101-chess")
    )


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
    monkeypatch.setattr(evaluate, "artifact_path_for", lambda family, alias: ARTIFACT_PATH)
    return host


COREWEAVE_BODY = {
    "policy_family": "smolvla",
    "version": "candidate",
    "policy_host": "coreweave",
    "rounds": 1,
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
        return EvaluationSession(request, WeaveTrialLog(logger, entity="team", project="so101-chess"))

    monkeypatch.setattr(evaluate, "create_session", factory)
    return logger


BODY = {
    "policy_family": "smolvla",
    "version": "candidate",
    "policy_host": "local",
    "rounds": 1,
}


def test_the_plan_is_the_sim_transfer_plan_per_round() -> None:
    assert session_plan(_request(rounds=2)) == list(TRANSFER_PLAN) * 2
    with pytest.raises(ValueError, match="rounds"):
        session_plan(_request(rounds=0))
    with pytest.raises(ValueError, match="rounds"):
        session_plan(_request(rounds=6))


def test_trials_follow_the_plan_and_take_operator_outcomes(fake_log) -> None:
    session = EvaluationSession(_request(), fake_log)
    first = session.summary()
    assert first["next_transfer"] == list(TRANSFER_PLAN[0])
    assert first["next_class"] == transfer_class(TRANSFER_PLAN[0])
    result = session.record_trial("grasp", 4.2)
    assert result == {
        "index": 1,
        "transfer": list(TRANSFER_PLAN[0]),
        "class": transfer_class(TRANSFER_PLAN[0]),
        "outcome": "grasp",
        "verdict_source": "operator",
    }
    assert session.summary()["next_transfer"] == list(TRANSFER_PLAN[1])


def test_unknown_outcomes_are_refused(fake_log) -> None:
    session = EvaluationSession(_request(), fake_log)
    with pytest.raises(ValueError, match="outcome"):
        session.record_trial("delivered", 1.0)
    assert "success" in OUTCOMES and "delivered" not in OUTCOMES


def test_override_changes_the_last_outcome_until_the_next_trial(fake_log) -> None:
    session = EvaluationSession(_request(), fake_log)
    session.record_trial("place", 3.0)
    assert session.override("success")["verdict_source"] == "override"
    session.record_trial("success", 2.0)
    assert fake_log.rows[0].outcome == "success"
    assert fake_log.rows[0].verdict_source == "override"


def test_the_plan_ends(fake_log) -> None:
    session = EvaluationSession(_request(), fake_log)
    for _ in TRANSFER_PLAN:
        session.record_trial("success", 1.0)
    assert session.summary()["next_transfer"] is None
    assert session.summary()["next_class"] is None
    with pytest.raises(ValueError, match="Every planned trial"):
        session.record_trial("success", 1.0)


def test_override_with_an_unknown_outcome_is_refused(fake_log) -> None:
    session = EvaluationSession(_request(), fake_log)
    session.record_trial("place", 3.0)
    with pytest.raises(ValueError, match="outcome"):
        session.override("flying")


def test_override_before_any_trial_is_refused(fake_log) -> None:
    with pytest.raises(ValueError, match="trial"):
        EvaluationSession(_request(), fake_log).override("success")


def test_a_logged_trial_can_no_longer_be_overridden(fake_log) -> None:
    session = EvaluationSession(_request(), fake_log)
    session.record_trial("place", 3.0)
    session.close()
    with pytest.raises(ValueError, match="trial"):
        session.override("success")


def test_create_session_logs_to_the_chess_evaluation(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def init(project: str) -> None:
        calls["project"] = project

    def evaluation_logger(**kwargs) -> _Logger:
        calls["logger"] = kwargs
        return _Logger()

    monkeypatch.setitem(sys.modules, "weave", SimpleNamespace(init=init, EvaluationLogger=evaluation_logger))
    cfg = SimpleNamespace(entity="team", project="so101-chess")

    session = evaluate.create_session(_request(policy_family="smolvla"), cfg)

    assert calls == {
        "project": "team/so101-chess",
        "logger": {"name": "so101-chess-sim-eval", "model": "smolvla-candidate"},
    }
    assert session.close() == {"weave_url": WEAVE_URL}


def test_a_trial_row_carries_the_transfer_and_the_host() -> None:
    logger = _Logger()
    session = _session(logger)
    session.record_trial("success", 12.5)
    assert logger.predictions == []
    session.close()
    assert len(logger.predictions) == 1
    inputs, output, prediction = logger.predictions[0]
    assert inputs == {
        "transfer": "-".join(TRANSFER_PLAN[0]),
        "class": transfer_class(TRANSFER_PLAN[0]),
        "domain": "real",
        "policy_host": "local",
        "cameras": "",
    }
    assert output == {
        "outcome": "success",
        "verdict_source": "operator",
        "duration_s": 12.5,
        "policy_host": "local",
        "requested_policy_host": "local",
    }
    assert prediction.scores == {"success": 1}


def test_close_without_trials_logs_nothing() -> None:
    logger = _Logger()
    session = _session(logger)
    assert session.close() == {"weave_url": WEAVE_URL}
    assert logger.predictions == []
    assert logger.summary == {"trials": 0}


def test_a_recorded_trial_is_logged_when_the_next_one_is_recorded() -> None:
    logger = _Logger()
    session = _session(logger)
    session.record_trial("success", 1.0)
    assert logger.predictions == []
    session.record_trial("place", 2.0)
    assert [p[1]["outcome"] for p in logger.predictions] == ["success"]
    session.override("topple")
    assert len(logger.predictions) == 1
    session.close()
    assert [(p[1]["outcome"], p[1]["verdict_source"]) for p in logger.predictions] == [
        ("success", "operator"),
        ("topple", "override"),
    ]
    assert [p[2].scores["success"] for p in logger.predictions] == [1, 0]
    assert logger.summary == {"trials": 2}


def test_forge_off_is_400(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    response = TestClient(app).post("/evaluations", json=BODY)
    assert response.status_code == 400
    assert response.json()["detail"] == "Evaluation needs Weights & Biases"


@pytest.mark.parametrize("rounds", [0, 6])
def test_rounds_outside_range_is_400(fake_logger, registry_downloads, rounds) -> None:
    response = TestClient(app).post("/evaluations", json={**BODY, "rounds": rounds})
    assert response.status_code == 400
    assert "rounds" in response.json()["detail"]
    assert registry_downloads == []


def test_start_reports_the_planned_trials(fake_logger) -> None:
    response = TestClient(app).post("/evaluations", json={**BODY, "rounds": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["trials"] == 2 * len(TRANSFER_PLAN)
    assert {k: body[k] for k in ("policy_family", "version", "policy_host")} == {
        "policy_family": "smolvla",
        "version": "candidate",
        "policy_host": "local",
    }


def test_second_session_is_409(fake_logger) -> None:
    client = TestClient(app)
    assert client.post("/evaluations", json=BODY).status_code == 200
    assert client.post("/evaluations", json=BODY).status_code == 409


def test_trial_without_a_session_is_400(forge_on) -> None:
    response = TestClient(app).post("/evaluations/current/trial", json={"outcome": "success"})
    assert response.status_code == 400


def test_trial_route_records_the_operator_outcome(fake_logger) -> None:
    client = TestClient(app)
    assert client.post("/evaluations", json=BODY).status_code == 200

    response = client.post("/evaluations/current/trial", json={"outcome": "drop"})

    assert response.status_code == 200
    assert response.json() == {
        "index": 1,
        "transfer": list(TRANSFER_PLAN[0]),
        "class": transfer_class(TRANSFER_PLAN[0]),
        "outcome": "drop",
        "verdict_source": "operator",
    }


def test_trial_route_with_an_unknown_outcome_is_400(fake_logger) -> None:
    client = TestClient(app)
    assert client.post("/evaluations", json=BODY).status_code == 200

    response = client.post("/evaluations/current/trial", json={"outcome": "nope"})

    assert response.status_code == 400
    assert evaluate.current_session().summary()["finished"] == 0


@pytest.mark.parametrize("duration", [-1.0, "inf", "nan"])
def test_trial_route_with_an_invalid_duration_is_422(fake_logger, duration) -> None:
    client = TestClient(app)
    assert client.post("/evaluations", json=BODY).status_code == 200

    response = client.post("/evaluations/current/trial", json={"outcome": "success", "duration_s": duration})

    assert response.status_code == 422
    assert evaluate.current_session().summary()["finished"] == 0


def test_route_override_and_close(fake_logger) -> None:
    client = TestClient(app)
    assert client.post("/evaluations", json=BODY).status_code == 200
    assert (
        client.post("/evaluations/current/trial", json={"outcome": "place", "duration_s": 5.0}).status_code
        == 200
    )

    bad = client.post("/evaluations/current/override", json={"outcome": "flying"})
    assert bad.status_code == 400

    good = client.post("/evaluations/current/override", json={"outcome": "success"})
    assert good.status_code == 200
    assert good.json()["outcome"] == "success"
    assert good.json()["verdict_source"] == "override"
    assert fake_logger.predictions == []

    closed = client.post("/evaluations/current/close")
    assert closed.status_code == 200
    assert closed.json() == {"weave_url": WEAVE_URL}
    assert len(fake_logger.predictions) == 1
    _, output, prediction = fake_logger.predictions[0]
    assert output["outcome"] == "success"
    assert output["verdict_source"] == "override"
    assert output["duration_s"] == 5.0
    assert prediction.scores["success"] == 1
    assert fake_logger.summary is not None
    assert client.post("/evaluations", json=BODY).status_code == 200


def test_override_without_a_session_is_400(forge_on) -> None:
    response = TestClient(app).post("/evaluations/current/override", json={"outcome": "success"})
    assert response.status_code == 400


def test_override_without_an_outcome_key_is_rejected_and_keeps_the_outcome(fake_logger) -> None:
    client = TestClient(app)
    assert client.post("/evaluations", json=BODY).status_code == 200
    client.post("/evaluations/current/trial", json={"outcome": "place"})

    response = client.post("/evaluations/current/override", json={})

    assert 400 <= response.status_code < 500
    client.post("/evaluations/current/close")
    _, output, _ = fake_logger.predictions[0]
    assert (output["outcome"], output["verdict_source"]) == ("place", "operator")


def test_second_close_does_not_log_the_trials_again(fake_logger) -> None:
    summaries: list[dict] = []
    original = fake_logger.log_summary
    fake_logger.log_summary = lambda summary: (summaries.append(summary), original(summary))[1]
    client = TestClient(app)
    assert client.post("/evaluations", json=BODY).status_code == 200
    client.post("/evaluations/current/trial", json={"outcome": "place"})

    first = client.post("/evaluations/current/close")
    assert first.status_code == 200
    assert first.json() == {"weave_url": WEAVE_URL}

    second = client.post("/evaluations/current/close")
    assert second.status_code == 400
    assert len(fake_logger.predictions) == 1
    assert summaries == [{"trials": 1}]


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


def test_evaluation_resolves_production_for_act(fake_logger, registry_downloads) -> None:
    body = {**BODY, "policy_family": "act", "version": "production"}

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


def test_checkpoint_dir_for_uses_the_smolvla_collection(forge_on, registry_downloads) -> None:
    from lelab.forge.registry import checkpoint_dir_for

    checkpoint_dir_for("smolvla", "production")

    assert registry_downloads[0]["collection"] == "so101-chess-smolvla"


def test_current_evaluation_is_closed_without_a_session() -> None:
    response = TestClient(app).get("/evaluations/current")

    assert response.status_code == 200
    assert response.json() == {"open": False}


def test_current_evaluation_reports_the_open_session(fake_logger) -> None:
    client = TestClient(app)
    assert client.post("/evaluations", json=BODY).status_code == 200
    client.post("/evaluations/current/trial", json={"outcome": "success"})

    response = client.get("/evaluations/current")

    assert response.json() == {
        "open": True,
        "policy_family": "smolvla",
        "version": "candidate",
        "policy_host": "local",
        "policy_runs_on": "this machine",
        "policy_host_state": "off",
        "policy_host_error": None,
        "trials": len(TRANSFER_PLAN),
        "finished": 1,
        "next_transfer": list(TRANSFER_PLAN[1]),
        "next_class": transfer_class(TRANSFER_PLAN[1]),
    }


def test_current_evaluation_closes_with_the_session(fake_logger) -> None:
    client = TestClient(app)
    client.post("/evaluations", json=BODY)
    client.post("/evaluations/current/close", json={})

    assert client.get("/evaluations/current").json() == {"open": False}


def test_a_coreweave_session_logs_local_as_the_host_and_keeps_the_request(fake_logger, fake_host) -> None:
    fake_host.start_release.set()
    fake_host.ready_release.set()
    client = TestClient(app)
    assert client.post("/evaluations", json=COREWEAVE_BODY).status_code == 200
    _join_host_thread()
    client.post("/evaluations/current/trial", json={"outcome": "success"})
    client.post("/evaluations/current/close", json={})

    inputs, output, _ = fake_logger.predictions[0]

    assert inputs["policy_host"] == "local"
    assert output["policy_host"] == "local"
    assert output["requested_policy_host"] == "coreweave"


def test_a_local_session_logs_local_as_the_host(fake_logger) -> None:
    client = TestClient(app)
    assert client.post("/evaluations", json=BODY).status_code == 200
    client.post("/evaluations/current/trial", json={"outcome": "success"})
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
    monkeypatch.setattr(evaluate, "artifact_path_for", lambda family, alias: ARTIFACT_PATH)
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


def test_trials_are_logged_with_the_cameras_the_checkpoint_reads(fake_log, tmp_path) -> None:
    config = {
        "input_features": {
            "observation.images.wrist": {"type": "VISUAL", "shape": [3, 480, 640]},
            "observation.images.top": {"type": "VISUAL", "shape": [3, 480, 640]},
            "observation.state": {"type": "STATE", "shape": [6]},
        }
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    session = EvaluationSession(_request(), fake_log)
    session.policy_dir = tmp_path
    session.record_trial("success", 3.0)
    session.record_trial("success", 3.0)
    assert fake_log.rows[0].cameras == ("top", "wrist")


def test_a_checkpoint_without_a_readable_config_logs_no_cameras(fake_log, tmp_path) -> None:
    session = EvaluationSession(_request(), fake_log)
    session.policy_dir = tmp_path  # no config.json
    session.record_trial("success", 3.0)
    session.record_trial("success", 3.0)
    assert fake_log.rows[0].cameras == ()
