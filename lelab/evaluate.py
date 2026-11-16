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

"""One evaluation session: a plan of skill trials and the operator's verdicts."""

import base64
import binascii
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import BaseModel

from .forge.judge import STAGES, JudgeVerdict, judge_frames
from .forge.policy_server import FAILURE_SUFFIX, FeatureSpecDict, PolicyHost, checkpoint_features
from .forge.registry import artifact_path_for, checkpoint_dir_for
from .forge.weave_eval import TrialRow, WeaveTrialLog
from .utils.forge_config import ForgeConfig, forge_configured, load_forge_config, sandbox_configured

logger = logging.getLogger(__name__)

SMOLVLA_SKILLS = ("bolt", "bracket", "gear")
MIN_TRIALS = 1
MAX_TRIALS = 50
# How long the CoreWeave policy host may take to download the checkpoint and load the model.
READY_TIMEOUT_S = 20 * 60
SANDBOX_NOT_CONFIGURED = "The CoreWeave policy host needs CWSANDBOX_API_KEY and CWSANDBOX_TRAIN_IMAGE"
HOST_STARTING_MESSAGE = "The CoreWeave policy host is still starting. Try again in a minute."
HOST_STOPPED_MESSAGE = "The CoreWeave policy host is not running. Close the evaluation and reopen it."


class EvaluationBusyError(RuntimeError):
    """An evaluation session is already open."""


class PolicyHostNotReady(RuntimeError):  # noqa: N818 - the name is the rollout API
    """The CoreWeave policy host cannot serve a rollout right now; the message says why."""


class RolloutTarget(NamedTuple):
    """Where a rollout's policy runs: a local checkpoint directory or a remote endpoint."""

    kind: Literal["local", "remote"]
    value: str


@dataclass
class PolicyHostState:
    state: Literal["off", "starting", "ready", "failed"] = "off"
    endpoint: str | None = None
    error: str | None = None


def make_policy_host(cfg: ForgeConfig) -> PolicyHost:
    """The real policy host on a CoreWeave Sandbox; tests replace this."""
    from .runners.cwsandbox_client import CwsandboxClient

    return PolicyHost(CwsandboxClient(), cfg)


def _unexpected_failure(exc: Exception) -> str:
    # The exception text may carry secrets, so only its type is exposed.
    return f"The CoreWeave policy host could not be set up ({type(exc).__name__}).{FAILURE_SUFFIX}"


class EvaluationRequest(BaseModel):
    policy_family: Literal["smolvla", "act"]
    version: Literal["candidate", "production"]
    policy_host: Literal["local", "coreweave"]
    trials_per_skill: int
    skill: str | None = None


def session_skills(request: EvaluationRequest) -> list[str]:
    if not MIN_TRIALS <= request.trials_per_skill <= MAX_TRIALS:
        raise ValueError(f"trials_per_skill must be between {MIN_TRIALS} and {MAX_TRIALS}")
    if request.policy_family == "act":
        if request.skill is None:
            raise ValueError("skill is required for act")
        if not request.skill.strip():
            raise ValueError("skill is required for act")
        skills: tuple[str, ...] = (request.skill,)
    else:
        skills = SMOLVLA_SKILLS
    return [skill for skill in skills for _ in range(request.trials_per_skill)]


class EvaluationSession:
    def __init__(self, request: EvaluationRequest, log: WeaveTrialLog) -> None:
        self.request = request
        self.skills = session_skills(request)
        self._log = log
        self._started = 0
        self._started_skills: list[str] = []
        self._running = False
        self._last: dict | None = None
        self._finished: list[dict] = []
        self._logged = 0
        # Registry checkpoint (pretrained_model dir) the rollouts of this session load.
        self.policy_dir: Path | None = None
        self.policy_host = PolicyHostState()
        # Guards policy_host and the fields below; never held while the host starts or waits.
        self._host_lock = threading.Lock()
        self._host: PolicyHost | None = None
        self._host_cancelled = False
        self._host_thread: threading.Thread | None = None

    def start_trial(self, skill: str) -> dict:
        if self._running:
            raise ValueError("Finish the running trial first")
        if self._started >= len(self.skills):
            raise ValueError("Every planned trial has been run")
        allowed = sorted(set(self.skills))
        if skill not in allowed:
            raise ValueError(f"Skill {skill!r} is not in this session ({', '.join(allowed)})")
        if self.skills.count(skill) <= self._started_skills.count(skill):
            raise ValueError(f"Every planned {skill} trial has been run")
        # The previous trial can no longer be overridden once the next one begins.
        self._log_pending()
        self._started += 1
        self._running = True
        self._last = {"skill": skill, "stage": None, "verdict_source": "judge"}
        self._started_skills.append(skill)
        return {"skill": skill, "index": self._started, "status": "running"}

    def finish_trial(self, verdict: JudgeVerdict, duration_s: float) -> dict:
        if not self._running or self._last is None:
            raise ValueError("No trial is running")
        self._running = False
        self._last = {
            "skill": self._last["skill"],
            "stage": verdict.stage if verdict.status == "scored" else None,
            "verdict_source": "judge",
            "duration_s": duration_s,
        }
        self._finished.append(self._last)
        return self._verdict()

    def override(self, stage: str | None) -> dict:
        if stage is not None and stage not in STAGES:
            raise ValueError(f"Unknown stage {stage!r}; expected one of {', '.join(STAGES)}")
        if self._last is None or self._running:
            raise ValueError("No finished trial to override")
        self._last["stage"] = stage
        self._last["verdict_source"] = "operator"
        return self._verdict()

    def close(self) -> dict:
        try:
            self._log_pending()
            return {"weave_url": self._log.finish()}
        finally:
            self.stop_policy_host()

    def start_policy_host(
        self,
        cfg: ForgeConfig,
        features: list[FeatureSpecDict],
        action: FeatureSpecDict,
        model_subdir: str,
    ) -> None:
        """Start the CoreWeave policy host in the background; `summary()` reports its state."""
        with self._host_lock:
            self.policy_host = PolicyHostState(state="starting")
            self._host_cancelled = False
        thread = threading.Thread(
            target=self._run_policy_host,
            args=(cfg, features, action, model_subdir),
            name="coreweave-policy-host",
            daemon=True,
        )
        self._host_thread = thread
        thread.start()

    def stop_policy_host(self) -> None:
        """Stop the host quietly, also while it is still starting (the thread then stops it)."""
        with self._host_lock:
            self._host_cancelled = True
            host, self._host = self._host, None
            if self.policy_host.state in ("starting", "ready"):
                self.policy_host = PolicyHostState()
        if host is not None:
            _stop_quietly(host)

    def rollout_target(self) -> RolloutTarget | None:
        if self.request.policy_host == "local":
            if self.policy_dir is None:
                return None
            return RolloutTarget("local", str(self.policy_dir))
        with self._host_lock:
            host = self.policy_host
            if host.state == "ready" and host.endpoint is not None:
                return RolloutTarget("remote", host.endpoint)
            if host.state == "starting":
                raise PolicyHostNotReady(HOST_STARTING_MESSAGE)
            if host.state == "failed" and host.error:
                raise PolicyHostNotReady(host.error)
        raise PolicyHostNotReady(HOST_STOPPED_MESSAGE)

    def _run_policy_host(
        self,
        cfg: ForgeConfig,
        features: list[FeatureSpecDict],
        action: FeatureSpecDict,
        model_subdir: str,
    ) -> None:
        host: PolicyHost | None = None
        try:
            host = make_policy_host(cfg)
            artifact_path = artifact_path_for(
                self.request.policy_family, self.request.skill, self.request.version
            )
            endpoint = host.start(artifact_path, model_subdir, features, action)
            with self._host_lock:
                cancelled = self._host_cancelled
                if not cancelled:
                    self._host = host
            if cancelled:
                _stop_quietly(host)
                return
            host.wait_until_ready(READY_TIMEOUT_S)
        except RuntimeError as exc:
            # Only PolicyHost raises RuntimeError here, with a message it has already redacted.
            self._fail_policy_host(host, str(exc))
            return
        except Exception as exc:
            self._fail_policy_host(host, _unexpected_failure(exc))
            return
        with self._host_lock:
            if self._host_cancelled:
                return
            self.policy_host = PolicyHostState(state="ready", endpoint=endpoint)

    def _fail_policy_host(self, host: PolicyHost | None, error: str) -> None:
        with self._host_lock:
            self._host = None
            if not self._host_cancelled:
                self.policy_host = PolicyHostState(state="failed", error=error)
        if host is not None:
            _stop_quietly(host)

    def summary(self) -> dict:
        with self._host_lock:
            host_state = self.policy_host.state
            host_error = self.policy_host.error
        return {
            "open": True,
            "policy_family": self.request.policy_family,
            "version": self.request.version,
            "policy_host": self.request.policy_host,
            "policy_runs_on": "CoreWeave" if self.request.policy_host == "coreweave" else "this machine",
            "policy_host_state": host_state,
            "policy_host_error": host_error,
            "trials": len(self.skills),
            "finished": len(self._finished),
            "running": self._running,
        }

    def _log_pending(self) -> None:
        while self._logged < len(self._finished):
            trial = self._finished[self._logged]
            self._log.log_trial(
                TrialRow(
                    skill=trial["skill"],
                    stage=trial["stage"],
                    verdict_source=trial["verdict_source"],
                    duration_s=trial["duration_s"],
                    chunk_latency_s=None,
                    policy_host=self.request.policy_host,
                    requested_policy_host=self.request.policy_host,
                    top_video=None,
                    side_video=None,
                )
            )
            self._logged += 1

    def _verdict(self) -> dict:
        assert self._last is not None
        return {
            "skill": self._last["skill"],
            "stage": self._last["stage"],
            "verdict_source": self._last["verdict_source"],
        }


def _stop_quietly(host: PolicyHost) -> None:
    try:
        host.stop()
    except Exception as exc:
        logger.warning("Could not stop the CoreWeave policy host: %s", type(exc).__name__)


def create_session(request: EvaluationRequest, cfg) -> EvaluationSession:
    import weave

    weave.init(f"{cfg.entity}/{cfg.project}")
    logger = weave.EvaluationLogger(model=f"{request.policy_family}-{request.version}")
    return EvaluationSession(request, WeaveTrialLog(logger, entity=cfg.entity, project=cfg.project))


_lock = threading.Lock()
_session: EvaluationSession | None = None


def reset_session() -> None:
    global _session
    with _lock:
        _session = None


def current_session() -> EvaluationSession:
    with _lock:
        if _session is None:
            raise ValueError("No evaluation session is open")
        return _session


def resolve_policy_dir(request: EvaluationRequest) -> Path:
    """Download the Registry candidate/production version this session evaluates.

    An ACT session covers a single skill, so its first trial's checkpoint is resolved here too.
    """
    return checkpoint_dir_for(request.policy_family, request.skill, request.version)


def _read_checkpoint_features(policy_dir: Path) -> tuple[list[FeatureSpecDict], FeatureSpecDict]:
    """The policy server's feature list, from the checkpoint the Mac already downloaded."""
    try:
        config = json.loads((policy_dir / "config.json").read_text())
    except (OSError, ValueError):
        raise ValueError(
            "The checkpoint's config.json could not be read, so the CoreWeave policy host cannot be set up."
        ) from None
    if not isinstance(config, dict):
        raise ValueError(
            "The checkpoint's config.json is not a JSON object, so the CoreWeave policy host cannot be set up."
        )
    return checkpoint_features(config)


def handle_start_evaluation(request: EvaluationRequest) -> dict:
    global _session
    cfg = load_forge_config()
    if not forge_configured(cfg):
        raise ValueError("Evaluation needs Weights & Biases")
    session_skills(request)
    if request.policy_host == "coreweave" and not sandbox_configured(cfg):
        raise ValueError(SANDBOX_NOT_CONFIGURED)
    with _lock:
        if _session is not None:
            raise EvaluationBusyError("An evaluation session is already open")
        policy_dir = resolve_policy_dir(request)
        host_setup = _read_checkpoint_features(policy_dir) if request.policy_host == "coreweave" else None
        _session = create_session(request, cfg)
        _session.policy_dir = policy_dir
        if host_setup is not None:
            model_subdir = "pretrained_model" if policy_dir.name == "pretrained_model" else ""
            _session.start_policy_host(cfg, host_setup[0], host_setup[1], model_subdir)
        return {
            "policy_family": request.policy_family,
            "version": request.version,
            "policy_host": request.policy_host,
            "policy_path": str(policy_dir),
            "trials": len(_session.skills),
        }


def active_policy_path() -> str | None:
    """Checkpoint directory an open evaluation selected, for the rollout process."""
    try:
        session = current_session()
    except ValueError:
        return None
    if session.policy_dir is None:
        return None
    return str(session.policy_dir)


def rollout_target() -> RolloutTarget | None:
    """Where the open evaluation's rollouts run their policy; None when no evaluation is open.

    Raises PolicyHostNotReady while a CoreWeave session's host is starting, failed or stopped.
    """
    try:
        session = current_session()
    except ValueError:
        return None
    return session.rollout_target()


def stop_policy_host() -> None:
    """Stop the open evaluation's CoreWeave policy host, if any. Never raises."""
    # No `_lock`: it can be held for minutes by a checkpoint download, and shutdown must not wait.
    session = _session
    if session is not None:
        session.stop_policy_host()


def handle_evaluation_status() -> dict:
    with _lock:
        if _session is None:
            return {"open": False}
        return _session.summary()


def handle_override(stage: str | None) -> dict:
    with _lock:
        if _session is None:
            raise ValueError("No evaluation session is open")
        return _session.override(stage)


class TrialFramesRequest(BaseModel):
    skill: str
    top_jpeg_base64: str
    side_jpeg_base64: str
    duration_s: float


def _decode_frames(top_b64: str, side_b64: str) -> tuple[bytes, bytes] | None:
    try:
        top = base64.b64decode(top_b64, validate=True)
        side = base64.b64decode(side_b64, validate=True)
    except (binascii.Error, ValueError):
        return None
    if not top or not side:
        return None
    return top, side


def handle_trial(request: TrialFramesRequest) -> dict:
    with _lock:
        if _session is None:
            raise ValueError("No evaluation session is open")
        _session.start_trial(request.skill)
        frames = _decode_frames(request.top_jpeg_base64, request.side_jpeg_base64)
        if frames is None:
            return _session.finish_trial(JudgeVerdict(None, None, "needs_review"), request.duration_s)
        import openai

        cfg = load_forge_config()
        client = openai.OpenAI(base_url="https://api.inference.wandb.ai/v1", api_key=cfg.api_key)
        verdict = judge_frames(client, cfg, frames[0], frames[1])
        return _session.finish_trial(verdict, request.duration_s)


def handle_close_evaluation() -> dict:
    global _session
    with _lock:
        if _session is None:
            raise ValueError("No evaluation session is open")
        session = _session
        try:
            return session.close()
        finally:
            _session = None
