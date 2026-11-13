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

import threading
from typing import Literal

from pydantic import BaseModel

from .forge.judge import STAGES, JudgeVerdict
from .forge.weave_eval import TrialRow, WeaveTrialLog
from .utils.forge_config import forge_configured, load_forge_config

SMOLVLA_SKILLS = ("bolt", "bracket", "gear")
MIN_TRIALS = 1
MAX_TRIALS = 50


class EvaluationBusyError(RuntimeError):
    """An evaluation session is already open."""


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
        for trial in self._finished:
            self._log.log_trial(
                TrialRow(
                    skill=trial["skill"],
                    stage=trial["stage"],
                    verdict_source=trial["verdict_source"],
                    duration_s=trial["duration_s"],
                    chunk_latency_s=None,
                    policy_host=self.request.policy_host,
                    top_video=None,
                    side_video=None,
                )
            )
        return {"weave_url": self._log.finish()}

    def _verdict(self) -> dict:
        assert self._last is not None
        return {
            "skill": self._last["skill"],
            "stage": self._last["stage"],
            "verdict_source": self._last["verdict_source"],
        }


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


def handle_start_evaluation(request: EvaluationRequest) -> dict:
    global _session
    cfg = load_forge_config()
    if not forge_configured(cfg):
        raise ValueError("Evaluation needs Weights & Biases")
    session_skills(request)
    with _lock:
        if _session is not None:
            raise EvaluationBusyError("An evaluation session is already open")
        _session = create_session(request, cfg)
        return {
            "policy_family": request.policy_family,
            "version": request.version,
            "policy_host": request.policy_host,
            "trials": len(_session.skills),
        }


def handle_override(stage: str | None) -> dict:
    with _lock:
        if _session is None:
            raise ValueError("No evaluation session is open")
        return _session.override(stage)


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
