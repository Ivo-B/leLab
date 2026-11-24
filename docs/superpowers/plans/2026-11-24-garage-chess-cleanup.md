# Garage Chess cleanup: one project, chess names, real evaluation on the transfer plan — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `forge/chess` carries only the chess demo: no kitting skills anywhere, every Forge name is `so101-chess*`, sim and real datasets are separate artifacts, and the real-arm Evaluate page runs the same 20-transfer plan as the sim evaluation, with the operator recording the sim judge's outcome classes into the same Weave evaluation.

**Architecture:** Kitting stays on the old `forge/*` branches; this plan removes it from `forge/chess`. The skill argument disappears from the registry helpers, training request, job runners and the Train page. `lelab/chess/transfers.py` gains the outcome vocabulary (`success` plus the sim judge's failures) so `lelab/sim/judge.py` and `lelab/evaluate.py` share it. The real evaluation session walks `TRANSFER_PLAN` × rounds; the operator records one outcome per transfer, and each trial is logged to `so101-chess-sim-eval` with `domain="real"` next to the sim trials (`domain="sim"`). The VLM stage judge is removed.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, W&B Artifacts and Weave, React + Vite + vitest.

**North star (spec `docs/superpowers/specs/2026-11-23-garage-chess-voice-design.md`, section 1):** a person plays a full chess game by voice against the real SO-101. The arm moves with a policy trained in simulation and adapted with fewer than 100 real correction episodes to ≥ 80 % on the real transfer plan; a Stockfish panel analyses the game; every artifact is visible in Forge under `so101-chess`. Decisions taken on 2026-11-24: kitting lives only on its old branches; the W&B project default is `so101-chess`; the full spec (M7) is the target; real trials are judged by the operator in the sim judge's outcome classes.

## Global Constraints

- Work on branch `forge/chess`. Push only to `origin` (`https://github.com/Ivo-B/leLab.git`), and only when the controller asks. Never push `upstream`.
- The working tree holds uncommitted camera-control, lens and colour-calibration work (`lelab/camera_config.py`, `lelab/record.py`, `lelab/rollout.py`, `lelab/scripts/lelab.py`, `lelab/server.py`, `lelab/utils/config.py`, `pyproject.toml`, `uv.lock`, `frontend/src/components/control/*`, `frontend/src/components/recording/*`, `frontend/src/contexts/ApiContext.tsx`, `frontend/src/hooks/*`, `frontend/src/lib/camera*`, `frontend/vite.config.ts`, `tests/conftest.py`, `tests/test_camera_config.py`, `tests/test_scripts_lelab.py`, `CHESS_DEMO_GUIDE.md`, `README.md`, three untracked plans). Do not commit, revert or reformat it. Stage files by name. Never `git add -A`, `git add .` or `git commit -a`. For `lelab/server.py`, `README.md` and `CHESS_DEMO_GUIDE.md`, stage only this plan's hunks with `git add -p`. Do not skip hooks.
- Names are exact: W&B project default `so101-chess`; collections `so101-chess-act`, `so101-chess-smolvla`; dataset artifacts `so101-chess-sim` and `so101-chess-real`; Weave evaluation `so101-chess-sim-eval`; policy-server deployment `so101-chess`; automation `so101-chess evaluation`.
- Outcome vocabulary is exact and ordered: `success`, `approach`, `grasp`, `drop`, `place`, `topple`, `disturbed`.
- `lelab/chess/` never imports `lelab.sim`, `lerobot.robots`, `lelab.record`, `lelab.forge` or `openai`.
- Do not start the robot arm in a unit test. Unit tests never call W&B, Weave, Serverless Inference or the network.
- `WANDB_API_KEY` and `CWSANDBOX_API_KEY` never appear in a response, a log line, or a logger call.
- New Python modules start with the Apache-2.0 header in `tests/test_forge_config.py` lines 1–13, then a blank line.
- Run Python tests with `uv run pytest`, frontend tests with `npm test -- --run` inside `frontend/`, type-check with `npx tsc --noEmit -p tsconfig.app.json` inside `frontend/`, lint with `uv run ruff check` and `uv run ruff format --check` on the touched Python files.
- After the last task, run `graphify update .` in the repo root.

---

### Task 1: Remove the skill from the registry, training and job runners

**Files:**

- Modify: `lelab/forge/registry.py` (`collection_name`, `artifact_path_for`, `checkpoint_dir_for`)
- Modify: `lelab/forge/tracking.py` (`_link_last_checkpoint`)
- Modify: `lelab/forge/sandbox.py` (`sandbox_create_kwargs`)
- Modify: `lelab/runners/sandbox.py` (the `sandbox_create_kwargs` call)
- Modify: `lelab/jobs.py` (`LocalJobRunner.start` env, `JobRegistry.start` skill check)
- Modify: `lelab/train.py` (`TrainingRequest.skill`)
- Modify: `lelab/evaluate.py` (call sites of `artifact_path_for` and `checkpoint_dir_for` only; Task 3 rewrites the rest)
- Modify: `lelab/server.py` (`RegistryPromoteBody`, `promote_registry_alias`) — stage with `git add -p`
- Modify tests: `tests/test_forge_registry.py`, `tests/test_forge_tracking.py`, `tests/test_forge_sandbox.py`, `tests/test_forge_launch.py`, `tests/test_forge_sandbox_runner.py`, `tests/test_train.py`, `tests/test_forge_evaluate.py` (call sites only)

**Interfaces:**

- Produces: `collection_name(policy_type: str) -> str` (`"smolvla"` → `so101-chess-smolvla`, `"act"` → `so101-chess-act`, else `ValueError`). `artifact_path_for(policy_family: str, alias: str) -> str`. `checkpoint_dir_for(policy_family: str, alias: str, store: RegistryStore | None = None) -> Path`. `sandbox_create_kwargs(cfg: ForgeConfig) -> dict[str, object]` with no `FORGE_SKILL` in `environment_variables`. `TrainingRequest` has no `skill` field. `POST /registry/promote` takes `{"policy_family": "smolvla" | "act"}`.

- [ ] **Step 1: Rewrite the tests**

In `tests/test_forge_registry.py`, replace the collection tests with:

```python
def test_each_policy_family_has_one_chess_collection() -> None:
    assert collection_name("smolvla") == "so101-chess-smolvla"
    assert collection_name("act") == "so101-chess-act"


def test_unknown_policy_type_has_no_collection() -> None:
    with pytest.raises(ValueError):
        collection_name("groot")
```

Change every `artifact_path_for("act", "bolt", "production")` to `artifact_path_for("act", "production")` and every `checkpoint_dir_for(family, skill, alias, ...)` to `checkpoint_dir_for(family, alias, ...)`. Replace the `episode_counts` fixture rows `("pick the bolt",)`/`("pick the bracket",)` with two rows of `(CHESS_TASK,)` and assert `{CHESS_TASK: 2}` (import `CHESS_TASK` from `lelab.chess.transfers`).

In `tests/test_forge_sandbox.py`: call `sandbox_create_kwargs(_cfg())`; assert `"FORGE_SKILL" not in env`; delete `test_blank_skill_is_an_empty_env_value`; drop `skill="bolt"` from any `TrainingRequest`.

In `tests/test_forge_launch.py`: replace `test_local_runner_exports_the_skill` with

```python
def test_local_runner_does_not_export_a_skill(monkeypatch, tmp_path) -> None:
    seen: dict = {}
    monkeypatch.setattr(LocalJobRunner, "_spawn", lambda self, cmd, thread_name, extra_env=None: seen.update(env=extra_env))
    LocalJobRunner().start("job", TrainingRequest(dataset_repo_id="me/so101", tracking="wandb"), str(tmp_path))
    assert "FORGE_SKILL" not in (seen["env"] or {})
```

Keep the existing imports and fakes of that file; adapt the `_spawn` patch to the signature the file already uses.

In `tests/test_forge_tracking.py`: delete every `monkeypatch.setenv("FORGE_SKILL", ...)`/`delenv`; rename `test_finish_links_act_without_a_skill` to `test_finish_links_act_into_the_chess_collection` and assert the link path ends with `so101-chess-act`. In `tests/test_forge_sandbox_runner.py` and `tests/test_train.py`: drop `skill="bolt"` from the `TrainingRequest`s. In `tests/test_forge_evaluate.py`: update only the `artifact_path_for`/`checkpoint_dir_for` fakes to the new signatures.

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_forge_registry.py tests/test_forge_sandbox.py tests/test_forge_launch.py tests/test_forge_tracking.py -q`
Expected: FAIL with `TypeError` on `collection_name()` / `sandbox_create_kwargs()` arity.

- [ ] **Step 3: Implement**

`lelab/forge/registry.py`:

```python
def collection_name(policy_type: str) -> str:
    """The chess Registry collection of a policy family."""
    if policy_type == "smolvla":
        return "so101-chess-smolvla"
    if policy_type == "act":
        return "so101-chess-act"
    raise ValueError(f"No registry collection for policy type {policy_type}")
```

`artifact_path_for(policy_family, alias)` and `checkpoint_dir_for(policy_family, alias, store=None)` call `collection_name(policy_family)`. Fix the `checkpoint_dir_for` docstring: "SmolVLA and ACT each have one collection."

`lelab/forge/tracking.py` `_link_last_checkpoint`: delete the `FORGE_SKILL` lines and call `collection_name(self._cfg.policy.type)`. Remove the `os` import if it becomes unused.

`lelab/forge/sandbox.py`: `def sandbox_create_kwargs(cfg: ForgeConfig) -> dict[str, object]:` with `env: dict[str, str] = {"PYTHONUNBUFFERED": "1"}`. `lelab/runners/sandbox.py`: `kwargs = sandbox_create_kwargs(cfg)`.

`lelab/jobs.py`: in `LocalJobRunner.start`, call `self._spawn(cmd, thread_name=f"job-{job_id}-stdout")` without `extra_env` (keep the parameter on `_spawn`). In `JobRegistry.start`, delete the five lines from `skill = (config.skill or "").strip()` through `collection_name("act", skill)`.

`lelab/train.py`: delete `skill: str | None = None`.

`lelab/evaluate.py`: `artifact_path_for(self.request.policy_family, self.request.version)` and `checkpoint_dir_for(request.policy_family, request.version)`.

`lelab/server.py`: `RegistryPromoteBody` keeps only `policy_family`; `collection = collection_name(body.policy_family)`.

- [ ] **Step 4: Run the whole Python suite**

Run: `uv run pytest -q`
Expected: pass. `rg -n "FORGE_SKILL|\.skill\b" lelab` prints only `lelab/evaluate.py` lines (Task 3 removes them).

- [ ] **Step 5: Commit**

```bash
git add lelab/forge/registry.py lelab/forge/tracking.py lelab/forge/sandbox.py lelab/runners/sandbox.py lelab/jobs.py lelab/train.py lelab/evaluate.py tests/test_forge_registry.py tests/test_forge_tracking.py tests/test_forge_sandbox.py tests/test_forge_launch.py tests/test_forge_sandbox_runner.py tests/test_train.py tests/test_forge_evaluate.py
git add -p lelab/server.py   # only the RegistryPromoteBody / promote hunks
git commit -m "refactor: drop kitting skills from the registry, training and runners"
```

---

### Task 2: Remove the Skill field from the Train page

**Files:**

- Modify: `frontend/src/components/training/tracking.ts`
- Modify: `frontend/src/components/training/config/EssentialsCard.tsx`
- Modify: `frontend/src/components/training/types.ts`
- Modify: `frontend/src/lib/jobsApi.ts`
- Modify: `frontend/src/pages/Training.tsx`
- Modify: the tests that cover them (`rg -l "skill|isSkillSlug" frontend/src --glob '*.test.*'`, except `Evaluate.test.tsx`, which Task 4 rewrites)

**Interfaces:**

- Consumes: `POST /jobs/training` without `skill` (Task 1).
- Produces: `trackingPayload(config: { trackio_enable: boolean; wandb_enable: boolean }): { tracking: "off" | "trackio" | "wandb" }`. `isSkillSlug` no longer exists.

- [ ] **Step 1: Update the tests**

In the `tracking.ts` test, replace the skill cases with:

```ts
it("never sends a skill", () => {
  expect(trackingPayload({ trackio_enable: false, wandb_enable: true })).toEqual({ tracking: "wandb" });
  expect(trackingPayload({ trackio_enable: true, wandb_enable: false })).toEqual({ tracking: "trackio" });
  expect(trackingPayload({ trackio_enable: false, wandb_enable: false })).toEqual({ tracking: "off" });
});
```

In any EssentialsCard or Training test that types into the Skill field, replace that case with an assertion that `screen.queryByLabelText("Skill")` is `null` when W&B is on and the policy is ACT.

- [ ] **Step 2: Run them and confirm they fail**

Run: `cd frontend && npm test -- --run src/components/training src/pages`
Expected: FAIL (the Skill field is still rendered).

- [ ] **Step 3: Implement**

`tracking.ts` becomes:

```ts
export function trackingPayload(config: {
  trackio_enable: boolean;
  wandb_enable: boolean;
}): { tracking: "off" | "trackio" | "wandb" } {
  if (config.wandb_enable) return { tracking: "wandb" };
  if (config.trackio_enable) return { tracking: "trackio" };
  return { tracking: "off" };
}
```

`EssentialsCard.tsx`: delete the `isSkillSlug` import, the `skill`/`skillInvalid` constants and the whole `{config.wandb_enable && config.policy_type === 'act' && (...)}` block. `types.ts` and `jobsApi.ts`: delete `skill?: string;`. `Training.tsx`: import only `trackingPayload`; delete the skill-slug toast block in `handleStart`.

- [ ] **Step 4: Run frontend tests and the type check**

Run: `cd frontend && npm test -- --run && npx tsc --noEmit -p tsconfig.app.json`
Expected: pass except `src/pages/Evaluate.test.tsx` cases that Task 4 rewrites; record which ones fail.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/training/tracking.ts frontend/src/components/training/config/EssentialsCard.tsx frontend/src/components/training/types.ts frontend/src/lib/jobsApi.ts frontend/src/pages/Training.tsx
# plus only the test files you changed
git commit -m "refactor: remove the kitting Skill field from the Train page"
```

---

### Task 3: Real evaluation walks the transfer plan with operator outcomes

**Files:**

- Modify: `lelab/chess/transfers.py` (add `FAILURES`, `OUTCOMES`)
- Modify: `lelab/sim/judge.py` (import `FAILURES` from `lelab.chess.transfers`)
- Modify: `lelab/sim/evaluate.py` (`log_to_weave` inputs gain `"domain": "sim"`)
- Rewrite: `lelab/evaluate.py` (request, session, trial and override handlers)
- Rewrite: `lelab/forge/weave_eval.py` (`TrialRow`, `WeaveTrialLog.log_trial`)
- Delete: `lelab/forge/judge.py`, `tests/test_forge_judge.py`
- Modify: `lelab/utils/forge_config.py` (remove `judge_model` and `FORGE_JUDGE_MODEL`)
- Modify: `lelab/server.py` (`EvaluationOverrideBody`, trial route body) — stage with `git add -p`
- Rewrite: `tests/test_forge_evaluate.py`, `tests/test_forge_weave_eval.py`; modify `tests/test_sim_evaluate.py`, `tests/test_forge_config.py` where they reference `judge_model` or the Weave inputs

**Interfaces:**

- Produces in `lelab/chess/transfers.py`: `FAILURES = ("approach", "grasp", "drop", "place", "topple", "disturbed")`, `OUTCOMES = ("success", *FAILURES)`.
- Produces in `lelab/evaluate.py`:
  - `EvaluationRequest(policy_family: Literal["smolvla","act"], version: Literal["candidate","production"], policy_host: Literal["local","coreweave"], rounds: int = 1)`; `MIN_ROUNDS = 1`, `MAX_ROUNDS = 5`.
  - `session_plan(request) -> list[Transfer]` = `list(TRANSFER_PLAN) * request.rounds`; raises `ValueError("rounds must be between 1 and 5")`.
  - `EvaluationSession.record_trial(outcome: str, duration_s: float) -> dict` records the outcome of the next planned transfer and returns `{"index": int, "transfer": [src, dst], "class": str, "outcome": str, "verdict_source": "operator"}`. Unknown outcome → `ValueError`; plan exhausted → `ValueError("Every planned trial has been run")`.
  - `EvaluationSession.override(outcome: str) -> dict` changes the last recorded outcome until the next trial is recorded; same return shape with `verdict_source="override"`.
  - `summary()` adds `"next_transfer": [src, dst] | None` and `"next_class": str | None`; `"trials"` is `len(plan)`.
  - `TrialOutcomeRequest(outcome: str, duration_s: float = 0.0)`; `handle_trial(request: TrialOutcomeRequest) -> dict`; `handle_override(outcome: str) -> dict`.
  - `create_session` opens `weave.EvaluationLogger(name="so101-chess-sim-eval", model=f"{family}-{version}")`.
- Produces in `lelab/forge/weave_eval.py`: `TrialRow(transfer: tuple[str, str], transfer_class: str, outcome: str, verdict_source: str, duration_s: float, policy_host: str, requested_policy_host: str | None = None)`. `log_trial` logs `inputs={"transfer": "src-dst", "class": ..., "domain": "real", "policy_host": ...}`, `output={"outcome", "verdict_source", "duration_s", "policy_host", "requested_policy_host"}`, and `log_score("success", 1 if outcome == "success" else 0)`.
- HTTP: `POST /evaluations` body `{policy_family, version, policy_host, rounds}`; `POST /evaluations/current/trial` body `{outcome, duration_s}`; `POST /evaluations/current/override` body `{outcome}`.

- [ ] **Step 1: Write the failing tests**

Replace the skill, stage and judge tests in `tests/test_forge_evaluate.py` (keep the policy-host, rollout-target, busy and close tests, adapting their requests to `rounds=1` and no `skill`). New core tests:

```python
from lelab.chess.transfers import OUTCOMES, TRANSFER_PLAN, transfer_class
from lelab.evaluate import EvaluationRequest, EvaluationSession, session_plan


def _request(**kw) -> EvaluationRequest:
    return EvaluationRequest(
        policy_family=kw.get("policy_family", "act"),
        version="candidate",
        policy_host="local",
        rounds=kw.get("rounds", 1),
    )


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
    with pytest.raises(ValueError, match="Every planned trial"):
        session.record_trial("success", 1.0)
```

`fake_log` is a fixture: an object with `rows: list`, `log_trial(row)` appending to `rows`, and `finish()` returning `"https://wandb.ai/team/so101-chess/weave"`. Route tests: `POST /evaluations/current/trial` with `{"outcome": "drop"}` returns 200 and the result above; with `{"outcome": "nope"}` returns 400.

In `tests/test_forge_weave_eval.py`:

```python
def test_a_real_trial_is_a_chess_row_with_domain_real() -> None:
    logger = FakeLogger()
    log = WeaveTrialLog(logger, entity="team", project="so101-chess")
    log.log_trial(TrialRow(("e2", "e3"), "near", "success", "operator", 3.5, "local", "local"))
    inputs, output = logger.predictions[0]
    assert inputs == {"transfer": "e2-e3", "class": "near", "domain": "real", "policy_host": "local"}
    assert output["outcome"] == "success"
    assert logger.scores[0] == ("success", 1)
```

Adapt `FakeLogger` in that file to record `(inputs, output)` and scores the way it already does. In `tests/test_sim_evaluate.py`, assert the logged inputs of a sim trial include `"domain": "sim"`. Add to `tests/test_chess_transfers.py` (or the existing transfers test file):

```python
def test_outcomes_are_success_then_the_sim_failures() -> None:
    from lelab.chess.transfers import FAILURES, OUTCOMES
    from lelab.sim.judge import FAILURES as SIM_FAILURES

    assert OUTCOMES == ("success", "approach", "grasp", "drop", "place", "topple", "disturbed")
    assert SIM_FAILURES is FAILURES
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_forge_evaluate.py tests/test_forge_weave_eval.py tests/test_sim_evaluate.py tests/test_chess_transfers.py -q`
Expected: FAIL with `ImportError` on `OUTCOMES` / `session_plan`.

- [ ] **Step 3: Implement**

`lelab/chess/transfers.py`, after `SIDECAR_NAME`:

```python
# How a transfer ends, in simulation (the judge) and on the real arm (the operator).
FAILURES = ("approach", "grasp", "drop", "place", "topple", "disturbed")
OUTCOMES = ("success", *FAILURES)
```

`lelab/sim/judge.py`: delete its `FAILURES` tuple and add `FAILURES` to the import from `..chess.transfers` (re-exported, so existing `from lelab.sim.judge import FAILURES` keeps working).

`lelab/sim/evaluate.py` `log_to_weave`: add `"domain": "sim",` to `inputs`.

`lelab/forge/weave_eval.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class TrialRow:
    transfer: tuple[str, str]
    transfer_class: str
    outcome: str
    verdict_source: str
    duration_s: float
    policy_host: str
    requested_policy_host: str | None = None


class WeaveTrialLog:
    def __init__(self, logger, entity: str, project: str) -> None:
        self._logger = logger
        self._entity = entity
        self._project = project
        self._count = 0

    def log_trial(self, row: TrialRow) -> None:
        prediction = self._logger.log_prediction(
            inputs={
                "transfer": "-".join(row.transfer),
                "class": row.transfer_class,
                "domain": "real",
                "policy_host": row.policy_host,
            },
            output={
                "outcome": row.outcome,
                "verdict_source": row.verdict_source,
                "duration_s": row.duration_s,
                "policy_host": row.policy_host,
                "requested_policy_host": row.requested_policy_host,
            },
        )
        prediction.log_score("success", 1 if row.outcome == "success" else 0)
        prediction.finish()
        self._count += 1

    def finish(self) -> str:
        self._logger.log_summary({"trials": self._count})
        return f"https://wandb.ai/{self._entity}/{self._project}/weave"
```

`lelab/evaluate.py` (keep the policy-host machinery, `RolloutTarget`, `PolicyHostState`, `resolve_policy_dir`, `_read_checkpoint_features`, `active_policy_path`, `rollout_target`, `stop_policy_host`, `handle_close_evaluation` as they are):

- Module docstring: `"""One real-arm evaluation session: the sim transfer plan, judged by the operator."""`
- Drop `base64`, `binascii`, the `forge.judge` import, `SMOLVLA_SKILLS`, `MIN_TRIALS`, `MAX_TRIALS`, `TrialFramesRequest`, `_decode_frames`.
- Add imports `from .chess.game import Transfer` and `from .chess.transfers import OUTCOMES, TRANSFER_PLAN, transfer_class`; `from .sim.evaluate import EVAL_NAME` is **not** allowed (it imports MuJoCo-side modules); define `EVAL_NAME = "so101-chess-sim-eval"` here and add a test asserting it equals `lelab.sim.evaluate.EVAL_NAME` (skip if MuJoCo is missing, like other sim tests).

```python
MIN_ROUNDS = 1
MAX_ROUNDS = 5
EVAL_NAME = "so101-chess-sim-eval"


class EvaluationRequest(BaseModel):
    policy_family: Literal["smolvla", "act"]
    version: Literal["candidate", "production"]
    policy_host: Literal["local", "coreweave"]
    rounds: int = 1


def session_plan(request: EvaluationRequest) -> list[Transfer]:
    if not MIN_ROUNDS <= request.rounds <= MAX_ROUNDS:
        raise ValueError(f"rounds must be between {MIN_ROUNDS} and {MAX_ROUNDS}")
    return list(TRANSFER_PLAN) * request.rounds


def _check_outcome(outcome: str) -> None:
    if outcome not in OUTCOMES:
        raise ValueError(f"Unknown outcome {outcome!r}; expected one of {', '.join(OUTCOMES)}")
```

In `EvaluationSession.__init__`: `self.plan = session_plan(request)`, `self._finished: list[dict] = []`, `self._logged = 0`; drop `_started`, `_started_skills`, `_running`, `_last`.

```python
    def record_trial(self, outcome: str, duration_s: float) -> dict:
        _check_outcome(outcome)
        if len(self._finished) >= len(self.plan):
            raise ValueError("Every planned trial has been run")
        # The previous trial can no longer be overridden once the next one is recorded.
        self._log_pending()
        transfer = self.plan[len(self._finished)]
        self._finished.append(
            {"transfer": transfer, "outcome": outcome, "verdict_source": "operator", "duration_s": duration_s}
        )
        return self._verdict()

    def override(self, outcome: str) -> dict:
        _check_outcome(outcome)
        if self._logged >= len(self._finished):
            raise ValueError("No recorded trial to override")
        self._finished[-1]["outcome"] = outcome
        self._finished[-1]["verdict_source"] = "override"
        return self._verdict()

    def _verdict(self) -> dict:
        trial = self._finished[-1]
        return {
            "index": len(self._finished),
            "transfer": list(trial["transfer"]),
            "class": transfer_class(trial["transfer"]),
            "outcome": trial["outcome"],
            "verdict_source": trial["verdict_source"],
        }

    def _log_pending(self) -> None:
        while self._logged < len(self._finished):
            trial = self._finished[self._logged]
            self._log.log_trial(
                TrialRow(
                    transfer=trial["transfer"],
                    transfer_class=transfer_class(trial["transfer"]),
                    outcome=trial["outcome"],
                    verdict_source=trial["verdict_source"],
                    duration_s=trial["duration_s"],
                    policy_host=self.request.policy_host,
                    requested_policy_host=self.request.policy_host,
                )
            )
            self._logged += 1
```

`summary()` keeps its keys, sets `"trials": len(self.plan)`, `"finished": len(self._finished)`, drops `"running"`, and adds:

```python
        upcoming = self.plan[len(self._finished)] if len(self._finished) < len(self.plan) else None
        ...
            "next_transfer": list(upcoming) if upcoming else None,
            "next_class": transfer_class(upcoming) if upcoming else None,
```

`create_session`: `weave.EvaluationLogger(name=EVAL_NAME, model=f"{request.policy_family}-{request.version}")`. `handle_start_evaluation`: call `session_plan(request)` instead of `session_skills(request)` and return `"trials": len(_session.plan)`.

```python
class TrialOutcomeRequest(BaseModel):
    outcome: str
    duration_s: float = 0.0


def handle_trial(request: TrialOutcomeRequest) -> dict:
    with _lock:
        if _session is None:
            raise ValueError("No evaluation session is open")
        return _session.record_trial(request.outcome, request.duration_s)


def handle_override(outcome: str) -> dict:
    with _lock:
        if _session is None:
            raise ValueError("No evaluation session is open")
        return _session.override(outcome)
```

`lelab/server.py`: import `TrialOutcomeRequest` instead of `TrialFramesRequest`; `class EvaluationOverrideBody(BaseModel): outcome: str`; `handle_override(body.outcome)`; the trial route takes `body: TrialOutcomeRequest`.

Delete `lelab/forge/judge.py` and `tests/test_forge_judge.py` (`git rm`). In `lelab/utils/forge_config.py` delete the `judge_model` field and its `FORGE_JUDGE_MODEL` line; remove any `judge_model` assertion in `tests/test_forge_config.py`.

- [ ] **Step 4: Run the whole Python suite and lint**

Run: `uv run pytest -q && uv run ruff check lelab tests && uv run ruff format --check lelab/evaluate.py lelab/forge/weave_eval.py lelab/chess/transfers.py lelab/sim/judge.py lelab/sim/evaluate.py lelab/utils/forge_config.py`
Expected: pass. `rg -n "skill|judge_frames|STAGES" lelab --glob '*.py'` prints nothing except `lelab/rollout.py` user messages about "a skill" (leave those).

- [ ] **Step 5: Commit**

```bash
git rm lelab/forge/judge.py tests/test_forge_judge.py
git add lelab/chess/transfers.py lelab/sim/judge.py lelab/sim/evaluate.py lelab/evaluate.py lelab/forge/weave_eval.py lelab/utils/forge_config.py tests/test_forge_evaluate.py tests/test_forge_weave_eval.py tests/test_sim_evaluate.py tests/test_forge_config.py
# plus the transfers test file you changed
git add -p lelab/server.py   # only the evaluation body hunks
git commit -m "feat: evaluate the real arm on the sim transfer plan with operator outcomes"
```

---

### Task 4: Evaluate page for the transfer plan

**Files:**

- Modify: `frontend/src/pages/Evaluate.tsx`
- Rewrite: `frontend/src/pages/Evaluate.test.tsx` (real-session and promote cases; keep the sim-evaluate cases)

**Interfaces:**

- Consumes: Task 3's HTTP bodies; `GET /evaluations/current` returns `next_transfer`, `next_class`, `trials`, `finished`.
- Produces: the page shows a **Rounds** number field (1–5, default 1) instead of Trials and Skill; an open session shows `Next: e2 → e3 (near) · 3 of 20`; an **Outcome** select with the seven `OUTCOMES` and a **Record outcome** button; after a trial, a **Correct last** button that posts the selected outcome to `/evaluations/current/override`. Promote posts `{policy_family}` only.

- [ ] **Step 1: Rewrite the tests**

Cases (use the file's existing `respond`/`posted` helpers and fetch mock):

1. Starting a session posts `{ policy_family: "smolvla", version: "candidate", policy_host: "local", rounds: 1 }` and there is no `Skill` field for either family.
2. With the status mock returning `{ open: true, trials: 20, finished: 2, next_transfer: ["e2", "e3"], next_class: "near", ... }`, the page shows `Next: e2 → e3 (near) · 3 of 20`.
3. Choosing `grasp` in **Outcome** and clicking **Record outcome** posts `{ outcome: "grasp", duration_s: 0 }` to `/evaluations/current/trial` and then re-reads `/evaluations/current`.
4. **Correct last** posts `{ outcome: "success" }` to `/evaluations/current/override`.
5. Promote posts `[{ policy_family: "act" }]` and shows `Promoted so101-chess-act v1 to production.` from a mocked `{ collection: "so101-chess-act", to_alias: "production", version: "v1" }`.

- [ ] **Step 2: Run them and confirm they fail**

Run: `cd frontend && npm test -- --run src/pages/Evaluate.test.tsx`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `Evaluate.tsx`:

```ts
const OUTCOMES = ["success", "approach", "grasp", "drop", "place", "topple", "disturbed"] as const;

interface Session {
  trials: number;
  finished: number;
  next_transfer: [string, string] | null;
  next_class: string | null;
}

interface Verdict {
  transfer: [string, string];
  class: string;
  outcome: string;
  verdict_source: string;
}

function nextText(session: Session): string {
  if (session.next_transfer === null) return `All ${session.trials} trials recorded.`;
  const [source, target] = session.next_transfer;
  return `Next: ${source} → ${target} (${session.next_class}) · ${session.finished + 1} of ${session.trials}`;
}

function verdictText(verdict: Verdict): string {
  const [source, target] = verdict.transfer;
  return `Last: ${source} → ${target} ${verdict.outcome} (${verdict.verdict_source})`;
}
```

- Replace `trials`/`skill` state with `rounds` (string, default `"1"`), valid when an integer in 1–5. `start` posts `{ policy_family, version, policy_host, rounds: Number(rounds) }`, then calls a `refresh()` that GETs `/evaluations/current` and stores the `Session`.
- Replace `runTrial` with `record()` posting `{ outcome, duration_s: 0 }` to `/evaluations/current/trial`, storing the `Verdict`, then `refresh()`. Replace `override(next)` with `correct()` posting `{ outcome }` to `/evaluations/current/override`.
- `promote` posts `{ policy_family: family }`. `canPromote = forge !== "off" && !busy`. `canStart = !busy && session === null && roundsValid`.
- The session panel shows `nextText(session)`, the latest `verdictText`, the **Outcome** select (`id="evaluate-outcome"`, label "Outcome") over `OUTCOMES`, **Record outcome** (disabled when `busy` or `session.next_transfer === null`), and **Correct last** (disabled when `verdict === null`).
- Replace `EMPTY_STATE` with `"Choose a checkpoint, then run the 20-transfer plan on the arm and record how each transfer ends."` Delete `STAGES`.

- [ ] **Step 4: Run frontend tests and the type check**

Run: `cd frontend && npm test -- --run && npx tsc --noEmit -p tsconfig.app.json`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/Evaluate.tsx frontend/src/pages/Evaluate.test.tsx
git commit -m "feat: Evaluate page runs the transfer plan and records outcomes"
```

---

### Task 5: Chess names for the W&B project, deployment and ARIA automation

**Files:**

- Modify: `lelab/utils/forge_config.py` (project default)
- Modify: `lelab/forge/policy_server.py` (`DEPLOYMENT`)
- Modify: `lelab/improve/aria.py` (`AUTOMATION_NAME`, `aria_prompt`)
- Modify: `frontend/src/pages/Improve.tsx` (fetch the automation from the backend)
- Modify: `lelab/server.py` (new `GET /improve/automation`) — stage with `git add -p`
- Modify: `frontend/src/components/setup/ForgeCredentialsForm.tsx` (default and placeholder)
- Modify tests: every file from `rg -l "so101-kitting" tests frontend/src`, plus `tests/test_improve_aria.py`, `frontend/src/pages/Improve.test.tsx`

**Interfaces:**

- Produces: `load_forge_config().project == "so101-chess"` when `WANDB_PROJECT` is unset. `DEPLOYMENT = "so101-chess"`. `AUTOMATION_NAME = "so101-chess evaluation"`. `GET /improve/automation` → `{"name": AUTOMATION_NAME, "event": AUTOMATION_EVENT, "prompt": aria_prompt()}`; the Improve page renders those three values, so the prompt lives only in `aria.py`.

- [ ] **Step 1: Update the tests**

Replace every `so101-kitting` in `tests/` and `frontend/src/**/*.test.*` with `so101-chess` (project, deployment, artifact paths such as `acme/wandb-registry-model/so101-chess-act:candidate`, ready lines `deployment=so101-chess`). In `tests/test_improve_aria.py`:

```python
def test_the_aria_brief_is_about_chess_transfers() -> None:
    prompt = aria_prompt()
    assert AUTOMATION_NAME == "so101-chess evaluation"
    for word in ("so101-chess-sim-eval", "transfer class", "domain", "approach", "disturbed"):
        assert word in prompt
    assert "skill" not in prompt.lower()


def test_the_automation_route_serves_the_brief() -> None:
    from fastapi.testclient import TestClient

    from lelab.server import app

    body = TestClient(app).get("/improve/automation").json()
    assert body == {"name": AUTOMATION_NAME, "event": AUTOMATION_EVENT, "prompt": aria_prompt()}
```

In `Improve.test.tsx`, mock `/improve/automation` with `{ name: "so101-chess evaluation", event: "A run reaches FINISHED", prompt: "P" }` and assert the three values render.

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest -q tests/test_improve_aria.py tests/test_forge_config.py tests/test_forge_policy_server.py tests/test_forge_policy_host.py && (cd frontend && npm test -- --run src/pages/Improve.test.tsx src/pages/Setup.test.tsx)`
Expected: FAIL on the names.

- [ ] **Step 3: Implement**

`forge_config.py`: `project=_blank_to_none(os.environ.get("WANDB_PROJECT")) or "so101-chess",`. `policy_server.py`: `DEPLOYMENT = "so101-chess"`. `ForgeCredentialsForm.tsx`: `useState("so101-chess")`, `status.project ?? "so101-chess"`, `placeholder="so101-chess"`.

`aria.py`:

```python
AUTOMATION_NAME = "so101-chess evaluation"
AUTOMATION_EVENT = "A run reaches FINISHED"


def aria_prompt() -> str:
    return (
        "Look at the Weave evaluation so101-chess-sim-eval in this project. "
        "Each trial is one transfer of a chess piece, with a transfer class (near, mid, far, bin, park), "
        "a domain (sim or real) and an outcome: success, approach, grasp, drop, place, topple or disturbed. "
        "Name the transfer class and the failure outcome that fail most, separately for sim and real, "
        "and state the gap between sim and real success per transfer class. "
        "Propose the next experiment: which transfer classes need real correction episodes, "
        "whether the sim randomization or the sim cell should change, "
        "and whether to retrain ACT or SmolVLA. "
        "Do not change an alias and do not launch a job."
    )
```

`server.py`:

```python
@app.get("/improve/automation")
def improve_automation():
    from .improve.aria import AUTOMATION_EVENT, AUTOMATION_NAME, aria_prompt

    return {"name": AUTOMATION_NAME, "event": AUTOMATION_EVENT, "prompt": aria_prompt()}
```

`Improve.tsx`: delete the three constants and the "keep in step" comment; after Forge status is `on`, GET `/improve/automation` into state `{ name, event, prompt } | null` and render those values (render nothing for the list while it is `null`).

- [ ] **Step 4: Run everything**

Run: `uv run pytest -q && (cd frontend && npm test -- --run && npx tsc --noEmit -p tsconfig.app.json)`
Expected: pass. `rg -n "so101-kitting" lelab frontend/src tests notebooks` prints only `notebooks/curate_dataset.py` (Task 6).

- [ ] **Step 5: Commit**

```bash
git add lelab/utils/forge_config.py lelab/forge/policy_server.py lelab/improve/aria.py frontend/src/pages/Improve.tsx frontend/src/components/setup/ForgeCredentialsForm.tsx tests/test_improve_aria.py frontend/src/pages/Improve.test.tsx
# plus every test file whose so101-kitting strings you replaced
git add -p lelab/server.py   # only the /improve/automation hunk
git commit -m "refactor: chess names for the W&B project, deployment and ARIA automation"
```

---

### Task 6: Separate sim and real dataset artifacts

**Files:**

- Modify: `lelab/forge/tracking.py` (`log_dataset_if_configured`)
- Modify: `notebooks/curate_dataset.py`
- Modify: `tests/test_forge_tracking.py`, `tests/test_curate_notebook.py`

**Interfaces:**

- Produces: `dataset_artifact_name(source: str | None) -> str` in `lelab/forge/tracking.py`: `"so101-chess-sim"` when `source == "sim"`, else `"so101-chess-real"`. The artifact metadata `source` is `"sim"` or `"real"` (never `None`). The curate notebook downloads `so101-chess-real:latest`.

- [ ] **Step 1: Write the failing tests**

```python
def test_sim_and_real_datasets_are_separate_artifacts() -> None:
    from lelab.forge.tracking import dataset_artifact_name

    assert dataset_artifact_name("sim") == "so101-chess-sim"
    assert dataset_artifact_name(None) == "so101-chess-real"
    assert dataset_artifact_name("real") == "so101-chess-real"
```

In the existing artifact tests of `tests/test_forge_tracking.py`: with `meta/garage_sim.json` `{"source": "sim"}` expect name `so101-chess-sim` and `metadata["source"] == "sim"`; without the file expect `so101-chess-real` and `metadata["source"] == "real"`. In `tests/test_curate_notebook.py`, expect `'use_artifact("so101-chess-real:latest")'` and assert `"kitting" not in text.lower()`.

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_forge_tracking.py tests/test_curate_notebook.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
def dataset_artifact_name(source: str | None) -> str:
    return "so101-chess-sim" if source == "sim" else "so101-chess-real"
```

In `log_dataset_if_configured`: `source = _dataset_source(dataset_dir) or "real"` before `wandb.init`; `wandb.Artifact(dataset_artifact_name(source), type="dataset", metadata={"repo_id": repo_id, "episode_counts": counts, "source": source})`.

Notebook: heading `# Curate the SO-101 chess dataset`; the paragraph adds "Curation is for real recordings; simulated episodes are generated clean."; `download_dataset` docstring `"""Download the latest real chess dataset artifact for an open W&B run."""` and `run.use_artifact("so101-chess-real:latest")`.

- [ ] **Step 4: Run**

Run: `uv run pytest -q`
Expected: pass. `rg -n -i "kitting" lelab frontend/src tests notebooks` prints nothing.

- [ ] **Step 5: Commit**

```bash
git add lelab/forge/tracking.py notebooks/curate_dataset.py tests/test_forge_tracking.py tests/test_curate_notebook.py
git commit -m "feat: log sim and real chess datasets as separate artifacts"
```

---

### Task 7: Docs follow the code

**Files:**

- Modify: `README.md` ("The demo" section, the `WANDB_PROJECT` row, the automation paragraph) — `git add -p`
- Modify: `CHESS_DEMO_GUIDE.md` (section 2 status, `WANDB_PROJECT`, Evaluate/Improve wording) — `git add -p`
- Modify: `docs/superpowers/plans/2026-11-23-garage-chess-m3-real-board.md` (status line)
- Modify: `docs/superpowers/specs/2026-11-13-lelab-forge-so101-design.md` (status line)
- Modify: `docs/superpowers/specs/2026-11-24-garage-chess-m2-code-design.md` and `docs/superpowers/plans/2026-11-24-garage-chess-m2-code.md` (one note each that the kept `so101-kitting` names were renamed by this plan)

- [ ] **Step 1: Edit**

- README "The demo": replace "One SO-101 cell picks parts into a tray. Each part is one skill with a fixed task string, for example `pick the bracket and place it in the tray`." with "One SO-101 cell plays chess. Every robot move is one or more transfers (a piece from one square to another, into the bin, or from a parking pad), all with one task string: `move the piece on the red square to the green square`; red and green discs on the `top` camera show which transfer." Rewrite steps 2–6 for chess: Generate in simulation, Train (ACT or SmolVLA, one collection each), Evaluate in simulation and on the arm with the 20-transfer plan, Record corrections in chess transfer mode for the weakest transfer classes, Promote. Default project row: `so101-chess`. Automation: `so101-chess evaluation`, prompt from `GET /improve/automation`.
- CHESS_DEMO_GUIDE section 2: under the milestone table add a "Status (2026-11-24)" line: M1 done; M2 code done (randomization, Staunton visuals, sim evaluation, app controls), M2 runs open (multi-dataset training, Sandbox sweeps, `sim-best`); M3–M7 not started. Real Evaluate: "records the operator's outcome per transfer of the plan".
- M3 plan status: replace "Tasks 1–3 are complete and depend only on M1" with "Tasks 1–3 are fully specified and depend only on M1; none is implemented yet (checked 2026-11-24)". Add to its Task 4 outline: "The real evaluation session and Weave rows already exist (cleanup plan Task 3); Task 4 adds the overlay to the rollout and runs `sim-best`."
- Old kitting spec status: "Superseded on `forge/chess` by the chess spec (2026-11-23). Kitting lives on the `forge/m*` branches. Sections 3, 4.1, 4.3, 4.4, 7 and 12 still apply."

- [ ] **Step 2: Check**

Run: `rg -n -i "tray|bolt|bracket|kitting" README.md CHESS_DEMO_GUIDE.md`
Expected: only the historical mentions (the spec link and "kitting demo" as history), no instructions.

- [ ] **Step 3: Commit**

```bash
git add -p README.md CHESS_DEMO_GUIDE.md
git add docs/superpowers/plans/2026-11-23-garage-chess-m3-real-board.md docs/superpowers/specs/2026-11-13-lelab-forge-so101-design.md docs/superpowers/specs/2026-11-24-garage-chess-m2-code-design.md docs/superpowers/plans/2026-11-24-garage-chess-m2-code.md docs/superpowers/plans/2026-11-24-garage-chess-cleanup.md
git commit -m "docs: chess storyline, status and names after the kitting cleanup"
graphify update .
```

---

## Operator steps (not code, after Task 7)

1. Create the W&B project `so101-chess` (or set `WANDB_PROJECT` to the old project deliberately) and recreate the automation as `so101-chess evaluation` with the prompt from `GET /improve/automation`.
2. Re-log the M1/M2 sim dataset so it appears as `so101-chess-sim` (the earlier `so101-chess` artifact stays as history).
3. Decide where the uncommitted camera-control, lens and colour-calibration work goes (its own branch or commits on `forge/chess`); it is outside this plan.

## Roadmap to the north star (each gets its own plan when it starts)

| Order | Work | Plan |
|---|---|---|
| 1 | This cleanup | this file |
| 2 | M2 runs: two-dataset training with a real-sample ratio, Sandbox generation and sim evaluation, data-size / randomization / ACT vs SmolVLA comparison, ≥ 80 % in sim, `sim-best` alias | new `2026-11-xx-garage-chess-m2-runs.md` |
| 3 | M3: board-state API and Setup board calibration (Tasks 1–3 of the M3 plan), overlay in the rollout, real evaluation of `sim-best`, correction recording with class filter, co-training, `real-best` gate | `2026-11-23-garage-chess-m3-real-board.md` |
| 4 | M4: voice (mlx-whisper, Kokoro), hold-to-talk, planner conversation | to write |
| 5 | M5: player LLM and puzzle evaluation, Stockfish analysis panel, game loop on the sim arm | to write |
| 6 | M6: `ArmBackend` via `rollout_entry`, the real arm plays a move | to write |
| 7 | M7: full real game, Agent Lens per game with PGN, ARIA, storyline, README | to write |
