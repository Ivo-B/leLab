# CoreWeave Policy Host Implementation Plan

**Goal:** When an evaluation is opened with the CoreWeave policy host, LeLab starts a GPU Sandbox running `lerobot-policy-server` with the session's Registry checkpoint, and every rollout during that session runs its policy there instead of on the Mac. Weave logs `coreweave` as the host for those trials.

**Architecture:** `PolicyHost` already creates the Sandbox, publishes a TLS-passthrough endpoint, writes certificates and the server config, and execs the server. This plan adds what is missing around it:

1. The Sandbox downloads the Registry version itself from W&B (decision: no upload from the Mac). It gets `WANDB_API_KEY` and `WANDB_ENTITY` in its environment, the same as training.
2. The server's feature list comes from the checkpoint's `config.json` (already downloaded on the Mac at session start), not from hard-coded `wrist`/`top` cameras.
3. The host reads the server's output until `Policy server ready:` appears. LeRobot logs to stderr and the Sandbox exec stream is stdout, so both commands run as `sh -c '... 2>&1'`.
4. The evaluation session starts the host in a background thread and tracks `starting`, `ready` or `failed`. Closing the session or shutting down the server stops it.
5. `rollout.py` asks the open session how to run. Local session: today's `--policy.path`. CoreWeave session and ready: `--inference.type=remote` with the endpoint. CoreWeave session and starting or failed: refuse with a 409 and say why (decision: no automatic fallback to the Mac).
6. The Evaluate page and status bar show the host state.

**Tech stack:** Python 3.12, FastAPI, `cwsandbox`, LeRobot v0.7.0 remote inference (Zenoh, `lerobot[remote]` extra), React + Vite + vitest.

## Facts this plan relies on (checked in LeRobot v0.7.0)

- Robot side: `lerobot-rollout --inference.type=remote` (`RemoteInferenceConfig` in `lerobot/inference/factory.py`) takes `--inference.endpoint`, `--inference.deployment`, `--inference.semantics`, `--inference.zenoh_config_path`, `--inference.encoding` (`raw`|`jpeg`). `build_remote_rollout_context` rejects `--policy.*` and `--policy.device` when remote.
- Server readiness log line: `Policy server ready: deployment=%s instance=%s; awaiting one client` (`lerobot/remote_inference/server.py`).
- The robot declares image features as `(H, W, 3)` `uint8`, `kind: rgb`, no names; state as `float32` with joint names (`_feature_spec` in `lerobot/rollout/context.py`). A checkpoint's `config.json` `input_features` stores images as `{"type": "VISUAL", "shape": [3, H, W]}` and state as `{"type": "STATE", "shape": [6]}`.
- The Mac venv has no `zenoh` module today. `lerobot[remote]` = `eclipse-zenoh>=1.9.0,<1.11.0`, `msgpack>=1.1.0,<2.0.0`.
- The Sandbox image (`CWSANDBOX_TRAIN_IMAGE`) already contains Python, CUDA, LeRobot, `lelab` and `wandb`. It must also contain `lerobot[remote]` for the policy server. That is an image requirement this plan documents; it cannot be verified from here.

## Global constraints

- Work in `/Users/ibaltruschat/Documents/01_repos/leLab` on branch `forge/garage`. Commit locally after each task. Never push.
- Run `graphify query "<question>"` before exploring code with Read/Grep/Glob.
- Never read or print `.env`. Never call real W&B, CoreWeave or the arm from tests.
- Stage only the task's files. Never commit `graphify-out/` or `frontend/public/brand/`. Do not amend or skip hooks.
- Python tests: `.venv/bin/python -m pytest`. Frontend: `cd frontend && npm test -- --run` and `npx tsc --noEmit -p tsconfig.app.json` (only the two pre-existing errors `src/lib/meshLoaders.ts(64,26)` and `src/pages/Evaluate.tsx(230,22)` are allowed; the second may disappear in Task 5).
- Test first: write the failing test, run it and see it fail for the stated reason, then implement.
- Secrets (`WANDB_API_KEY`, `CWSANDBOX_API_KEY`, HF token, private key bytes) never appear in errors, logs or API responses. Reuse `PolicyHost._redact`.

## Out of scope

- Starting the host from the order agent (`lelab/agent/executor.py`).
- Measuring latency (`chunk_latency_s` stays `None`).
- Building or publishing the Sandbox image.
- A live Sandbox run. The user does one on hardware after Task 6, with the checklist at the end.

---

## Task 1: Robot-side Zenoh dependency

**Files:** `pyproject.toml`, `tests/test_forge_policy_server.py`.

- Add `remote` to the LeRobot extras: `lerobot[core_scripts,feetech,remote,training] @ git+...@v0.7.0`.
- Install into the venv without reinstalling LeRobot: `.venv/bin/python -m pip install "eclipse-zenoh>=1.9.0,<1.11.0" "msgpack>=1.1.0,<2.0.0"` (network needed; request permissions).
- Test: write `zenoh_client_config()` to a tmp file and build it with `lerobot.transport.ZenohConfig(config_file=<path>).build()` (check the constructor fields in `lerobot/transport/zenoh.py`; use whatever field names it actually has). Do the same for `zenoh_server_config()` with the cert paths rewritten to tmp files only if `build()` reads them; otherwise building is enough. RED is `ModuleNotFoundError: zenoh` before the install.

Commit: `build: add LeRobot's remote extra for the CoreWeave policy host`.

## Task 2: Server config from the checkpoint, and the Sandbox downloads it

**Files:** `lelab/forge/policy_server.py`, `tests/test_forge_policy_server.py`, `tests/test_forge_policy_host.py`.

Produces:

- `checkpoint_features(config: dict) -> tuple[list[FeatureSpecDict], FeatureSpecDict]` reading `input_features` and `output_features`:
  - `type == "VISUAL"`, shape `[3, H, W]` → `{"name", "shape": [H, W, 3], "dtype": "uint8", "kind": "rgb", "semantics": "rgb-<camera>-v1"}` where `<camera>` is the part after `observation.images.`.
  - `observation.state` → shape as stored, `float32`, `names: JOINT_NAMES` when the length is 6, `semantics: SEMANTICS`.
  - Action from `output_features["action"]` with `names: JOINT_NAMES` when the length is 6.
  - Missing `input_features`, no image, or no action raises `ValueError` with a plain sentence.
- `policy_server_yaml(model_path: str, features, action) -> str` emits the given features instead of the hard-coded ones. Everything else stays (deployment, semantics, robot type, CUDA, Zenoh config). Update existing tests that call the old signature.
- `PolicyHost.start(artifact_path: str, model_subdir: str, features, action) -> str`:
  - `artifact_path` is the W&B path the Mac downloaded (`<registry path>:<alias>`). Expose it from `lelab/forge/registry.py` as `artifact_path_for(policy_family, skill, alias) -> str` built from `collection_name` and `RegistryStore().resolve`, without downloading.
  - `model_subdir` is `"pretrained_model"` or `""`, matching the Mac's layout of the same artifact.
  - `_create_kwargs` adds `WANDB_API_KEY` and `WANDB_ENTITY` to `environment_variables` when set. It still never adds `CWSANDBOX_API_KEY`.
  - Order after create: certificates, client CA, Zenoh config, then exec the download as `["sh", "-c", "<python download> 2>&1"]` and wait for its `returncode`. Non-zero raises the redacted failure sentence with the last output lines. The download uses `wandb.Api().artifact(path).download(root="/work/policy")`. Quote the path with `shlex.quote`.
  - Then write `/work/server.yaml` with `model_path = "/work/policy/<model_subdir>"` (no trailing slash when empty) and exec the server as `["sh", "-c", "exec lerobot-policy-server --config_path=/work/server.yaml --model.device=cuda 2>&1"]`. Keep `policy_server_argv()` as the source of that argv and join it with `shlex.join`.
  - Return the endpoint. Keep the server process on `self._server_process`.
- `PolicyHost.wait_until_ready(timeout_s: float) -> None` reads `self._server_process.stdout` line by line until a line contains `Policy server ready:`. If the stream ends, the time runs out, or a line matches LeRobot's traceback tail, raise the redacted failure sentence including the last 20 lines (redacted). Stop the Sandbox on failure.
- The failure sentence suffix changes from `" Use the local policy host instead."` to `" Close the evaluation and reopen it with the local policy host."` because there is no automatic fallback. Update both test files.

Tests use the existing `FakeHandle`/`FakeClient`. Extend `FakeHandle.exec` to return a fake process with `stdout` (an iterator of lines) and `returncode`, configurable per call. Cover: features from a VISUAL+STATE config; `ValueError` without images; env carries the W&B key and not the Sandbox key; exec order (download, then server) and both wrapped in `sh -c ... 2>&1`; yaml model path with and without the subdir; download failure is redacted and stops the Sandbox; ready line returns; stream ending without the ready line raises and stops.

Commit: `feat: start the CoreWeave policy server on the session's Registry checkpoint`.

## Task 3: The evaluation session runs the host

**Files:** `lelab/evaluate.py`, `lelab/server.py`, `tests/test_forge_evaluate.py`.

- New `PolicyHostState` on the session: `state: Literal["off", "starting", "ready", "failed"]`, `endpoint: str | None`, `error: str | None`. Local sessions are `off`.
- `handle_start_evaluation`: for `policy_host == "coreweave"`, raise `ValueError("The CoreWeave policy host needs CWSANDBOX_API_KEY and CWSANDBOX_TRAIN_IMAGE")` unless `sandbox_configured(cfg)`. After `resolve_policy_dir`, read `config.json` from it, call `checkpoint_features`, and start a daemon thread that builds the host through a module-level factory `make_policy_host(cfg) -> PolicyHost` (tests replace it), calls `start(...)` then `wait_until_ready(READY_TIMEOUT_S)`, and sets `ready`/`failed`. `READY_TIMEOUT_S = 20 * 60`. The thread only writes state under the session's own lock. `model_subdir` is `"pretrained_model"` when `policy_dir.name == "pretrained_model"`, else `""`.
- `close()` and a new `stop_policy_host()` stop the host quietly (also when the thread is still starting: set a cancel flag; the thread stops the host itself when it finishes `start`).
- `summary()` adds `policy_host_state`, `policy_host_error`; `policy_runs_on` is `"CoreWeave"` for CoreWeave sessions and `"this machine"` for local ones. Never include the endpoint's certificates or secrets; the endpoint host string is fine.
- Weave: delete `ACTUAL_POLICY_HOST`. Rows log `policy_host = self.request.policy_host`, keep `requested_policy_host`. This is honest because Task 4 refuses every CoreWeave-session rollout that would run locally. Replace the test `test_a_coreweave_request_is_logged_as_the_local_host_it_ran_on` with one asserting `coreweave`.
- New `rollout_target() -> RolloutTarget | None` for `rollout.py`: `None` without a session; `("local", policy_dir)` for local sessions; `("remote", endpoint)` when ready; raises `PolicyHostNotReady(message)` when starting (`"The CoreWeave policy host is still starting. Try again in a minute."`) or failed (the stored error). Keep `active_policy_path()` working for existing callers or remove it if `rollout.py` was its only caller.
- `server.py`: the shutdown hook calls `stop_policy_host()` before the Sandbox sweep. `/evaluations` maps the new `ValueError` to 400 like the others.

Tests: CoreWeave start without Sandbox config is 400; with a fake host the state goes `starting` → `ready` (use an event in the fake to control timing; no sleeps longer than needed); a failing fake leaves `failed` with a redacted error; close stops the host; `rollout_target` for each state; `/evaluations/current` shows the new fields.

Commit: `feat: run the evaluation's policy on the CoreWeave host`.

## Task 4: Rollouts use the remote host

**Files:** `lelab/rollout.py`, `tests/test_rollout*.py` (find the existing file with `rg -ln "handle_start_inference|_rollout_inference_args" tests`), `lelab/forge/policy_server.py` (one helper).

- `policy_server.py`: `write_client_zenoh_config() -> str` writes `zenoh_client_config()` to `CLIENT_ZENOH_PATH = "/tmp/lelab-policy-zenoh.json5"` and returns the path.
- `remote_rollout_args(endpoint: str) -> list[str]`: `--inference.type=remote`, `--inference.endpoint=<endpoint>`, `--inference.deployment=so101-kitting`, `--inference.semantics=so101-degrees-gripper-percent-v1`, `--inference.zenoh_config_path=<path>`, `--inference.encoding=jpeg`. Import the constants from `policy_server.py`.
- `handle_start_inference`: replace `_policy_path_for_inference` with a call to `evaluate.rollout_target()`. Remote: omit `--policy.path`, `--policy.device` and `_rollout_inference_args`, add `remote_rollout_args`. Local session: today's behaviour with the session's dir. No session: today's behaviour with the request's ref. `PolicyHostNotReady` returns `{"success": False, "status_code": 409, "message": <message>}` before the slot is claimed, so nothing has to be released.
- Store `"policy_host": "coreweave" | "local"` in `_inference_meta` and return it from `handle_inference_status`.

Tests: argv for remote (no `--policy.` flags, the six inference flags present); argv for a local session unchanged; 409 while starting and while failed, with `inference_active` still `False`. Stub `subprocess.Popen` the way existing rollout tests do.

Commit: `feat: point rollouts at the CoreWeave policy host during an evaluation`.

## Task 5: Evaluate page and status bar show the host

**Files:** `frontend/src/pages/Evaluate.tsx`, `frontend/src/pages/Evaluate.test.tsx`, `frontend/src/components/shell/StatusBar.tsx`, `frontend/src/components/shell/StatusBar.test.tsx`.

- Delete `COREWEAVE_HOST_NOTE`. While a CoreWeave session is open, poll `GET /evaluations/current` every 5 s and show one line under the session heading:
  - starting: `Starting the CoreWeave policy host. This takes a few minutes; rollouts wait until it is ready.`
  - ready: `CoreWeave policy host ready. Rollouts run their policy there.`
  - failed: the `policy_host_error` text, styled as an error.
- Before a session opens and CoreWeave is selected: `The policy runs on a CoreWeave GPU. Rollouts start once the host is ready.`
- Status bar: the Compute slot reads `CoreWeave` for an open CoreWeave session (from `policy_runs_on`), with ` · starting` appended while `policy_host_state === "starting"`. Local stays `Local`.
- Fix the pre-existing `Evaluate.tsx(230,22)` tsc error only if it sits in code this task touches.

Commit: `feat: show the CoreWeave policy host state`.

## Task 6: Docs and final verification

**Files:** `docs/superpowers/specs/2026-11-14-forge-auth-status-design.md` (Weave honesty paragraph), `README.md` only if it documents the Sandbox image requirements (add `lerobot[remote]`).

- Full pytest (needs `required_permissions: ["all"]`), full vitest, tsc, `git status` clean apart from ignored files, `git log origin/main..HEAD`.
- Report the live checklist below to the user.

Commit: `docs: describe the CoreWeave policy host`.

## Live checklist for the user (not automated)

1. The Sandbox image has `lerobot[remote]`: `python -c "import zenoh"` in the image.
2. Open an evaluation with the CoreWeave host. The Evaluate page shows "Starting…", then "ready" within the timeout.
3. Start a rollout from the usual place. Its log shows `Remote rollout admitted: deployment=so101-kitting`.
4. The arm moves; the camera names and resolutions on the robot match the checkpoint's (otherwise the log says `Remote observation features differ`).
5. Close the evaluation. The Sandbox stops (CoreWeave console, tag `lelab`).
