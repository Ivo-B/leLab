# LeLab × CoreWeave Forge: SO-101 order-picking demo — design

- Date: 2026-11-13
- Status: approved in brainstorming, pending written-spec review
- Owner: Ivo Baltruschat
- Repo: personal public fork of [huggingface/leLab](https://github.com/huggingface/leLab) (Apache-2.0)

## 1. Goal

A reusable field demo that runs the full CoreWeave Forge loop on a real SO-101 arm, driven from one web app. A user records demonstrations, trains robot policies on CoreWeave, evaluates them on the real arm, lets an LLM agent fulfil free-text picking orders with the trained skills, and improves the policies from what failed. Every Forge product is used for what it is built for.

Success criteria:

1. The full storyline in section 9 runs end to end on one SO-101 cell without leaving the app, except for "Open in Forge" links.
2. Every artifact (dataset, run, checkpoint, evaluation, agent trace, ARIA report) is visible in Forge with lineage from dataset version to deployed checkpoint.
3. With Forge disabled, the fork behaves like upstream leLab.
4. The UI follows the CoreWeave brand guidelines (section 7).

Out of scope: voice input, multi-arm or bimanual setups, sim training (Isaac twin, ART-Embodied RL; a later spec), the agent feeding lessons into its own prompts automatically.

## 2. Hardware cell

| Item | Detail |
|---|---|
| Arms | SO-101 leader + follower, USB to the Mac |
| Host | Apple Silicon Mac running `lelab` on `:8000` |
| Cameras | `wrist` (gripper), `top`, `side`. All three are recorded. Policies use `wrist` + `top` by default; the judge and agent also use `side`. |
| Scene | Parts area with three visually distinct parts (for example bolt, bracket, gear) plus a few distractors, and one tray |

Skills: one per part, each with a fixed task string, for example `pick the bracket and place it in the tray`. Starting target: 50 episodes per skill.

## 3. Forge product map

| Forge product | Role in the demo |
|---|---|
| Weights & Biases Models | Training metrics, eval videos, dataset and checkpoint Artifacts, Automations |
| CoreWeave Registry | One collection per model (section 5, step 4), aliases `candidate` and `production`, lineage |
| CoreWeave Sandboxes | GPU training jobs; GPU policy server for live execution |
| CoreWeave Notebooks | Dataset curation notebook (marimo) |
| W&B Weave | Real-robot evaluations and checkpoint comparison |
| Serverless Inference | Agent planner and the VLM success judge |
| CoreWeave Agent Lens | One trace per order; Insights over failures |
| CoreWeave ARIA | Reports after each evaluation, next-experiment proposals |

Not used: Serverless SFT/RL and Distillation (LoRA on a fixed LLM catalogue, not for robot policies), Dedicated Inference.

Prerequisites (confirmed): a Forge MT SaaS org, GPU Sandboxes allowlisted, ARIA / Smart features on.

## 4. Architecture

### 4.1 Integration approach

Providers inside leLab. Each concern has a small protocol with the upstream implementation as default and a Forge implementation. LeRobot is not forked. The CoreWeave look is a theme layer.

| Concern | Protocol | Upstream implementation | Forge implementation |
|---|---|---|---|
| Training jobs | existing `JobRunner` in `lelab/jobs.py` | `LocalJobRunner`, `HfCloudJobRunner` | `SandboxJobRunner` |
| Tracking | `Tracker` | Trackio (LeRobot built-in) | W&B (`WandbLogger`) |
| Model store | `ModelStore` | HF Hub | Registry (HF Hub push is kept) |
| Policy host | `PolicyHost` | local `lerobot-rollout` | Sandbox running `lerobot-policy-server` |
| Evaluator | `Evaluator` | none (feature hidden) | Weave + VLM judge |

### 4.2 Backend layout (new files unless noted)

```
lelab/
  providers/            # protocols + selection from forge_config
  forge/
    tracking.py         # W&B run/artifact helpers used by record and train
    sandbox_runner.py   # SandboxJobRunner (cwsandbox SDK), implements JobRunner
    registry.py         # link checkpoints, move aliases, resolve alias -> path
    policy_server.py    # start/stop a GPU Sandbox running lerobot-policy-server
    weave_eval.py       # Weave Evaluation per eval plan, one row per trial
    judge.py            # VLM judge on Serverless Inference, stage-ladder output
    remote/
      train_entry.py    # runs inside the Sandbox: lerobot-train with WandbLogger
      policy_server_entry.py
  agent/
    planner.py          # tool-calling loop against Serverless Inference
    tools.py            # look, pick, verify_tray, ask_operator
    executor.py         # binds pick() to the rollout path
    tracing.py          # Agent Lens instrumentation
  improve/
    automations.py      # eval-finished webhook, ARIA links
  utils/forge_config.py # credentials, entity/project, provider selection
notebooks/
  curate_dataset.py     # marimo notebook, opened in CoreWeave Notebooks
```

Upstream files touched: `server.py` (routes), `jobs.py` (runner selection), `train.py` (target choice), `record.py` (artifact logging hook), `rollout.py` (policy host choice), and the frontend theme and shell. Everything else is additive.

### 4.3 Boundaries

- Forge calls only go through providers. `forge/` does not import hardware code.
- The agent drives the arm only through the existing rollout path and its mutex. Recording, teleoperation, replay, inference and agent picks remain mutually exclusive.
- Code under `forge/remote/` runs in the Sandbox and depends only on LeRobot, W&B and the standard library.

### 4.4 W&B logging without forking LeRobot

LeRobot v0.7 hardwires `TrackioLogger` in `lerobot_train.py`. `train_entry.py` replaces it before calling the training main with a `WandbLogger` exposing the same four methods: `log_dict(metrics, step, mode)`, `log_policy(checkpoint_dir)`, `log_video(path, step, mode)`, `finish()`. A contract test (section 8) fails if LeRobot changes that surface.

## 5. Data flow

1. **Record.** The existing recording flow with the three cameras and a task string from the skill list. On finalize: push to HF Hub (unchanged) and log a W&B Artifact `so101-kitting:vN` (type `dataset`) with per-skill episode counts in metadata.
2. **Curate.** `notebooks/curate_dataset.py` in CoreWeave Notebooks loads the artifact, flags episodes with an idle start over 1 s, dropped frames, or outlier length, and writes a new artifact version. leLab links to it from the Record page.
3. **Train.** Train page: dataset version, policy, target "Run on CoreWeave". `SandboxJobRunner` creates a GPU Sandbox (RTX PRO 6000, explicit max lifetime, `lelab` tag), uploads the run config, runs `train_entry.py`, and streams logs into the existing job monitor. Per dataset version: one SmolVLA conditioned on all three task strings, and one ACT per skill.
4. **Register.** The final checkpoint is logged as a model Artifact and linked with alias `candidate` into its collection: `so101-kitting-smolvla` for SmolVLA, `so101-kitting-act-<skill>` for each ACT. Each collection has its own `production` alias. A setting "policy family" (SmolVLA or ACT) decides which collections evaluation and the agent resolve: SmolVLA uses one checkpoint with the skill's task string; ACT uses the skill's own collection.
5. **Evaluate.** Evaluate page: policy family and version (`candidate`, `production`, or a pinned version), eval plan (default 10 fixed placements per skill), policy host. Per trial: rollout → judge scores `top` + `side` video on the stage ladder (started, reached, grasped, delivered) → operator confirms or overrides with one click → Weave row with videos, stage, verdict source, duration, chunk latency, policy host.
6. **Orders.** The agent resolves each skill through the `production` alias of the selected policy family. One Agent Lens trace per order.
7. **Improve.** Eval finished → W&B Automation → ARIA report on failing skill and stage plus a proposal. Promoting a checkpoint moves `production`.

## 6. Order-picking agent

- **Input:** free-text order on the Orders page, for example "Kit 7: two brackets and one bolt".
- **Planner:** a vision-capable Serverless Inference catalogue model with tool calling, via the OpenAI-compatible API. Default chosen at implementation from the current catalogue; configurable.
- **Loop:** parse order into item counts, then look → pick → verify until the tray matches or limits are hit.

| Tool | Behavior |
|---|---|
| `look()` | Returns current `top` and `side` frames |
| `pick(item)` | Validates `item` against the skill list, runs that skill's `production` policy as one rollout, returns the judge's stage and duration |
| `verify_tray()` | VLM counts parts in the tray from `top`, returns counts per item |
| `ask_operator(message)` | Shows a prompt in the UI and blocks until the operator clicks |

Guardrails in code: known skills only; max 2 retries per item; max rollouts per order (default: 2 × items + 2); 30 s timeout per rollout; `pick` refuses while another feature holds the arm; the UI stop button aborts the rollout and the loop.

Tracing: planner calls, tool calls, judge verdicts, retries and the final tray check as spans, with frames on `look` and `verify` spans. Agent Lens SDK or OpenTelemetry, whichever the docs recommend at implementation.

Orders page: order box, live item checklist, three camera tiles, step log, "Open trace in Agent Lens", policy host toggle.

## 7. UI/UX and CoreWeave CI

### 7.1 Theme

Implemented as CSS variables in `frontend/src/index.css` and `tailwind.config.ts`; shadcn components inherit them.

| Rule | Value |
|---|---|
| Accent | `#0541E9` primary, `#2374FF` bright. Only for primary action, active nav, focus, charts. |
| Structure | CoreWeave 12-step grey ramp |
| Surfaces | Dark default (black, deep blue `#001581`); light theme available |
| Type | Plus Jakarta Sans via `@fontsource`, weights 400/700/800, sentence case |
| Grid | 8 pt spacing |
| Forbidden | Teal/cyan, blue page backgrounds, other typefaces |
| Status | Icon + label always. Success/warning/error hues chosen for WCAG AA and flagged for brand review, since the brand palette has none. |

Sources: [brand guidelines JSON](https://github.com/coreweave/ai-services-pmm-projects/blob/main/creative-content/coreweave-brand-guidelines.json), [DESIGN.md tokens](https://github.com/coreweave/canvas/blob/main/apps/it-engineering/ask-hr-dashboard/DESIGN.md), live guide at live.standards.site/coreweave (wins on conflict).

### 7.2 Shell and pages

- Left nav: Setup, Record, Train, Evaluate, Orders, Improve.
- Status bar: arm connection, cameras, active job, policy host, Forge entity/project.
- "Open in Forge" on every Forge object: run, artifact, Registry version, Weave evaluation, Agent Lens trace, ARIA report.
- Setup: one checklist for ports, calibration, cameras, Forge login.
- Page template: title, one primary action, main area, right details panel.
- Empty states that name the next step.
- WCAG AA contrast, visible focus rings.

### 7.3 Brand assets in a public repo

The theme tokens and the open font are committed. The CoreWeave logo and wordmark are not: the app loads them from a gitignored local assets folder and falls back to a neutral mark. Ask #help-brand-support before external showings. Product names use the official forms (CoreWeave Sandboxes, CoreWeave Registry, Weights & Biases Models, W&B Weave, CoreWeave Agent Lens, CoreWeave ARIA).

## 8. Error handling and testing

### 8.1 Failure handling

| Failure | Behavior |
|---|---|
| Sandbox create fails | Error on the job card with the reason; offer local run; no silent retry |
| Orphaned Sandboxes | Explicit max lifetime and `lelab` tag on every Sandbox; delete on stop and on shutdown; sweep tagged Sandboxes at startup |
| Policy server unreachable mid-rollout | No chunk within the timeout → arm holds position, trial marked failed, UI offers local host |
| W&B unreachable | W&B offline mode, sync later, offline badge in UI |
| Judge error or low confidence | Trial marked "needs review"; operator verdict required; never defaults to success |
| Planner error or invalid tool call | Order aborted, arm idle, trace kept |
| Secrets | `WANDB_API_KEY`, `CWSANDBOX_API_KEY` stay in the backend `.env`; never sent to the frontend or logged |

### 8.2 Tests

Following leLab's pattern (schemas, pure helpers, idle/mutex branches; subprocess and cloud happy paths are not unit-tested):

- Provider selection and Forge config parsing.
- Sandbox training argv parsed by LeRobot's own parser, like `tests/test_lerobot_cli.py`.
- `WandbLogger` contract test against `TrackioLogger`'s method signatures.
- Agent guardrails with a scripted fake planner and fake executor: unknown item, retry cap, rollout cap, mutex refusal, stop.
- Judge response parsing against fixture frames.
- Frontend vitest for shell, nav and theme.
- Live Forge tests marked `-m forge`, skipped by default.
- Manual hardware end-to-end checklist per milestone.

## 9. Demo storyline

1. Setup checklist green; status bar shows Forge connected.
2. Record 50 episodes per skill; dataset appears as an Artifact.
3. Curate in the Notebook; new dataset version.
4. Train SmolVLA and three ACTs on CoreWeave Sandboxes; compare in W&B.
5. Evaluate both on the arm; Weave comparison; ARIA report says the bracket fails at the grasp stage.
6. Record 30 targeted bracket demos, retrain, re-evaluate, promote to `production`.
7. Type an order; the agent fulfils it; open the trace in Agent Lens.
8. Toggle the policy host to CoreWeave and repeat an order.

## 10. Milestones

| # | Deliverable | Demo at the end |
|---|---|---|
| M1 | Fork, theme, app shell, Setup checklist | Rebranded leLab drives the arm as before |
| M2 | W&B tracking for local training, dataset Artifacts, Registry linking | Local training run with lineage in Forge |
| M3 | `SandboxJobRunner` | Training on a GPU Sandbox from the Train page |
| M4 | Judge, Weave evaluation, Evaluate page, Sandbox policy server | Checkpoint comparison from real-arm trials |
| M5 | Order-picking agent with Agent Lens | Free-text order fulfilled and traced |
| M6 | Automations, ARIA, curation Notebook, storyline run, README and W&B Report | Full section 9 storyline |

## 11. Verify at the start of each milestone

| Milestone | Check |
|---|---|
| M3 | Sandbox image for LeRobot + CUDA on RTX PRO 6000; lifetime and volume limits |
| M4 | Sandbox public endpoints carry the gRPC traffic of `lerobot-policy-server`; round-trip latency from Europe |
| M4/M5 | Serverless Inference catalogue models with vision and tool calling; image payload limits |
| M5 | Agent Lens SDK vs OpenTelemetry ingestion |
| M6 | ARIA as an Automations action; Notebooks access to Artifacts |

## 12. Upstream sync

New code in new files; upstream edits limited to wiring and theme. Rebase on huggingface/leLab regularly and run `pytest` after each rebase, including the LeRobot CLI test after any LeRobot bump.
