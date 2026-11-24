<h1 align="center">Garage</h1>

<p align="center">
  <b>A LeLab demo on CoreWeave.</b>
</p>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>

Garage is a field demo that runs the CoreWeave Forge loop on a real SO-101 arm from one web app. You record demonstrations, train robot policies on CoreWeave, score them on the arm, let a voice agent play chess with the trained skill, and improve the weakest transfers from what failed. Every run, dataset, checkpoint, and evaluation shows up in Forge.

Garage is a fork of [huggingface/leLab](https://github.com/huggingface/leLab), the web interface for [LeRobot](https://github.com/huggingface/lerobot). The name is a nickname for the robot practice room, not a CoreWeave product. The Python package and the `lelab` command keep the LeLab name. Without Forge credentials, the app behaves like upstream LeLab.

## The demo

One SO-101 cell plays chess. Every robot move is one or more transfers (a piece from one square to another, into the bin, or from a parking pad), all with one task string: `move the piece on the red square to the green square`; red and green discs on the `top` camera show which transfer.

1. **Setup:** calibrate the arms and cameras, then connect Weights & Biases and CoreWeave Sandboxes.
2. **Generate:** generate transfer demonstrations in simulation (**Record → Generate in simulation**). Each finished dataset is logged as a W&B Artifact: `so101-chess-sim` for simulation, `so101-chess-real` for recordings on the arm.
3. **Curate:** `notebooks/curate_dataset.py` holds the flagging rules (`flag_episode`) and the artifact download helper, and a person uses those flags to decide which episodes to drop.
4. **Train:** train ACT or SmolVLA on a GPU in CoreWeave Sandboxes from the Train page. Each policy type has one Registry collection, `so101-chess-act` or `so101-chess-smolvla`, and the checkpoint is linked into it with the alias `candidate`.
5. **Evaluate:** evaluate a checkpoint on the 20-transfer plan. In simulation the policy runs the plan and the sim judge classifies each trial. On the arm, the Evaluate page records the operator's outcome for each transfer of the plan (`success`, `approach`, `grasp`, `drop`, `place`, `topple` or `disturbed`); the rollout does not draw the red/green markers yet (M3 Task 4). Each trial is a row in the Weave evaluation `so101-chess-sim-eval`, with `domain` set to `sim` or `real`.
6. **Record corrections and Promote:** record correction episodes in chess transfer mode (M3) for the weakest transfer classes, retrain, and Promote the better policy.
7. **Play:** play a game against the arm by voice.
8. **Repeat on the CoreWeave policy host:** run the same loop with the policy served from CoreWeave.

| Page | Forge products | Status |
|---|---|---|
| Setup | Weights & Biases Models, CoreWeave Registry | Works |
| Record | Weights & Biases Models (dataset Artifacts) | Works |
| Curate | CoreWeave Notebooks | Notebook works; open it from CoreWeave Notebooks |
| Train | CoreWeave Sandboxes, Weights & Biases Models, CoreWeave Registry | Works, locally or on CoreWeave |
| Evaluate | W&B Weave, CoreWeave Sandboxes, CoreWeave Registry | Works; real-arm trials run on this machine and are logged as local, with the requested host kept as `requested_policy_host` |
| Play | Serverless Inference, CoreWeave Agent Lens | Placeholder; the chess agent arrives in M4–M7 |
| Improve | CoreWeave ARIA | Shows the automation to create in W&B |

## Hardware

| Item | Detail |
|---|---|
| Arms | SO-101 leader and follower, over USB |
| Host | Apple Silicon Mac running `lelab` |
| Cameras | `wrist`, `top`, and `side`. All three are recorded. Policies use `wrist` and `top`; the agent also uses `side`. |
| Scene | A chess board and pieces, a bin for captured pieces, and two parking pads for spare queens |

## Getting started

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first. uv brings its own Python 3.12.

### Run the demo

```bash
uv tool install git+https://github.com/Ivo-B/leLab
lelab
```

The install builds the UI, so you don't need Node.js. `lelab` serves the app on http://localhost:8000 and opens it. Stop it with Ctrl+C, or with `lelab --stop` from another terminal.

Update to the latest version with:

```bash
uv tool upgrade lelab
```

### Work on the code

With Node.js 22, from a clone of this repo:

```bash
uv sync
uv run lelab --dev
```

`uv sync` creates `.venv` from `uv.lock` with LeLab in editable mode plus pytest, ruff, and pre-commit. `lelab --dev` installs the frontend dependencies when they are missing and starts the UI on http://localhost:8080 and the API on http://localhost:8000. It opens http://localhost:8080/?api=http://localhost:8000. Stop both with `uv run lelab --stop`.

An editable install does not build the UI bundle. Run `uv run lelab --rebuild` the first time you want `lelab` without `--dev`.

## Connect Forge

Open **Setup → Connect Forge** in the app, or copy `.env.template` to `.env` and fill it in. In a clone, LeLab uses `.env` at the repo root, which is gitignored. A `uv tool` install uses `~/.config/lelab/.env`, so credentials survive upgrades.

| Variable | Needed for |
|---|---|
| `WANDB_API_KEY`, `WANDB_ENTITY` | Everything Forge: tracking, Registry, Weave, Serverless Inference, Agent Lens |
| `WANDB_PROJECT` | Optional. Defaults to `so101-chess` |
| `CWSANDBOX_API_KEY`, `CWSANDBOX_TRAIN_IMAGE` | Training on CoreWeave and the CoreWeave policy host |
| `CWSANDBOX_MAX_LIFETIME_SECONDS` | Optional. Sandbox lifetime, default 14400 |
| `HF_TOKEN` | Optional. Hub downloads and pushes |
| `FORGE_PLANNER_MODEL` | Optional. Override the Serverless Inference planner model |

Datasets logged before 2026-11-24 sit under the single Artifact `so101-chess`. New datasets go to `so101-chess-sim` (generated in simulation) or `so101-chess-real` (recorded on the arm).

You need a Forge organization with GPU Sandboxes allowlisted and ARIA turned on. Secrets stay in the backend and are never sent to the browser.

Every Sandbox gets a `lelab` tag and a maximum lifetime. LeLab stops its Sandboxes on shutdown and stops leftover tagged ones at startup.

## ARIA report

Create a W&B automation named `so101-chess evaluation` on the project Automations tab. Set its event to `A run reaches FINISHED` and its action to Trigger ARIA. The prompt below is the brief ARIA should write in the W&B Report:

> Look at the Weave evaluation so101-chess-sim-eval in this project. Each trial is one transfer of a chess piece, with a transfer class (near, mid, far, bin, park), a domain (sim or real) and an outcome: success, approach, grasp, drop, place, topple or disturbed. Name the transfer class and the failure outcome that fail most, separately for sim and real, and state the gap between sim and real success per transfer class. Propose the next experiment: which transfer classes need real correction episodes, whether the sim randomization or the sim cell should change, and whether to retrain ACT or SmolVLA. Do not change an alias and do not launch a job.

The prompt text lives in `aria_prompt()` in `lelab/improve/aria.py`, and a test keeps this README in sync with it.

## Camera rotation

If a camera is mounted upside down or sideways, select its **Rotation** in the camera setup on the Calibration page: 0°, 90°, 180°, or 270° clockwise. LeLab saves the rotation per camera and applies it to previews, recordings, and inference. A 90° or 270° rotation swaps the output dimensions.

Before inference, select the same rotation the training dataset used. Checkpoint image dimensions do not tell 0° from 180°.

## How it is built

Forge code lives in new modules, and upstream files only carry the wiring and the theme:

- `lelab/forge/`: W&B tracking, CoreWeave Registry, the Sandbox policy server, and the Weave evaluation. Real trials are judged by the operator; `lelab/sim/judge.py` judges trials in simulation.
- `lelab/runners/`: `SandboxJobRunner`, which runs training on CoreWeave Sandboxes.
- `lelab/chess/`: game rules, board calibration, board reading and the goal overlay.
- `lelab/improve/`: episode flagging and the ARIA automation.
- `notebooks/curate_dataset.py`: the curation notebook.

The design is in [docs/superpowers/specs/2026-11-23-garage-chess-voice-design.md](docs/superpowers/specs/2026-11-23-garage-chess-voice-design.md). The earlier kitting design, [docs/superpowers/specs/2026-11-13-lelab-forge-so101-design.md](docs/superpowers/specs/2026-11-13-lelab-forge-so101-design.md), is superseded and kept as history. [CLAUDE.md](CLAUDE.md) covers the upstream architecture.

## Development

Run the tests with `uv run pytest`, the linters with `uv run pre-commit run --all-files`, and the frontend tests with `npm test` in `frontend/`. Live Forge calls are not unit-tested.

`uv.lock` pins every Python dependency, and CI installs with `uv sync --locked`. After changing dependencies in `pyproject.toml`, run `uv lock` and commit `uv.lock`. pip still works with `pip install -e ".[test]"`.

Rebase on `huggingface/leLab` regularly and run `uv run pytest` after each rebase. After a LeRobot bump, `tests/test_lerobot_cli.py` checks the training and rollout commands LeLab builds.

The CoreWeave logo is not in git. Put `cw-dark.png` and `cw-light.png` in `frontend/public/brand/` to use it locally; without them the sidebar shows the letter G. Ask in #help-brand-support before showing the logo outside CoreWeave.

## Credits

LeLab was built by [Team LeLab at the 2025 LeRobot Worldwide Hackathon](https://www.linkedin.com/posts/nicolas-rabault-_lerobot-hackathon-lerobot-ugcPost-7341065019368828930-jTnl/) and is maintained by the [LeRobot](https://huggingface.co/lerobot) team at Hugging Face. Garage builds on it under the Apache 2.0 license.
