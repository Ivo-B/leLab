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

## Camera colour on macOS

On macOS, OpenCV cannot change a camera's white balance, exposure or gain: AVFoundation ignores those properties. Some UVC cameras (the Waveshare IMX335 top camera of the SO-101 kit, USB `0abd:8050`) misjudge their automatic white balance under warm indoor light and send an image with almost no blue, so everything looks yellow. A good frame cannot be rebuilt from it afterwards, and a software recolour only amplifies noise.

LeLab sets the controls on the camera itself over USB, with no root and no kernel driver, while the stream is running:

```bash
uv run lelab-camera list                  # cameras and their adjustable controls, with ranges
uv run lelab-camera set 0abd:8050 white_balance_temperature_auto=0 white_balance_temperature=2800 --save
uv run lelab-camera reset 0abd:8050       # back to the camera's defaults, and forget the saved values
```

`--save` stores the values in `~/.cache/huggingface/lerobot/camera_controls.json`. The camera forgets them when it loses power, so LeLab applies the saved values again when the server starts and whenever the camera list is read. `lelab-camera apply` does the same by hand. The endpoints behind it are `GET /camera-controls`, `PUT /camera-controls/{device}` and `POST /camera-controls/apply-saved`.

On the top camera, 2800 K (the lowest the camera allows) turns the yellow cast into a pale green one: blue goes from about 5% to about 50% of green. Hue and saturation cannot remove that tint, so a fixed per-channel gain does it.

The calibration tool does both steps for you. Hold a white sheet of paper under the camera and run:

```bash
uv run lelab-camera-calibrate             # window: live picture, drag a box over the sheet, Auto calibrate, Save
uv run lelab-camera calibrate "USB Camera" # the same without a window; the box is the centre (--roi x,y,w,h)
```

It leaves the camera's automatic exposure on, because in a dim room the automatic mode also raises the sensor gain, which a fixed exposure time cannot (measured on the top camera at 30 fps: a locked 33 ms exposure gave a sheet at 19 of 255, the automatic one 150). A bright sheet on a dark desk still comes out partly clipped, though, so the tool fits the camera's own **brightness** offset: lowered until nothing clips, raised if the sheet is dim, aiming at a green level of about 140 of 255. It then switches the automatic white balance off, tries seven temperatures across the camera's range, keeps the one where the sheet is closest to neutral, checks the brightness again (the white balance changes it) and measures the remaining cast as the gain. It refuses a box that is too dark, still overexposed with the brightness all the way down, or not white, and warns when the box is uneven or dim. `--no-brightness` skips the brightness fit. Nothing is stored until **Save** (or without `--dry-run`); then the temperature, the brightness and the gain are saved. A calibration that fails, is not saved, or ends with `--dry-run` puts the camera back as it was. **Reset camera** restores the camera's defaults and forgets the gain. Cameras without USB controls (the MacBook camera) get the gain only. Close LeLab's own capture of that camera first if the picture does not appear. The automatic exposure follows the light by itself; the brightness offset and the gain do not, so calibrate again when the lighting changes a lot.

**Two identical cameras.** Cameras are told apart by their USB port, not their name, so two cameras that both show up as "USB Camera" each keep their own gain, lens and USB settings. The calibration window lists them with the same id as `lelab-camera list` (`USB Camera (USB 0abd:8050@130000, index 1)`), and the gain is saved under `USB Camera [0x11300000abd8050]`. A camera moved to another USB port counts as a new camera and needs calibrating again. On the command line, name one of two identical cameras by its OpenCV index or by `vid:pid@location` (`lelab-camera list` prints both):

```bash
uv run lelab-camera calibrate 1                      # the camera at OpenCV index 1
uv run lelab-camera gain 1 --blue 1.5 --red 1.3      # saved for that camera only
uv run lelab-camera set 0abd:8050@1141000 hue=-10    # plain 0abd:8050 is refused while two are attached
```

A gain or lens saved under the bare name `USB Camera` (by an older version) is not applied to any camera; save it again for the camera.

**Picking cameras in LeLab.** The Calibration page lists the cameras the same way, with the same label, and each card has a camera select with that label. The picture on the card is the server's picture of that camera: the device the recorder opens, with its saved lens and gain applied, so the card, the reference frame and the recording always show the same camera. The robot profile stores the camera's id (`0x1300000abd8050`, which holds the USB port), so the camera keeps its card when the OpenCV index changes. A camera moved to another port has to be picked again.

The gain can also be set by hand:

```bash
uv run lelab-camera gain "USB Camera" --blue 1.7868 --red 1.4464   # saved; green stays 1.0
uv run lelab-camera gain                                           # show saved gains
uv run lelab-camera gain "USB Camera" --forget
```

A name shared by two attached cameras is refused; name the camera by index or `vid:pid@location` instead. The gain is saved in `~/.cache/huggingface/lerobot/camera_gains.json` and applied to every frame that camera produces: recording, the live view and inference. Each camera has a **Colour correction** switch on its calibration card. Switching it off shows and records that camera's own colours; leaving it unset keeps the correction. The chess reference frame, a snapshot the server takes, follows the same switch. Values are per room and lighting: remeasure when the lights change, and capture the chess reference again afterwards. Frames already recorded keep the gain they were recorded with. The gain is applied on macOS only.

### Fisheye lens

The top camera's wide lens bends straight lines, which breaks what assumes a straight picture: the chess board corners you click, the empty-board reference and the policy's view. LeLab straightens the picture with a saved lens model, the same way it applies the colour gain. Calibrate it by eye:

```bash
uv run lelab-camera-lens --size 640x480   # window: live picture with a grid; --size is the size you record at
uv run lelab-camera lens "USB Camera" --fov 135 --fit 640x480   # or set it by hand
uv run lelab-camera lens                    # show saved lenses
uv run lelab-camera lens "USB Camera" --forget
```

Look at something that is really straight (the table edge, the chess board). Either **drag a line along that edge** on the picture and **Field of view** is set so the edge comes out straight, or raise **Field of view** by hand until it lies along the grid. Drag along an edge off to the side of the picture: a line through the middle stays straight under any lens and cannot be measured. **Curvature** is a fine adjustment for lenses that are not quite equidistant, **Zoom** crops (above 1) or adds black (below 1), and **Fit** picks the widest view with no black borders. **Save** keeps it for that camera (by USB port, like the gain) in `~/.cache/huggingface/lerobot/camera_lens.json`. The top camera of the kit looks straight at about 135°.

Recording, the live view, inference and the camera previews all use it, and each camera has a **Lens correction** switch next to the colour switch (off shows and records the lens as it is). A lens fits pictures of the aspect ratio it was calibrated at, so calibrate at the size you record at, and capture the chess reference again afterwards (the reference frame is straightened too). Frames already recorded keep the geometry they were recorded with. Like the gain, it is applied on macOS only.

On Linux use `v4l2-ctl -d /dev/videoN --set-ctrl white_balance_automatic=0,white_balance_temperature=2800` instead; `lelab-camera` is macOS only. The tests in `tests/test_uvc_live.py` run against a real camera when you set `LELAB_UVC_LIVE=0abd:8050`.

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
