# Garage Chess — Project Guide

Garage plays chess against you with an SO-101 arm. You speak to the agent, it reads the board through the top camera, asks a selectable LLM from the CoreWeave Forge Serverless catalogue for a move, and moves the pieces with a policy that was **trained in simulation** and adapted to the real arm with a few real correction episodes. While you play, a **Stockfish analysis panel** shows the evaluation, the best lines and the quality of every move. Every dataset, training run, evaluation and conversation shows up in Forge.

This guide is for the person building and running the demo. The design is in [`docs/superpowers/specs/2026-11-23-garage-chess-voice-design.md`](docs/superpowers/specs/2026-11-23-garage-chess-voice-design.md). Implementation plans live in `docs/superpowers/plans/2026-11-23-garage-chess-*.md`, one per milestone (M1 sim and M3 real board are written; the others are written when their milestone starts). Sections marked with a milestone describe features that are not built yet.

## 1. What you are building

| Layer | What it does | Where |
|---|---|---|
| Motion | One learned skill: *move the piece on the red square to the green square*. Red and green discs are painted into the top camera image; the policy moves whatever is under red to green. Captures, castling and promotion are sequences of such transfers. | `lelab/chess/overlay.py`, `transfers.py`, LeRobot ACT / SmolVLA |
| Simulation | MuJoCo model of the cell (Menagerie SO-101, board, pieces, bin, parking, cameras). A scripted IK expert generates demonstrations; policies are evaluated on the same transfer plan as the real arm. | `lelab/sim/` |
| Board reading | Classical vision against an empty-board reference frame: which squares hold a white or black piece; the human's move is the legal move that matches. Runs on sim frames too. | `lelab/chess/calibration.py`, `vision.py` |
| Rules | python-chess: legality, FEN, results, move → transfers | `lelab/chess/game.py` |
| Analysis (M5) | Stockfish over UCI: evaluation bar, best lines, move classes (best … blunder), evaluation graph, game review, PGN with evaluations | `lelab/chess/analysis.py` |
| Player (M5) | A text LLM from Serverless Inference chooses the robot's move. Selectable per game; evaluated on Lichess puzzles in Weave. | `lelab/chess/player.py` |
| Planner (M4–M7) | A vision tool-calling LLM runs the conversation: listens, looks, reads the board, asks the player, plays the move, talks back. Hard guardrails in code. | `lelab/agent/` |
| Voice (M4) | Local speech to text (mlx-whisper) and text to speech (Kokoro via mlx-audio). Hold-to-talk in the browser. | `lelab/voice/` |
| Forge | W&B tracking and Artifacts, Registry aliases, GPU Sandboxes for training and sim evaluation, Weave evaluations, Agent Lens traces, ARIA reports | `lelab/forge/`, `lelab/runners/`, `lelab/improve/` |

Nothing in the game loop can move the arm except `play_move()`, and only with the move the player chose and the rules validated. Stockfish never chooses or plays a move. Recording, teleoperation, replay, evaluation and the game are mutually exclusive on the arm.

## 2. The development path

The path follows NVIDIA's [Sim-to-Real SO-101 course](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html) — sim data, sim training, [sim evaluation](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/11-sim-evaluation.html), randomization, co-training — but for chess, in MuJoCo on the Mac.

| | Scope | Done when |
|---|---|---|
| **M1** | `chess/` game, calibration, overlay, vision, transfers; `sim/` scene, world, judge, expert, env, generate; cell size chosen in sim | Expert ≥ 90 % on the transfer plan in sim; 1,000-episode sim dataset with markers and sidecar, logged as an Artifact |
| M2 | Collections renamed; training on Sandboxes; domain randomization; Staunton visuals; sim evaluation in Weave; Record → Generate in simulation; Evaluate → In simulation | A sim-trained policy ≥ 80 % in sim; Weave compares data size, randomization, ACT vs SmolVLA; `sim-best` promoted |
| M3 | Real cell built to `sim_cell.json`; board calibration on Setup; real recording in chess transfer mode; real evaluation; corrections and co-training | Real ≥ 80 % on the transfer plan with < 100 real episodes, or the fallback chosen by the gate |
| M4 | Voice, hold-to-talk, speech queue, planner conversation (no motion) | You talk to the planner and it answers aloud |
| M5 | Player models and puzzle evaluation; Stockfish analysis panel; game loop against the **sim arm** and in digital mode | A voice game against the sim arm with live analysis |
| M6 | The real arm plays moves from the game (`ArmBackend` via `lerobot-rollout` with the overlay) | The real arm plays a move from the game |
| M7 | Full real game loop, Agent Lens per game with PGN, ARIA, storyline, README | A real game end to end |

Status (2026-11-24): M1 is done. The M2 code is done (randomization, Staunton visuals, sim evaluation, app controls); the M2 runs are open (multi-dataset training, Sandbox sweeps, `sim-best`). M3–M7 are not started. The kitting code was removed from `forge/chess` in the 2026-11-24 cleanup; it stays on the older `forge/*` branches. Real Evaluate records the operator's outcome per transfer of the plan; the rollout runs the active policy with the fixed task string and does not draw the red/green markers yet (M3 Task 4).

Before each milestone, run its "verify at start" check from the spec (section 15): for M1, MuJoCo renders on the demo Mac and the pinned asset revision still exists; for M3, the real camera poses are measured into `sim_cell.json`; for M5, Stockfish is installed and analysis takes about 0.3 s per position.

What prototyping on 2026-11-23 showed, and what M1 therefore tunes: the scene with 32 pieces compiles in 0.06 s; with the gripper vertical the arm reaches ranks 1–7 of a 28–30 mm board placed 10–11 cm in front of the base, a8/h8 miss by about 2 cm (fix: tilt the approach outward for far squares, or move the board); a scripted grasp lifts a pawn cleanly; release toppled the piece and the elbow knocked pieces on ranks 7–8 (fix: lower release height, slower opening, higher travel clearance). M1 settled on a board starting 14 cm from the base with the approach tilted up to 35° and every plan rehearsed in a copy of the scene: the scripted expert passes 97 % of the transfer plan (the remaining failures nudge a neighbour).

## 3. The cell

The real cell is built **to the sim cell**, not the other way round. M1 settles the dimensions in simulation and writes them to `~/.cache/huggingface/lerobot/garage/sim_cell.json`; M3 builds the table to match and measures the real cameras back into the file.

| Item | Requirement |
|---|---|
| Arms | SO-101 leader and follower over USB. The follower sits behind rank 1, centred on the d/e files. |
| Board | Square size and offset from the arm base as in `sim_cell.json`. M1 gate: 28 mm squares (224 mm board), rank-1 edge 14 cm in front of the arm base axis, 4 mm thick, centred on the base. Nearer than 14 cm the arm cannot reach d1/e1 from above. |
| Pieces | Light, similar height, sizes as in `sim_cell.json`; grip width inside the gripper opening with margin. M1 defaults (base radius / height, mm): pawn 8.5/30, knight 9.5/36, bishop 9.5/40, rook 9.5/34, queen 10/46, king 10/50. Measure the real set at M3 and re-run the gate. |
| Bin | Open box beside the board, inside reach. Captured pieces go here. M1: 80 × 70 mm inside, 30 mm walls, centred 30 cm forward and 17 cm to the arm's right (sim y = −0.17). |
| Parking | Two marked pads beside the board, each holding a spare queen (one white, one black) for promotions. M1: 32 mm pads at 17 cm (white) and 23 cm (black) forward, 15 cm to the arm's left (sim y = +0.15). |
| Cameras | `top` (above the board, rotation 0, sees board + bin + parking with the arm parked), `wrist`, `side`. Same names in sim and real. Policies use `top` and `wrist`; on the real arm the operator records each trial's outcome. |
| Host | Apple Silicon Mac. Simulation, Stockfish, STT and TTS run locally on it. |
| Lighting | Fixed. The empty-board reference frame is captured under demo lighting; re-capture it when lighting changes. |

Mark the camera positions on the table. The top camera must not move between calibration, recording and play.

## 4. Install

```bash
uv tool install git+https://github.com/Ivo-B/leLab      # demo machine
lelab                                                    # serves http://localhost:8000
```

From a clone, for development:

```bash
uv sync                                                  # includes the sim extra in the dev group
uv run lelab --dev                                       # UI on :8080, API on :8000
```

Optional pieces:

| Feature | Install | Without it |
|---|---|---|
| Simulation | `uv sync --extra sim` (MuJoCo, Gymnasium). The SO-101 model downloads on first use from `lerobot/so101-tabletop-mujoco` at a pinned revision. | Sim cards disabled; real path unaffected |
| Stockfish analysis (M5) | `brew install stockfish`, or set `STOCKFISH_PATH` | Analysis panel shows "Install Stockfish"; the game runs unchanged |
| Voice (M4) | `uv sync --extra voice` (macOS arm64) | Text field only |

MuJoCo rendering on macOS needs a window-server connection: run `python -m lelab.sim` and rendering tests from a normal terminal (inside the Cursor agent sandbox they fail with "invalid CoreGraphics connection" and skip). Sandboxes render headless with `MUJOCO_GL=egl`.

Connect Forge on **Setup → Connect Forge** or in `.env`:

| Variable | Purpose |
|---|---|
| `WANDB_API_KEY`, `WANDB_ENTITY` | Tracking, Registry, Weave, Serverless Inference, Agent Lens |
| `WANDB_PROJECT` | Default `so101-chess` (M2) |
| `CWSANDBOX_API_KEY`, `CWSANDBOX_TRAIN_IMAGE` | Training and sim evaluation on CoreWeave Sandboxes; the CoreWeave policy host |
| `FORGE_PLANNER_MODEL` | Planner; default Nemotron 3.5 Super VL |
| `FORGE_PLAYER_MODEL` | Default player; default DeepSeek V4.1-Flash, overridable per game on Play (M5) |
| `STOCKFISH_PATH`, `GARAGE_ANALYSIS_TIME` | Stockfish binary (default: `stockfish` on `PATH`) and seconds per position (default 0.3) (M5) |
| `GARAGE_STT_MODEL`, `GARAGE_TTS_VOICE` | Voice models (M4); defaults `mlx-community/whisper-large-v3-turbo`, `af_heart` |

Datasets logged before 2026-11-24 sit under the single Artifact `so101-chess`. New datasets go to `so101-chess-sim` (`source=sim` in `meta/garage_sim.json`) or `so101-chess-real` (everything else).

Secrets stay in the backend. `WANDB_API_KEY` and `CWSANDBOX_API_KEY` never appear in logs or responses. Stockfish is GPLv3 and runs as a separate program; Garage does not bundle it.

## 5. Simulation first (M1–M2)

### Check the cell and the expert (M1)

```bash
uv run python -m lelab.sim preview --transfer e2 e4 --out /tmp/preview.png   # top | side | wrist after reset
uv run python -m lelab.sim gate --seeds 5                                    # expert on the 20-transfer plan
```

`preview` shows the red disc on e2 and the green disc on e4 in the `top` image. `gate` prints success and failure classes per transfer class (near, mid, far, bin, park). The judge's failure classes are `approach` (never got above the piece), `grasp` (never lifted it), `drop` (let go away from the target), `place` (landed off the target), `topple` (piece lying down) and `disturbed` (another piece moved). Tune the expert or the cell until the gate reaches **≥ 90 %**; the M1 plan's Task 11 lists the tuning order (far-square tilt, travel clearance, release height and speed, collision radius and friction, and only then the board position).

### Generate the dataset (M1)

```bash
uv run python -m lelab.sim generate --repo-id local/so101-chess-sim --episodes 1000 --seed 0
```

Each episode starts from a position reached by 0–60 random legal plies and runs one sampled transfer: 70 % board-to-board (source ranks cycle near/mid/far), 20 % board-to-bin, 10 % parking-to-board. Episodes the judge does not mark `success` are dropped and counted. The dataset uses the real follower's feature names (`shoulder_pan.pos` … `gripper.pos`, `observation.images.top|wrist|side`) and the fixed task string, so it mixes with real data. Alongside it: `meta/garage_transfers.json` (source and target per episode) and `meta/garage_sim.json` (seed, cell, expert parameters, asset revision, drop counts). With Forge connected the dataset is logged as an Artifact with `source=sim`.

Open the dataset in the dataset browser and watch a few episodes before training.

### Train and evaluate in simulation (M2)

From M2, **Record → Generate in simulation** runs the same generation as a background job with randomization on (light, colours, camera pose ±1 cm / ±2°, field of view, piece mass and friction, actuator gain, initial joints), and the pieces render as Staunton meshes.

**Train** launches LeRobot training for ACT or SmolVLA on a CoreWeave Sandbox; the checkpoint lands in the Registry collection `so101-chess-act` or `so101-chess-smolvla` as `candidate`.

**Evaluate → In simulation** runs a checkpoint on the 20-transfer plan × N seeds (N = 3 on the Mac, 20 in a Sandbox) with randomization on, using the judge above, and logs a Weave evaluation (`so101-chess-sim-eval`) with a `top` video per trial and a summary by transfer class and failure class. Compare data size (250 / 1,000 / 4,000 episodes), randomization on vs off, and ACT vs SmolVLA. The best checkpoint becomes `sim-best`. Target: ≥ 80 % in sim.

## 6. One-time setup on the real cell (M3)

Work down the **Setup** page in order.

1. **Sim and Stockfish status.** The sim card shows MuJoCo, cached assets, the cell file and the renderer; the Stockfish card shows the engine version (M5).
2. **Calibrate arms and cameras.** Ports, motor ranges, the three cameras. **Camera contract:** the cameras are named `top`, `wrist` and `side`, each set to **640×480 at 30 fps** in the camera form (the sim datasets are rendered that way; Record refuses a chess session that differs, and a dataset at another size cannot be mixed with the sim data). Check that each camera really delivers that size with the same field of view as at 1080p (a cropped picture is not usable for `top` and `side`) and holds 30 fps with all three running; if a camera only offers 16:9, change the contract once in `lelab/chess/transfers.py` (`DATASET_SIZE`) and regenerate the sim dataset. Set the `top` camera's rotation to 0; chess transfer mode refuses any other value because clicked calibration points must match recorded pixels. Put the follower in the home pose and compare with the sim render: the joint zero must match. If the top image is strongly yellow or green, fix the colour before step 3: put a white sheet of paper on the board, run `uv run lelab-camera-calibrate`, drag the box over the sheet, press **Auto calibrate**, then **Save** (see "Camera colour on macOS" in the README). If the board's edges are visibly bowed, straighten the lens too: `uv run lelab-camera-lens --size 640x480`, raise **Field of view** until the table and board edges lie along the grid, **Fit**, **Save** (see "Fisheye lens" in the README), and only then capture the empty-board reference.
3. **Top camera reference.** On the Calibration page, with the board **empty** and the arm parked, capture the `top` reference frame. This image is both the alignment ghost for later sessions and the empty-board baseline for vision.
4. **Board calibration.** On Setup, click 16 points on the reference image in this order: board corners **a1, h1, h8, a8**; the four corners of the **bin**; of **white parking**; of **black parking**. Save. The JSON lands in `~/.cache/huggingface/lerobot/garage/board_calibration.json`.
5. **Check position.** Set up the starting position and press *Check position*. The 8×8 grid should show 16 light and 16 dark pieces in the right places. If not, see Troubleshooting.
6. **Camera positions for the simulation.** On Setup, click the four board corners (a1, h1, h8, a8) on the `side` reference and press **Fit**: the card shows the camera position, its field of view ("fitted") and the pixel error. If the error is at most 3 px press **Write to sim cell**. For `top`, press **Use saved board corners** (it reuses the board calibration clicks), **Fit**, **Write**; a camera looking straight down cannot tell its field of view, so the card keeps the value in `sim_cell.json` ("kept"). The sim's `top` and `side` pictures then match the real ones, and randomization (M2) centres on the real cameras. Click on the straightened reference only; redo the clicks after changing the lens.
   Then compare: `uv run python -m lelab.sim parity --robot <robot>` writes real | sim | blend per camera. Board edges in the blend should lie on top of each other within a few pixels for `top` and `side`; the wrist picture must be the same way up with the gripper jaws in the same place (if it is upside down, set the wrist camera's rotation to 180 and capture its reference again; do not change the sim). Regenerate the sim dataset afterwards, because the cell changed.
7. **Forge.** Save credentials; the Registry card turns green when the collections exist (M2).
8. **Microphone and speaker (M4).** *Test microphone* shows a transcript; *Test speaker* plays a sentence.

## 7. Real transfer: record only what fails (M3)

1. **Run `sim-best` on the real arm** over the same 20-transfer plan (**Evaluate**, 1–5 rounds). After each transfer the operator records the outcome: `success`, or the failure class of the sim judge (`approach`, `grasp`, `drop`, `place`, `topple`, `disturbed`). Each trial is logged into the same Weave evaluation as the sim trials (`so101-chess-sim-eval`, `domain=real`), so one table shows sim vs real per transfer class.
2. **Go/no-go:**

   | Real success | Next step |
   |---|---|
   | ≥ 50 % | Corrections: record 10–30 episodes per failing transfer class, co-train |
   | 20–49 % | Measure the gap first (camera poses, lighting, servo calibration into `sim_cell.json`), regenerate, then corrections |
   | < 20 % after one gap pass | Fallback: the sim expert's IK carries the arm to hover poses; the learned skill does descend, grasp, lift and place only |

3. **Record corrections.** Open **Record**, pick the robot, switch on **Chess transfer mode** and limit it to the failing classes (e.g. "far board only"). The task string is fixed and the Task field locks. During reset the banner shows the next transfer, e.g. `e2 → e4` or `d7 → bin`, and the `top` tile shows the red and green discs. Put a piece on the red square (for `park_white → c8`, make sure the spare queen is parked), then record with the leader arm: approach, grasp, lift, move, place, retreat. Each episode's source and target go to `meta/garage_transfers.json`; the dataset is logged as `so101-chess-real` with `source=real`.
4. **Co-train** on `so101-chess-sim` + `so101-chess-real`, real sampled at 30 % of each batch. Re-run step 1. Repeat until the real plan reaches **≥ 80 %** with fewer than 100 real episodes; promote the checkpoint to `production`.

The curation notebook flags episodes by duration and motion and checks that both markers are visible in the first frame; save a new dataset version from it before training.

## 8. Play (M5–M7)

**Play** shows the board diagram with the Stockfish evaluation bar and best-move arrow, the top three lines, the move list with class badges, the evaluation graph, camera tiles, a hold-to-talk button, a text field, the dialogue log, the step log, and the selectors.

1. Choose the **board mode**: `sim` (your move is spoken or clicked on the diagram and applied in MuJoCo; the robot's move runs the policy in simulation; the tiles show the sim cameras), `digital` (no motion, for voice and player testing) or `real` (camera, vision and the arm, from M6).
2. Choose the player model, policy family, checkpoint alias and host (local Mac or CoreWeave), and your colour.
3. Say *"new game, I play white"*. The agent reads the board, confirms the starting position, and tells you to move.
4. Make your move, then say *"your turn"* (or name the move: *"knight f3"*). The agent reads the board; if the camera is ambiguous it asks.
5. The agent announces its move, executes the transfers, verifies the result, and hands the turn back. Captures go to the bin first; castling moves king then rook; promotion bins the pawn and fetches the parked queen.
6. After every move Stockfish analyses the new position in the background: evaluation (centipawns or mate in N, from White's side), win probability, best lines, and the class of the move just played — best (0 centipawns lost), excellent (≤ 20), good (≤ 50), inaccuracy (≤ 100), mistake (≤ 300), blunder (> 300). The agent can read the analysis and comment ("that knight move dropped a pawn"), but its moves come from the player model only.
7. *"Stop"* or the Stop button aborts the rollout, the loop and the speech queue.
8. After the game: the review card (accuracy per side, counts per class, biggest swings), **Download PGN** with `[%eval]` comments, and *Open trace in Agent Lens* (one turn per utterance and per robot move, with audio clips and the PGN).

**Evaluate player models** (M5) runs the selected catalogue models on Lichess puzzles without the arm: legal-move rate, best-move rate, centipawn loss against Stockfish, latency.

### Improve (M7)

A W&B Automation named `so101-chess evaluation` triggers ARIA when an evaluation finishes; the **Improve** page shows its prompt, which `GET /improve/automation` serves. The report names the transfer class, the failure class and the sim-vs-real gap, and proposes the next experiment: regenerate with other randomization, record targeted real episodes, retrain, re-evaluate, promote.

## 9. Demo storyline

1. Setup checklist green: sim, Stockfish, ports, calibration, cameras, board calibration, microphone, Forge.
2. Generate 1,000 transfer episodes in simulation; the dataset appears as an Artifact with `source=sim`.
3. Train ACT and SmolVLA on CoreWeave Sandboxes; evaluate both in simulation; Weave shows data-size and randomization comparisons; promote `sim-best`.
4. Run `sim-best` on the real arm: the sim-vs-real table shows far-row grasps failing.
5. Record 30 far-row correction episodes; co-train; real success rises in Weave; promote to `production`.
6. Evaluate three player models on puzzles; pick one.
7. Play a game against the sim arm by voice, with the analysis panel live.
8. Play on the real board: a capture, a blunder flagged by Stockfish and commented by the agent; open the trace in Agent Lens; download the PGN.
9. Switch the policy host to CoreWeave and play another move.

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `python -m lelab.sim` fails with "invalid CoreGraphics connection" | No window-server access (agent sandbox, SSH) | Run from a normal terminal on the Mac; in Sandboxes set `MUJOCO_GL=egl` |
| Setup: "MuJoCo not installed" | `sim` extra missing | `uv sync --extra sim` |
| Sim asset download fails | No network or the pinned revision moved | The error names repo and revision; a cached copy in `~/.cache/huggingface/lerobot/garage/sim_assets/` is used when present |
| Gate: far transfers fail at `approach` | a8/h8 out of reach with a vertical gripper | Raise the far-square tilt; if still failing, move the board closer in `sim_cell.json` |
| Gate: failures at `topple` | Release too high or opening too fast | Lower the release height, slow the opening, check piece friction |
| Gate: failures at `disturbed` | Elbow or gripper sweeping neighbours | Raise travel clearance; check collision radii |
| Sim `top` image shows the board mirrored (h1 dark) | Checker texture phase | `sim/scene.py` flips the texture colours; Task 9's test checks a light h1 |
| Analysis panel: "Install Stockfish" | Engine not found | `brew install stockfish` or set `STOCKFISH_PATH`; restart `lelab` |
| Analysis lags behind the game | Long time per position | Lower `GARAGE_ANALYSIS_TIME` (default 0.3 s) |
| Record refuses to start: "needs a camera named `top`" / "rotation 0" | Chess mode requires `top` at rotation 0 | Rename or re-bind the camera on Calibration; set rotation 0 and re-capture the reference |
| Top camera image is strongly yellow | The camera's automatic white balance drops the blue channel; macOS ignores OpenCV's white-balance settings | Calibrate with a white sheet in view (`uv run lelab-camera-calibrate`, or `uv run lelab-camera calibrate <index>`; the tool also sets the camera's brightness), then re-capture the empty-board reference. A power cycle resets the camera; LeLab re-applies the saved white balance and brightness on start |
| Two cameras are both called "USB Camera" and one has no colour correction | Settings are saved per USB port; an older name-only gain is skipped when two cameras share the name | Calibrate each one again in `uv run lelab-camera-calibrate`, picking it by port in the dropdown |
| Top camera image is green, or the empty board reads occupied after a lighting change | The colour gain was measured under other light, or the reference was captured before the gain was set | Calibrate again with a white sheet of paper (`uv run lelab-camera-calibrate`), then re-capture the empty-board reference |
| Board lines bow, or the clicked squares drift off the real squares towards the edge of the picture | The wide lens bends straight lines and a board calibration (four clicked corners) assumes a straight picture | Straighten the lens (`uv run lelab-camera-lens --size 640x480`, then Save), capture the empty-board reference again and redo the board calibration clicks |
| Record refuses to start: "'wrist' delivers 1920x1080; the sim datasets are 640x480" (or a missing size, fps or camera) | The camera form differs from the camera contract | Set width 640, height 480 and fps 30 on each of `top`, `wrist`, `side` on the Calibration page; a quarter-turn rotation swaps the native size, so enter the native one |
| Camera positions: "kept 45°" or similar for the field of view | A view straight down onto the board cannot tell its field of view from four corners | Expected for `top`. For `side` it should read "fitted"; if not, check that the four corners are not nearly in a line |
| Camera positions: the error is above 3 px | Corners clicked out of order (a1, h1, h8, a8 as seen in the picture), or the reference is not straightened | Reset points and click again; recapture the reference after fixing the lens |
| Training: the policy should read only `top` and `wrist` | `side` is recorded for review and experiments | On Train, the **Policy cameras** boxes appear for a chess dataset (top and wrist preselected); the choice is passed as `--policy.input_features` |
| "Calibrate the board on the Setup page" | No `board_calibration.json` | Setup → Board calibration |
| Check position: white pieces read as black on light squares | `colour_threshold` wrong for your lighting | Adjust `colour_threshold` in the calibration JSON (default 128); re-check |
| Check position: empty squares read occupied | Shadows or a moved camera against the reference | Re-capture the empty-board reference; raise `pixel_diff_threshold` (default 40) or `occupied_fraction` (default 0.15) |
| Check position: whole rows off by one | Corners clicked in the wrong order | Reset points, click a1, h1, h8, a8 as seen in the image (a1 is nearest the arm) |
| Markers not visible in the recorded video | Overlay step not attached | Chess mode must be on before the session starts; check the `top` tile shows discs during reset |
| `meta/garage_transfers.json` has fewer entries than episodes | Session ended before a save | Entries are written after each successful save only; re-recorded episodes keep their transfer |
| `sim-best` works in sim, fails everywhere on the real arm | Sim-vs-real gap: camera pose, lighting, joint zero | Section 7 step 2: measure into `sim_cell.json`, regenerate, then corrections |
| Agent asks "which move did you make?" (M5+) | Several legal moves match the camera or none | Say the move; centre pieces on their squares |
| Player picks illegal moves repeatedly (M5) | Weak model or long position | Three strikes → random legal move with `fallback=true` in the trace; choose another model on Play |
| Transfer fails three times (M6) | Grasp or placement out of distribution | Agent asks you to finish the transfer by hand, then re-reads the board; record corrections for that class |
| Hold-to-talk hidden (M4) | Microphone denied or `voice` extra missing | Allow the microphone or install the `voice` extra; the text field always works |

## 11. For developers

Layout under `lelab/`:

```
chess/        game.py calibration.py overlay.py vision.py transfers.py   (M1)
              camera_pose.py (camera contract and pose fit)
              analysis.py player.py puzzles.py (M5)   rollout_entry.py (M6)
sim/          assets.py cell.py scene.py positions.py world.py judge.py
              expert.py env.py generate.py __main__.py (M1)   camera_fit.py parity.py   randomize.py evaluate.py (M2)
voice/        audio.py stt.py tts.py (M4)
agent/        loop.py planner.py executor.py tracing.py session.py (M4–M7)
board_state.py                       # /chess/board-state: PNG -> occupancy (M3)
forge/, runners/, improve/           # Forge integration
```

Rules of the road:

- `chess/` imports python-chess, numpy, OpenCV and (in `overlay.py`) `lerobot.processor` only. No hardware, no sim, no Forge, no planner. Only `sim/` imports MuJoCo and Gymnasium.
- Sim and real share the task string, the markers, the feature names, the sidecar and the transfer plan. Changing any of them breaks mixing sim and real data.
- The overlay is a LeRobot `ObservationProcessorStep`. The sim env and real recording both apply it; the rollout path (M6) builds its context with the same step, so local and CoreWeave policy servers receive the same overlaid frames.
- The sim world produces a `BoardCalibration` by projecting the board into the `top` camera, so `vision.py` and `overlay.py` run unchanged on sim frames.
- Paths come from `lelab/utils/config.py` (`GARAGE_CONFIG_PATH`, `BOARD_CALIBRATION_FILE`, `SIM_CELL_FILE`, `SIM_ASSETS_PATH`, `CAMERA_REFERENCES_PATH`); read them through the module so tests can redirect them.
- Tests: `uv run pytest` (synthetic boards in `tests/chess_render.py`; sim tests skip without MuJoCo, rendering tests skip without a GL context), `npm test` in `frontend/`. Live Forge calls run under `-m forge`, real speech models under `-m voice`; Stockfish tests run only when it is on `PATH`. No unit test starts the arm.
- Branch `forge/chess`; push only to `origin`. Rebase on `huggingface/leLab` regularly and run the tests after each rebase and after any LeRobot bump (`tests/test_lerobot_cli.py`).

API by milestone:

| Route | Purpose |
|---|---|
| `GET /chess/board-calibration` (M3) | Saved calibration (404 when none) |
| `PUT /chess/board-calibration` (M3) | Save a calibration JSON |
| `POST /chess/board-state` (M3) | Body: PNG of a `top` frame → `squares` (64) and `extras` (`bin`, `park_white`, `park_black`) |
| `POST /start-recording` with `chess_transfers: true` (M3) | Chess transfer mode; status gains `chess_transfers` and `transfer` |
| `POST /analysis/position` (M5) | Stockfish analysis of any FEN |
