# Garage Chess — Project Guide

Garage plays chess against you with an SO-101 arm. You speak to the agent, it reads the board through the top camera, asks a selectable LLM from the CoreWeave Forge Serverless catalogue for a move, and moves the pieces with a policy you trained on CoreWeave. Every dataset, training run, evaluation and conversation shows up in Forge.

This guide is for the person building and running the demo. The design is in [`docs/superpowers/specs/2026-11-23-garage-chess-voice-design.md`](docs/superpowers/specs/2026-11-23-garage-chess-voice-design.md); implementation plans live in `docs/superpowers/plans/2026-11-23-garage-chess-*.md`, one per milestone. Sections marked with a milestone (M2–M6) describe features that are planned but not built yet; the Setup page and this guide are updated as each lands.

## 1. What you are building

| Layer | What it does | Where |
|---|---|---|
| Motion | One learned skill: *move the piece on the red square to the green square*. Red and green discs are painted into the top camera image; the policy moves whatever is under red to green. Captures, castling and promotion are sequences of such transfers. | `lelab/chess/overlay.py`, `transfers.py`, LeRobot ACT / SmolVLA |
| Board reading | Classical vision against an empty-board reference frame: which squares hold a white or black piece. Detects the human's move by matching legal moves to the observed occupancy. | `lelab/chess/calibration.py`, `vision.py` |
| Rules | python-chess: legality, FEN, results, and move → transfers | `lelab/chess/game.py` |
| Player (M4) | A text LLM from Serverless Inference chooses the robot's move. Selectable per game; evaluated on Lichess puzzles in Weave. | `lelab/chess/player.py` |
| Planner (M3–M6) | A vision tool-calling LLM runs the conversation: listens, looks, reads the board, asks the player, plays the move, talks back. Hard guardrails in code. | `lelab/agent/` |
| Voice (M3) | Local speech to text (mlx-whisper) and text to speech (Kokoro via mlx-audio). Hold-to-talk in the browser. | `lelab/voice/` |
| Forge | W&B tracking and Artifacts, Registry aliases, GPU Sandboxes for training, Weave evaluations, Agent Lens traces, ARIA reports | `lelab/forge/`, `lelab/runners/`, `lelab/improve/` |

Nothing in the game loop can move the arm except `play_move()`, and only with the move the player chose and the rules validated. Recording, teleoperation, replay, evaluation and the game are mutually exclusive on the arm.

## 2. The cell

| Item | Requirement |
|---|---|
| Arms | SO-101 leader and follower over USB. The follower sits behind rank 1, centred on the d/e files. |
| Board | Tournament pattern, 25–30 cm edge. Verify (M1 checklist): the gripper reaches a8, h8 and the far edge of the bin with the wrist vertical. Note the edge length you settled on. |
| Pieces | Light, similar height, grip width inside the gripper opening with margin. The king must not touch neighbours when the gripper descends in a crowded centre. |
| Bin | Open box beside the board, inside reach. Captured pieces go here. |
| Parking | Two marked squares beside the board, each holding a spare queen (one white, one black) for promotions. |
| Cameras | `top` (above the board, rotation 0, sees board + bin + parking with the arm parked), `wrist`, `side`. All three are recorded; policies use `top` and `wrist`; the judge uses `top` and `side`. |
| Host | Apple Silicon Mac. STT/TTS run locally on it. |
| Lighting | Fixed. The empty-board reference frame is captured under demo lighting; re-capture it when lighting changes. |

Mark the camera positions on the table. The top camera must not move between calibration, recording and play.

## 3. Install

```bash
uv tool install git+https://github.com/Ivo-B/leLab      # demo machine
lelab                                                    # serves http://localhost:8000
```

From a clone, for development:

```bash
uv sync
uv run lelab --dev                                       # UI on :8080, API on :8000
```

Voice (M3) is an optional extra on macOS arm64: `uv tool install "lelab[voice] @ git+https://github.com/Ivo-B/leLab"` or `uv sync --extra voice`. Without it the Play page works in text mode.

Connect Forge on **Setup → Connect Forge** or in `.env`:

| Variable | Purpose |
|---|---|
| `WANDB_API_KEY`, `WANDB_ENTITY` | Tracking, Registry, Weave, Serverless Inference, Agent Lens |
| `WANDB_PROJECT` | Default `so101-chess` (M2) |
| `CWSANDBOX_API_KEY`, `CWSANDBOX_TRAIN_IMAGE` | Training on CoreWeave Sandboxes and the CoreWeave policy host |
| `FORGE_PLANNER_MODEL` | Planner; default Nemotron 3.5 Super VL |
| `FORGE_PLAYER_MODEL` | Default player; default DeepSeek V4.1-Flash, overridable per game on Play (M4) |
| `FORGE_JUDGE_MODEL` | VLM judge for Weave trials |
| `GARAGE_STT_MODEL`, `GARAGE_TTS_VOICE` | Voice models (M3); defaults `mlx-community/whisper-large-v3-turbo`, `af_heart` |

Secrets stay in the backend. `WANDB_API_KEY` and `CWSANDBOX_API_KEY` never appear in logs or responses.

## 4. One-time setup

Work down the **Setup** page in order.

1. **Calibrate arms and cameras.** Ports, motor ranges, the three cameras. Set the `top` camera's rotation to 0; chess transfer mode refuses any other value because clicked calibration points must match recorded pixels.
2. **Top camera reference.** On the Calibration page, with the board **empty** and the arm parked, capture the `top` reference frame. This image is both the alignment ghost for later sessions and the empty-board baseline for vision.
3. **Board calibration.** On Setup, click 16 points on the reference image in this order: board corners **a1, h1, h8, a8**; the four corners of the **bin**; the four corners of **white parking**; the four corners of **black parking**. Save. The JSON lands in `~/.cache/huggingface/lerobot/garage/board_calibration.json`.
4. **Check position.** Set up the starting position and press *Check position*. The 8×8 grid should show 16 light and 16 dark squares in the right places. If not, see Troubleshooting.
5. **Forge.** Save credentials; the Registry card turns green when the collections exist (M2).
6. **Microphone and speaker (M3).** *Test microphone* shows a transcript of what you said; *Test speaker* plays a sentence.

## 5. The loop

### Record

Open **Record**, pick the robot, switch on **Chess transfer mode**. The task string is fixed and the Task field locks. For each episode:

- During the reset phase the banner shows the next transfer, e.g. `e2 → e4` or `d7 → bin`, and the top camera tile shows the red and green discs. Put a piece on the red square (any piece; for `park_white → c8` make sure the spare queen is parked).
- Record the episode with the leader arm: approach, grasp, lift, move, place, retreat. Keep the path and the grasp height consistent.
- After the save the next transfer is drawn.

Mix: 70 % board-to-board (source ranks cycle near / mid / far), 20 % board → bin, 10 % parking → board. Target: **300 episodes**. Each episode's `source`/`target` is written to `meta/garage_transfers.json` inside the dataset and travels with the dataset Artifact.

### Curate (Notebook)

The curation notebook flags episodes by duration and motion and (M2) checks that the two markers are visible in the first frame. Save a new dataset version from it.

### Train (M2)

**Train** launches LeRobot training on a CoreWeave Sandbox for ACT or SmolVLA. The run, checkpoints and the dataset Artifact appear in W&B; the final checkpoint is linked into the Registry collection `so101-chess-act` or `so101-chess-smolvla` with the alias `candidate`.

**Go/no-go gate after the first 100 episodes.** Train a short ACT run, try it on 10 unseen square pairs on the arm. Five or more reaching the target: continue recording. Fewer: switch to the fallback — LeRobot's kinematics carry the arm to a hover pose above the source and target, the learned skill does the descend, grasp, lift and place. Only `executor.py` and the recording scheme change.

### Evaluate (M5)

**Evaluate** runs a fixed transfer plan on the arm (16 board-to-board pairs covering corners and centre, 2 to the bin, 2 from parking) for the chosen checkpoint, judges each trial from the `top` and `side` cameras on the stage ladder *started → reached → grasped → delivered*, and logs a Weave evaluation. **Evaluate player models** (M4) runs the selected catalogue models on 100 Lichess puzzles without the arm: legal-move rate, best-move rate, latency.

Promote the better checkpoint to `production` from the Evaluate page.

### Play (M3–M6)

**Play** shows the board diagram, the three camera tiles, a hold-to-talk button, a text field, the dialogue log, the step log, and the model and policy selectors.

1. Choose the player model, policy family, checkpoint alias and host (local Mac or CoreWeave).
2. Say *"new game, I play white"*. The agent reads the board, confirms the starting position, and tells you to move.
3. Make your move on the board, then say *"your turn"* (or name the move: *"knight f3"*). The agent reads the board; if the camera is ambiguous it asks.
4. The agent announces its move, executes the transfers, verifies the result, and hands the turn back. Captures go to the bin first; castling moves king then rook; promotion bins the pawn and fetches the parked queen.
5. *"Stop"* or the Stop button aborts the rollout, the loop and the speech queue.
6. *Open trace in Agent Lens* shows the conversation: one turn per utterance and per robot move, with the audio clips.

### Improve (M6)

A W&B Automation triggers ARIA when an evaluation finishes. The report names the transfer class (near/far board, bin, parking) and stage where trials failed and proposes the next experiment. Record targeted episodes, retrain, re-evaluate, promote.

## 6. Demo storyline

1. Setup checklist green: ports, calibration, cameras, board calibration, microphone, Forge.
2. Record 300 transfer episodes; dataset appears as an Artifact.
3. Curate; new dataset version.
4. Train ACT and SmolVLA on Sandboxes; compare in W&B.
5. Evaluate both on the arm; Weave comparison; ARIA says far-row transfers fail at grasp.
6. Record 60 far-row episodes, retrain, re-evaluate, promote to `production`.
7. Evaluate three player models on puzzles in Weave; pick one.
8. Start a game by voice; play several moves including a capture; open the trace in Agent Lens.
9. Switch the policy host to CoreWeave and play another move.

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Record refuses to start: "needs a camera named `top`" / "rotation 0" | Chess mode requires `top` at rotation 0 | Rename or re-bind the camera on Calibration; set rotation 0 and re-capture the reference |
| "Calibrate the board on the Setup page" | No `board_calibration.json` | Setup → Board calibration |
| Check position: white pieces read as black on light squares | `colour_threshold` too high for your lighting | Lower `colour_threshold` in the calibration JSON (default 128); re-check |
| Check position: empty squares read occupied | Shadows or a moved camera against the reference | Re-capture the empty-board reference; raise `pixel_diff_threshold` (default 40) or `occupied_fraction` (default 0.15) |
| Check position: whole rows off by one | Corners clicked in the wrong order | Reset points, click a1, h1, h8, a8 as seen in the image (a1 is nearest the arm) |
| Markers not visible in the recorded video | Overlay step not attached | Chess mode must be on before the session starts; check the `top` tile shows discs during reset |
| `meta/garage_transfers.json` has fewer entries than episodes | Session ended before a save | Entries are written after each successful save only; re-recorded episodes keep their transfer |
| Agent asks "which move did you make?" (M3+) | Several legal moves match the camera (e.g. promotion piece) or no match (piece off-centre) | Say the move; centre pieces on their squares |
| Player picks illegal moves repeatedly (M4) | Weak model or long position | Three strikes → random legal move with `fallback=true` in the trace; choose another model on Play |
| Transfer fails three times (M5) | Grasp or placement out of distribution | Agent asks you to finish the transfer by hand, then re-reads the board; record more episodes for that transfer class |
| Hold-to-talk hidden (M3) | Microphone denied or `voice` extra missing | Allow the microphone in the browser or install `lelab[voice]`; the text field always works |

## 8. For developers

Layout under `lelab/`:

```
chess/        game.py calibration.py vision.py overlay.py transfers.py
              player.py puzzles.py data/ (M4)   rollout_entry.py (M5)
voice/        audio.py stt.py tts.py (M3)
agent/        loop.py planner.py executor.py tracing.py session.py (M3–M6)
board_state.py                       # /chess/board-state: PNG -> occupancy
forge/, runners/, improve/           # unchanged Forge integration
```

Rules of the road:

- `chess/` imports python-chess, numpy, OpenCV and (in `overlay.py`) `lerobot.processor` only. No hardware, no Forge, no planner.
- The overlay is a LeRobot `ObservationProcessorStep`. Recording appends it to the robot observation pipeline; the rollout path (M5) builds its context with the same step, so local and CoreWeave policy servers receive the same overlaid frames.
- Calibration pixel coordinates refer to the reference image size; `BoardCalibration.scaled_to()` adapts them to another frame size.
- Paths come from `lelab/utils/config.py` (`GARAGE_CONFIG_PATH`, `BOARD_CALIBRATION_FILE`, `CAMERA_REFERENCES_PATH`); read them through the module so tests can redirect them.
- Tests: `uv run pytest` (synthetic boards in `tests/chess_render.py`), `npm test` in `frontend/`. Live Forge calls run under `-m forge`, real speech models under `-m voice`; both are skipped by default. No unit test starts the arm.
- Branch `forge/chess`; push only to `origin`. Rebase on `huggingface/leLab` regularly and run the tests after each rebase and after any LeRobot bump (`tests/test_lerobot_cli.py`).

API added in M1:

| Route | Purpose |
|---|---|
| `GET /chess/board-calibration` | Saved calibration (404 when none) |
| `PUT /chess/board-calibration` | Save a calibration JSON |
| `POST /chess/board-state` | Body: PNG of a `top` frame → `squares` (64) and `extras` (`bin`, `park_white`, `park_black`) |
| `POST /start-recording` with `chess_transfers: true` | Chess transfer mode; status gains `chess_transfers` and `transfer` |

## 9. Milestones

| | Scope | Done when |
|---|---|---|
| M1 | Kitting code removed; `chess/game`, calibration, vision, overlay; Setup board calibration; Record transfer mode | Starting position read live; episodes recorded with markers and sidecar |
| M2 | Collections renamed, training on Sandboxes, Registry, go/no-go gate | Trained ACT reaches unseen squares |
| M3 | `voice/`, hold-to-talk, speech queue, text fallback, audio in traces | You talk to the planner and it answers aloud (no arm) |
| M4 | `player.py`, model select, puzzle evaluation in Weave | Voice-only game on the digital board; models compared |
| M5 | `play_move` transfers, Evaluate transfer plan, judge wiring | Robot executes random transfers; Weave trials |
| M6 | Full game loop, Agent Lens per game, ARIA text, storyline run, README | A game end to end |

Before each milestone, run its "verify at start" check from the spec (section 14): hardware fit for M1, Sandbox image for M2, `mlx-whisper`/`mlx-audio` install and Agent Lens audio for M3, live `/v1/models` for M4, policy-server round trip with overlaid frames for M5, ARIA on the renamed project for M6.
