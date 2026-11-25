# Garage × CoreWeave Forge: SO-101 chess agent with voice — design

- Date: 2026-11-23 (revised the same day: simulation first, Stockfish analysis)
- Status: approved in brainstorming; revision pending review
- Owner: Ivo Baltruschat
- Branch: `forge/chess`, from `chore/uv-install` (which carries all Forge work to date)
- Supersedes: section 6, the Orders page and the kitting skills of [2026-11-13-lelab-forge-so101-design.md](./2026-11-13-lelab-forge-so101-design.md). Sections 3, 4.1, 4.3, 4.4, 7 and 12 of that spec stay in force.

## 1. Goal

A human plays a full game of chess against the SO-101 by voice. The human moves their own pieces by hand and talks to the agent with hold-to-talk; the agent answers aloud, reads the board with the `top` camera, asks a selectable Serverless Inference LLM for its move, and executes that move with a learned policy. The policy is **trained in simulation first** and adapted to the real arm with a small number of real correction episodes. While playing, a **Stockfish analysis panel** shows the evaluation, best lines and the quality of every move. Every Forge product keeps the role it has in the kitting demo.

Success criteria:

1. A game runs to checkmate, stalemate or resignation without leaving the app, except for "Open in Forge" links.
2. Voice round trip (end of speech to start of the spoken reply, arm time excluded) is under 3 s on the demo Mac.
3. The transfer policy reaches ≥ 80 % success on the real transfer plan (section 5.5) with fewer than 100 real episodes in its training data.
4. Every artifact (sim and real datasets, runs, checkpoints, sim and real evaluations, player-model evaluation, game trace, ARIA report) is visible in Forge with lineage.
5. With Forge disabled, the fork behaves like upstream leLab. With Stockfish missing, everything but the analysis panel works.

Out of scope: German or any language other than English, hands-free microphone, wake words, electronic sensor boards, clocks and time controls, more than one board, the agent tuning its own prompts, Stockfish choosing the robot's moves, Isaac Sim.

Research behind the choices (checked 2026-11-23): the only complete physical SO-101 chess game found is [Dev-devadath/so101-chess](https://github.com/Dev-devadath/so101-chess) (IK travel plus an ACT grasp, 247 real episodes). [RebelDot/so101-chess-physical](https://huggingface.co/datasets/RebelDot/so101-chess-physical) records 200 real "red square to green square" episodes. No SO-101 demo with a sim-trained, square-conditioned policy exists. NVIDIA's [Sim-to-Real SO-101 course](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html) trains a sim-only policy from 75 sim demonstrations, evaluates it in simulation ([module 11](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/11-sim-evaluation.html)), then closes the gap with domain randomization and sim/real co-training; this spec follows that path for chess, in MuJoCo on the Mac instead of Isaac Lab on an NVIDIA workstation. LeRobot publishes an SO-101 MuJoCo scene, [`lerobot/so101-tabletop-mujoco`](https://huggingface.co/lerobot/so101-tabletop-mujoco), with MuJoCo Menagerie's SO-101 (Apache-2.0) in LeRobot's units (degrees, gripper 0–100, 30 fps). Serverless Inference serves no speech model and no `/audio` endpoint, so speech runs on the Mac. Weave renders `wav`/`mp3` logged as `Content`.

Prototype findings (2026-11-23, MuJoCo 4.0.1 on the demo Mac, primitive pieces, 30 mm squares, board starting 10 cm in front of the arm base):

- The scene with 32 pieces compiles in 0.06 s; the Menagerie arm attaches at a mount site unchanged.
- The approach axis of the arm's `gripperframe` site is its local x axis. With that axis vertical, IK reaches every square up to rank 7 with sub-millimetre error; a8 and h8 miss by 19 mm, so far squares need a tilted approach or a smaller board.
- A scripted grasp lifts a pawn 9 cm. Release left it 7 mm off target and lying down, and the elbow swept low over ranks 7–8 and knocked black pieces over. Release and arm clearance are the M1 tuning work (section 5.3).
- Off-screen rendering works on macOS through CGL, outside the Cursor sandbox only.

## 2. Hardware cell

| Item | Detail |
|---|---|
| Arms | SO-101 leader + follower, USB to the Mac (unchanged) |
| Host | Apple Silicon Mac running `lelab` on `:8000`; built-in or USB microphone and speaker |
| Cameras | `top` (whole board, bin and parking squares, nothing occluded when the arm is parked), `wrist` (grasp), `side` (oblique view; judge and planner). All three attached and recorded at one contract: names `top`, `wrist`, `side`, 640×480, 30 fps, `top` at rotation 0 (updated 2026-11-25). Policies read `top` + `wrist` by default; whether `side` helps is a measured experiment after the M3 gate (section 5.5), not an assumption. `top` and `side` are the same camera model, told apart by USB port. |
| Board | Tournament pattern; square size and position chosen in simulation (M1) for full reach and elbow clearance, then built to match. M1 result: 28 mm squares, rank 1 edge 14 cm from the arm base axis (nearer, the arm cannot reach d1/e1 from above); approach tilted up to 35° for far squares. |
| Pieces | Light, similar height, grip width inside the gripper's opening; their measured sizes go into the sim cell file |
| Bin | Open box beside the board for captured pieces |
| Parking | Two marked squares beside the board holding a spare white and a spare black queen |

The real cell is built to match `sim_cell.json` (section 5.2), not the other way round. Camera poses and field of view are measured on the real cell in M3 and written back into the same file.

## 3. Forge product map

| Forge product | Role in the chess demo |
|---|---|
| GPU Sandboxes | Training ACT and SmolVLA on sim and co-trained datasets; large sim evaluations and sim dataset generation (headless EGL) |
| W&B Models | Dataset Artifacts `so101-chess-sim:vN` and `so101-chess-real:vN` with `source` metadata; training runs; checkpoint Artifacts |
| W&B Weave | **Sim evaluations** and real-arm evaluations of the transfer plan; player-model evaluation on chess puzzles |
| CoreWeave Registry | Collections `so101-chess-act` and `so101-chess-smolvla`, aliases `candidate`, `sim-best`, `production` |
| Serverless Inference | Planner (vision, tool calling), the **player** LLM (text, selectable per game), the VLM judge (removed 2026-11-24, see the note in section 6) |
| CoreWeave Agent Lens | One conversation per game, one turn per human utterance or robot move |
| ARIA | Report after each evaluation: transfer class and failure stage that fail most, sim vs real gap |

W&B project default: `so101-chess`. Not used: Serverless SFT/RL, Distillation, Dedicated Inference.

## 4. Architecture

### 4.1 Backend layout

```
lelab/
  chess/
    game.py        # GameState over python-chess: FEN, legal moves, apply, result, move -> transfers
    calibration.py # BoardCalibration: 4 corners, bin, parking -> homography and 67 polygons; load/save JSON
    overlay.py     # draw_goal_markers + GoalOverlayStep (LeRobot ObservationProcessorStep)
    vision.py      # square occupancy and colour vs a reference frame; detect_human_move; verify_squares
    transfers.py   # CHESS_TASK, transfer plan, sidecar (M1); real recording sampler, chess-mode preconditions (M3)
    analysis.py    # Stockfish over UCI: evaluation, best lines, move classification, game review
    player.py      # ChessPlayer: Serverless chat completion -> legal chess.Move, retry policy
    puzzles.py     # Lichess puzzle subset loader for the Weave player evaluation
    rollout_entry.py # lerobot-rollout with GoalOverlayStep in the robot observation pipeline
  sim/
    assets.py      # Menagerie SO-101 from the pinned lerobot/so101-tabletop-mujoco revision
    cell.py        # SimCell: board, bin, parking, cameras, piece sizes; sim_cell.json
    scene.py       # FEN + extras -> MJCF string
    positions.py   # random positions and sampled transfers for generation
    world.py       # ChessWorld: compiled model, stepping in LeRobot units, rendering, camera projection
    judge.py       # transfer outcome and failure class from the physics state
    expert.py      # scripted IK pick-and-place that generates demonstrations
    randomize.py   # domain randomization of light, colours, camera pose, physics
    env.py         # ChessTransferEnv (Gymnasium): one transfer per episode, overlay on `top`
    generate.py    # sim demonstrations -> LeRobotDataset with the real feature names + sidecar
    evaluate.py    # policy rollouts on the transfer plan in sim -> Weave evaluation
    __main__.py    # `python -m lelab.sim gate|preview|generate` (M1), `evaluate` (M2)
  voice/
    audio.py       # webm/opus or wav bytes -> 16 kHz mono float32 (PyAV)
    stt.py         # Transcriber protocol; MlxWhisperTranscriber; FakeTranscriber
    tts.py         # Speaker protocol; KokoroSpeaker -> wav bytes; FakeSpeaker
  agent/
    loop.py        # GameLoop: guardrails per utterance and per robot move
    planner.py     # tool schema for the chess tools; InferencePlanner
    executor.py    # binds tools to chess/, voice/ and a MoveBackend (sim or real arm)
    tracing.py     # Agent Lens conversation per game; audio Content on listen/say
    session.py     # GameSession state machine, speech queue, analysis, HTTP handlers
  board_state.py   # /chess/board-state: PNG -> occupancy
  forge/weave_eval.py    # + sim evaluation and player puzzle evaluation
  utils/forge_config.py  # + player_model, stt_model, tts_voice, stockfish_path
```

Removed on this branch: `lelab/agent/` kitting code (`SKILLS`, `item_counts`, `rollout_budget`, `pick`/`verify_tray`), `collection_name(policy_type, skill)` per-skill logic, the Orders page, `so101-kitting` strings, kitting fixtures in tests and the kitting storyline in `README.md`.

### 4.2 Boundaries

- `chess/` imports python-chess, numpy, OpenCV and (in `overlay.py`) `lerobot.processor`. It does not import hardware code, `sim/`, the planner or Forge.
- `sim/` imports `chess/`, MuJoCo, Gymnasium, numpy and LeRobot's dataset and policy APIs. MuJoCo lives in the optional extra `lelab[sim]`, which the dev group installs. Nothing outside `sim/` imports MuJoCo.
- `voice/` imports PyAV; the real models (`mlx-whisper`, `mlx-audio`) are imported lazily inside the concrete classes, in the optional extra `lelab[voice]` (macOS arm64).
- Stockfish runs as a separate process over UCI through `chess.engine`; it is GPLv3 and is never bundled or linked.
- The agent drives the arm only through a `MoveBackend`: `SimBackend` (in-process MuJoCo) or `ArmBackend` (the existing rollout path and its mutex). Recording, teleoperation, replay, inference and robot moves stay mutually exclusive on the real arm.
- The planner cannot move the arm except through `play_move()`, which executes only the move the player chose and the `GameLoop` validated. Stockfish output never reaches `play_move()`.

## 5. Motion: one overlay-conditioned skill, trained in simulation

### 5.1 The skill

One policy per family. Task string, fixed: `move the piece on the red square to the green square`. `overlay.py` paints a filled red disc centred on the source square and a filled green disc on the target square into the `top` frame, diameter 40 % of the square's shorter side, in simulation, at record time and at inference time. Markers are baked into the recorded video; datasets hold the overlaid `top` stream, the raw `wrist` and `side` streams, and a sidecar `meta/garage_transfers.json` with `episode_index`, `source`, `target` per episode. Sim and real datasets use identical feature names, shapes (640×480) and units, so they can be mixed in one training run.

`GoalOverlayStep` is a LeRobot `ObservationProcessorStep` that edits `observation["top"]` only. The sim env applies it to rendered frames; real recording appends it to the robot observation pipeline; real inference builds LeRobot's rollout context with it in `lelab.chess.rollout_entry`. In simulation the board calibration comes from projecting the known 3D corners through the sim camera; on the real cell from the operator's clicks. The `top` camera must be at rotation 0 on the real cell.

### 5.2 Transfers

`game.py` turns a `chess.Move` into an ordered list of transfers `(source, target)`:

| Move | Transfers |
|---|---|
| Simple | `(from, to)` |
| Capture | `(to, bin)`, `(from, to)` |
| En passant | `(captured_pawn_square, bin)`, `(from, to)` |
| Castling | `(king_from, king_to)`, `(rook_from, rook_to)` |
| Promotion | `(from, bin)`, `(park_<colour>, to)`; a capture adds `(to, bin)` first |

Each transfer is one rollout with a 30 s timeout. The robot's move budget is `3 × len(transfers)` rollouts. A transfer whose target is not occupied afterwards is retried; after the third failure the agent asks the human to finish that transfer by hand, waits for confirmation, and re-reads the board.

### 5.3 Simulation (M1, M2)

**Engine.** MuJoCo, CPU, on the Mac and in Sandboxes (`MUJOCO_GL=egl`). The arm is MuJoCo Menagerie's `robotstudio_so101`, fetched from `lerobot/so101-tabletop-mujoco` at a pinned revision into `~/.cache/huggingface/lerobot/garage/sim_assets/`. Actions and state use LeRobot's `so101_follower` units with `use_degrees=True`: degrees for the five arm joints, 0–100 for the gripper, absolute position targets at 30 fps.

**Cell file.** `sim_cell.json` (under `GARAGE_CONFIG_PATH`) holds the square size, board position relative to the arm base, bin and parking positions and sizes, piece dimensions per type, and pose plus field of view of `top` and `side`. M1 chooses the board size and position in simulation (reach of a8/h8 and the bin, elbow clearance over a full back rank); M3 measures the real cameras and writes them back.

**Scene.** `scene.py` writes MJCF from a FEN plus the contents of bin and parking: a board with an 8×8 checker texture, one free body per piece named `p_<location>_<symbol>`, an open bin, two parking pads, lights, `top` and `side` cameras; the arm's `wrist_cam` comes with the Menagerie model. Pieces start as primitives (a collision cylinder, visual base, body and head sized per type); in M2 the visuals switch to scaled Staunton meshes (MIT, [clarkerubber/Staunton-Pieces](https://github.com/clarkerubber/Staunton-Pieces)) while collisions stay convex primitives. Compile time per reset stays well under 0.1 s, so each episode builds its own scene.

**Judge.** `judge.py` classifies every transfer from the physics state: `success` (moved piece upright, at rest, centre within 35 % of a square of the target centre, or inside the bin footprint; no other piece moved more than 5 mm or tipped), else the first failure in `approach` (gripper never within 1 cm above the source), `grasp` (piece never lifted 2 cm), `drop` (lifted, released away from the target), `place` (off target), `topple` (moved piece lying down), `disturbed` (another piece moved or tipped).

**Expert.** `expert.py` generates demonstrations: damped-least-squares IK on the MuJoCo model for the `gripperframe` site, approach axis tilted outward by up to 35° for far squares, waypoints hover → descend → close → lift → travel at clearance height → descend → open → retreat → home, smoothed into 30 fps joint targets. The M1 exit gate is ≥ 90 % `success` over the 20-transfer plan × 5 seeds with the default cell. Tuning knobs: tilt, hover and travel height, grasp height fraction, release height, opening speed, collision radius, friction, piece mass.

**Domain randomization** (`randomize.py`, M2): light direction and intensity, board and piece colours, table texture, `top`/`side` camera position ±1 cm and angle ±2°, field of view ±2°, piece mass and friction ±30 %, actuator gain ±15 %, initial joints ±2°. Off for the M1 expert gate, on by default for dataset generation from M2.

**Positions.** Each sim episode starts from a position reached by 0–60 random legal plies, so occupancy and clutter vary like real games. Transfers are sampled against that position: 70 % board-to-board (source occupied, target empty, source ranks cycling near/mid/far), 20 % board-to-bin, 10 % parking-to-board (a queen of that colour placed on the pad).

**Dataset.** `generate.py` writes a LeRobotDataset (`robot_type="so101_follower"`, features built with LeRobot's `hw_to_dataset_features` from the real follower's feature shapes), the transfer sidecar, and `meta/garage_sim.json` with the seed, cell, expert parameters, randomization and asset revision; the W&B Artifact gets `source=sim`. Expert episodes the judge does not mark `success` are dropped. Starting size: 1,000 episodes; M2 compares 250 / 1,000 / 4,000.

### 5.4 Sim evaluation (M2)

The equivalent of NVIDIA's sim evaluation module, for chess. `evaluate.py` loads a checkpoint (local path or Registry alias), runs it in `ChessTransferEnv` through LeRobot's `predict_action` with the checkpoint's pre/post processors, and plays the **transfer plan** (section 5.5) × N seeds with randomization on. Per trial it logs the judge's outcome and failure class, duration, and a `top` video. Results go to a Weave Evaluation (`so101-chess-sim-eval`) with a summary by transfer class and failure class. Runs from **Evaluate → In simulation** on the Mac (N = 3) or in a Sandbox (N = 20).

Comparisons it must answer: data quantity (250 / 1,000 / 4,000 episodes), randomization on vs off, ACT vs SmolVLA. The best sim checkpoint gets the Registry alias `sim-best`.

### 5.5 Real transfer and corrections (M3)

**Transfer plan.** Fixed 20 transfers: 16 board-to-board covering the four corners, the centre and near/mid/far rows, 2 board-to-bin, 2 parking-to-board. The same plan runs in sim and on the real arm, so the sim-vs-real gap is one table.

**Go/no-go.** The `sim-best` checkpoint runs the plan on the real arm (judge: section 6 vision for the result, VLM stages for the failure point).

| Real success | Next step |
|---|---|
| ≥ 50 % | Corrections: record 10–30 real episodes per failing transfer class, co-train |
| 20–49 % | Measure the gap first (camera poses, lighting, servo calibration into `sim_cell.json`), regenerate, then corrections |
| < 20 % after one gap pass | Fallback: scripted IK travel to the hover poses (the sim expert's IK, same joint units), learned skill for descend, grasp, lift, place only |

**Corrections.** The Record page's chess transfer mode (section 5.6) records only the failing classes. Training mixes sim and real episodes with real upweighted (default: real sampled at 30 % of each batch). Target: ≥ 80 % real success with < 100 real episodes. Each round is a Weave evaluation, so the curve "real episodes added → real success" is visible in Forge.

**Camera parity (2026-11-25).** The real `top` and `side` poses and fields of view are measured on Setup by clicking the four board corners on each straightened reference; a pinhole fit (the model MuJoCo renders with) is written into `sim_cell.json`. `side` recovers its field of view; `top`, which looks straight down, cannot, so its field of view is fixed. The residual (about 1 cm, 1°) is the width of the domain randomization. `python -m lelab.sim parity` puts the real references next to the sim render. Recording refuses to start unless the cameras match the contract in section 2, and datasets are compared only on the cameras a policy reads.

**Side-camera experiment (after the gate).** Arm A reads `top` + `wrist`, arm B adds `side`, on identical data and at least three training seeds each; sim 20 transfers × 20 seeds per arm, real 20 transfers × 5 repeats per arm. `side` stays in the policy only if sim shows at least +5 points with non-overlapping intervals and real at least +10 points, or it removes a failure class that corrections would otherwise need. The camera set is a column in the Weave tables. Protocol: `docs/superpowers/plans/2026-11-25-garage-chess-cameras.md`.

### 5.6 Real recording

The Record page gains a **Chess transfer** mode: fixed task string, the next transfer shown during reset with the markers live in the `top` tile, sidecar written after each saved episode, dataset Artifact `source=real`. The sampler can be limited to transfer classes (e.g. "far board only", "bin only") for corrections.

## 6. Board reading

`calibration.py`: the empty-board reference frame is the `top` camera reference captured on the Calibration page (`camera_references/<robot>/top.png`) with the board empty under demo lighting. On Setup the operator clicks the four board corners (a1, h1, h8, a8), the four corners of the bin and of each parking square; a homography maps the board corners to 64 square polygons, the other three polygons are stored as clicked. Saved to `~/.cache/huggingface/lerobot/garage/board_calibration.json` with the robot name, image size and thresholds. In simulation `ChessWorld.board_calibration()` produces the same object by projection.

`vision.py` classifies each polygon as empty, white or black against the reference frame (mean absolute difference for occupancy, luminance split for colour). `detect_human_move(game, frame)` lists the legal moves whose resulting occupancy matches the frame. One candidate: the move. Zero or several: the agent asks by voice. The human can always state the move ("knight f3"). `verify_squares` confirms a transfer's result and the whole position after the robot's move. The same functions run on rendered sim frames, so board reading is tested end to end in simulation. The VLM judge scores `top` + `side` on the stage ladder (`started`, `reached`, `grasped`, `delivered`) for real Weave trials; it is not in the game loop's decision path.

Note (2026-11-24): real trials are now judged by the operator in the sim outcome classes (`success`, `approach`, `grasp`, `drop`, `place`, `topple`, `disturbed`), and the VLM judge was removed from `lelab/forge/`.

## 7. The two LLMs

| | Planner | Player |
|---|---|---|
| Model | `FORGE_PLANNER_MODEL`, default Nemotron 3.5 Super VL | `FORGE_PLAYER_MODEL`, chosen per game on Play from the live `/v1/models` list, default DeepSeek V4.1-Flash |
| Input | Transcript, game state, tool results, frames on `look`, the latest analysis summary | FEN, SAN history, legal-move list, colour |
| Output | Tool calls and `say` text | One move in SAN or UCI |
| Guardrail | `GameLoop` (section 8) | python-chess legality; illegal or unparseable → retry quoting the error → second retry → random legal move, spoken apology, `fallback=true` in the trace |

Player-model evaluation (`weave_eval.py`): 100 Lichess puzzles (rating 800 to 1600, bundled CSV under `lelab/chess/data/`). Per model: legal-move rate, best-move rate, **average centipawn loss measured by Stockfish**, median latency. Runs from Evaluate without the arm.

## 8. Stockfish game analysis

`analysis.py` wraps `chess.engine.SimpleEngine.popen_uci` on the binary at `STOCKFISH_PATH` (default: `shutil.which("stockfish")`; install with `brew install stockfish`). One engine process per server, a lock around calls, a time limit per position (`GARAGE_ANALYSIS_TIME`, default 0.3 s) and `MultiPV = 3`.

- **Live analysis.** After every move, the session analyses the new position in a background thread: evaluation from White's point of view (centipawns or mate in N), win probability (Lichess formula), top three lines in SAN, best move.
- **Move classification.** Centipawn loss of the move just played versus the best move from the previous analysis, from the mover's point of view: best (0), excellent (≤ 20), good (≤ 50), inaccuracy (≤ 100), mistake (≤ 300), blunder (> 300); a missed forced mate counts as a blunder.
- **Game review.** At game end: accuracy per side (Lichess accuracy formula), counts per class, evaluation graph, PGN with `[%eval]` comments and class annotations, downloadable from Play and attached to the game trace.
- **Planner access.** A read-only `analysis()` tool returns the latest evaluation and the last move's class, so the agent can comment ("that knight move dropped a pawn"). It never feeds `choose_move()` or `play_move()`.
- **GUI.** On Play: an evaluation bar beside the board diagram, the best-move arrow (toggle), the top three lines, the move list with class badges, the evaluation graph, and the review card after the game. Works for real, sim and digital-only games. `POST /analysis/position` analyses any FEN for the Evaluate page and tests.
- **Missing engine.** `analysis_available=false` with the reason; the panel shows "Install Stockfish (brew install stockfish)" and the game runs unchanged.

## 9. Agent loop

Tools exposed to the planner:

| Tool | Behaviour |
|---|---|
| `look()` | Current `top` (with overlay when a transfer is active), `side`, `wrist` frames |
| `read_board()` | Vision result: detected human move or candidate list, plus the FEN the game expects |
| `confirm_human_move(san)` | Applies the human's move if legal and consistent with the camera; otherwise returns the mismatch |
| `choose_move()` | Calls the player; returns the chosen legal move and whether the fallback fired |
| `play_move()` | Executes the chosen move's transfers on the active `MoveBackend`; returns per-transfer outcome, duration, retries |
| `analysis()` | Latest Stockfish evaluation, best line and last move's class (read-only) |
| `say(text)` | Synthesises speech and queues it for the page |
| `ask_operator(message)` | Shows a prompt, speaks it, blocks until the operator answers (voice or click) |
| `new_game(human_colour, board)` | Resets the game; `board` is `real`, `sim` or `digital` |
| `resign()` | Ends the game for the robot |

Guardrails in code: turn order (the planner cannot `play_move` on the human's turn); only the move from `choose_move` can be played; rollout budget per move `3 × transfers`; 30 s per rollout; arm mutex; at most 8 planner calls per utterance; Stop aborts the rollout, the loop and the speech queue. Prompts are English.

**Board modes.** `real`: camera, vision and the arm. `sim`: the human's move is spoken or clicked on the board diagram and applied in MuJoCo; the robot's move runs the policy in the sim env and the page shows the sim `top` and `side` streams. `digital`: no motion, for voice and player testing.

Tracing: one Agent Lens conversation per game (`agent_name="so101-chess"`), one turn per utterance and per robot move; tool spans as today; `listen` and `say` spans carry the audio as Weave `Content` (`wav`); the PGN with evaluations is attached at game end.

## 10. Voice

- **Capture**: hold-to-talk button on Play (mouse, touch, or Space). `MediaRecorder` (`audio/webm;codecs=opus`) posts the clip to `POST /voice/utterance`. Denied permission hides the button and leaves the text field.
- **STT**: `audio.py` decodes to 16 kHz mono with PyAV. `MlxWhisperTranscriber` runs `mlx-community/whisper-large-v3-turbo` (`GARAGE_STT_MODEL`) with a chess vocabulary prompt (`e4, Nf3, knight, bishop, castle, resign, takes`). Empty transcripts produce "I didn't catch that".
- **TTS**: `KokoroSpeaker` runs Kokoro-82M through `mlx-audio` (`GARAGE_TTS_VOICE`, default `af_heart`), 24 kHz `wav`.
- **Delivery**: speech queue `[{id, text, wav_url}]` in `GET /play/status`; `GET /voice/speech/{id}` serves the wav.
- **Fallback**: text field always available; without the `voice` extra `voice_available=false`.
- **Protocols**: `Transcriber.transcribe(pcm, sample_rate) -> str`, `Speaker.speak(text) -> bytes`, with fakes.

## 11. Pages

Left nav: Setup, Record, Train, Evaluate, **Play**, Improve.

- **Setup**: board calibration (click corners, bin, parking on the `top` reference; check position), sim status (MuJoCo installed, assets cached, cell file, renderer available), Stockfish status, microphone and speaker tests.
- **Record**: chess transfer mode for real corrections; **Generate in simulation** card (episodes, randomization on/off, seed, preview frame) that runs `lelab.sim.generate` as a background job with progress.
- **Train**: dataset picker accepts several datasets (sim + real) with a real-sample ratio; collections renamed.
- **Evaluate**: **In simulation** (checkpoint, seeds, randomization) and **On the arm** (same transfer plan); a comparison table sim vs real per transfer class; **Evaluate player models**.
- **Play**: board diagram with evaluation bar and best-move arrow, analysis lines, move list with class badges, evaluation graph, camera tiles (real or sim), hold-to-talk and text field, dialogue log, step log, player model select, board mode (real / sim / digital), policy family / version / host, human colour, Start, Stop, PGN download, "Open trace in Agent Lens".
- **Improve**: the ARIA text names transfer class, failure class and the sim-vs-real gap.

## 12. Error handling and testing

### 12.1 Failure handling

| Failure | Behaviour |
|---|---|
| MuJoCo missing or no GL context | `sim_available=false` with the reason on Setup; sim cards disabled; real path unaffected |
| Sim assets not downloadable | Clear error naming the repo and revision; cached copy used when present |
| Expert transfer fails in generation | Episode dropped, counted per failure class in `garage_sim.json` |
| Stockfish missing or crashes | `analysis_available=false`; engine restarted once on crash; game unaffected |
| Transcript empty or under 2 characters | "I didn't catch that", no planner call |
| Player illegal three times | Random legal move, spoken apology, `fallback=true` |
| Vision finds zero or several candidate human moves | Spoken question; the human states the move |
| Transfer fails three times | Agent asks the human to finish by hand, waits, re-reads |
| Board does not match the expected position after a move | Agent names the squares that differ, asks for a fix, re-reads |
| Microphone denied or `voice` extra missing | Button hidden, text field only |
| Planner error or invalid tool call | "Let's try that again"; game state untouched; trace kept |
| Catalogue model list unavailable | Player select falls back to the configured default |
| Sandbox, policy server, W&B, judge, secrets | As in the kitting spec, section 8.1 |

### 12.2 Tests

Following leLab's pattern (schemas, pure helpers, idle/mutex branches; subprocess, cloud and model happy paths are not unit-tested):

- `chess/`: transfers for every move type; homography and polygons; vision on synthetic boards; overlay pixels; sampler mix; sidecar.
- `sim/`: scene XML from FEN (body names, counts, bin and parking contents); world compiles with the arm; projection of the board centre lands on the rendered board; `board_calibration()` from projection reads the starting position from a rendered frame with `vision.py`; judge classes on scripted physics states; expert success on three named transfers; generated dataset has the real feature names and a sidecar. Tests that render skip when no GL context can be created; tests that need MuJoCo skip when it is not installed.
- `analysis.py`: classification thresholds and accuracy formula as pure functions; engine tests run only when Stockfish is on `PATH`; a fake engine covers the session wiring.
- `player.py`, `loop.py`, `voice/`, `session.py`: as before (fakes for client, planner, transcriber, speaker, backend).
- Frontend vitest: Play states, analysis panel (bar, lines, badges), Setup calibration steps, Generate-in-sim card.
- Live Forge tests `-m forge`, skipped by default. Manual hardware checklist per milestone.

## 13. Demo storyline

1. Setup checklist green: sim, Stockfish, ports, calibration, cameras, board calibration, microphone, Forge.
2. Generate 1,000 transfer episodes in simulation; dataset appears as an Artifact with `source=sim`.
3. Train ACT and SmolVLA on CoreWeave Sandboxes; evaluate both in simulation; Weave shows data-size and randomization comparisons; promote `sim-best`.
4. Run `sim-best` on the real arm: the sim-vs-real table shows far-row grasps failing.
5. Record 30 far-row correction episodes; co-train; real success rises in Weave; promote to `production`.
6. Evaluate three player models on puzzles (best-move rate and centipawn loss); pick one.
7. Play a game against the sim arm by voice, with the analysis panel live.
8. Play on the real board: a capture, a blunder flagged by Stockfish and commented by the agent; open the trace in Agent Lens; download the PGN.
9. Switch the policy host to CoreWeave and play another move.

## 14. Milestones

| # | Scope | Exit |
|---|---|---|
| M1 | Kitting code removed; `chess/game`, `calibration`, `overlay`, `vision`, `transfers`; `sim/` assets, cell, scene, world, judge, expert, env, generate; cell size chosen in sim | Expert ≥ 90 % on the transfer plan in sim; 1,000-episode sim dataset with overlay and sidecar, logged as an Artifact |
| M2 | Collections renamed; multi-dataset training on Sandboxes; domain randomization; Staunton visuals; `sim/evaluate.py` + Weave; Evaluate → In simulation; Record → Generate in simulation | A sim-trained policy ≥ 80 % on the transfer plan in sim; Weave comparison of data size, randomization, ACT vs SmolVLA |
| M3 | Real cell built to `sim_cell.json`; board-state API, board calibration on Setup, real recording in chess transfer mode; real evaluation; corrections and co-training | Real ≥ 80 % on the transfer plan with < 100 real episodes, or the fallback decided by the gate |
| M4 | `voice/`, hold-to-talk, speech queue, text fallback, planner conversation (no motion) | Talk to the planner, it answers aloud |
| M5 | `player.py`, puzzle evaluation, `analysis.py` + analysis panel, `GameLoop` with `SimBackend` and `digital` mode | A voice game against the sim arm with live Stockfish analysis |
| M6 | `ArmBackend` via `rollout_entry`, real `play_move`, Evaluate on the arm wired into Play | The real arm plays a move from the game |
| M7 | Full real game loop, Agent Lens per game with PGN, ARIA text, storyline run, README | A real game end to end |

## 15. Verify at the start of each milestone

| Milestone | Check |
|---|---|
| M1 | MuJoCo version in the lockfile renders on the demo Mac outside the sandbox; the pinned `lerobot/so101-tabletop-mujoco` revision still contains `robotstudio_so101/`; measured gripper opening and piece sizes |
| M2 | Sandbox image renders headless with `MUJOCO_GL=egl`; LeRobot trains on two datasets with identical features; `predict_action` signature unchanged |
| M3 | Real camera poses and fov measured into `sim_cell.json`; real board, bin and parking built to the sim cell; follower calibration offsets match sim joint zero (home pose overlay check) |
| M4 | `mlx-whisper` and `mlx-audio` install on Python 3.12 on the demo Mac; Kokoro latency for a 10-word sentence; Agent Lens accepts `wav` Content |
| M5 | `stockfish` version on the demo Mac; analysis latency at 0.3 s; live `/v1/models` list; player defaults' plain-text behaviour |
| M6 | Policy server round trip from Europe with overlaid frames; overlay applied before frames leave the Mac |
| M7 | ARIA as an Automations action still fires on the renamed project |

## 16. Upstream sync

As in the kitting spec: new code in new files, upstream edits limited to wiring and theme, rebase on huggingface/leLab regularly, run `pytest` after each rebase and the LeRobot CLI test after any LeRobot bump.
