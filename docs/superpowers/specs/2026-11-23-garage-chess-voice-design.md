# Garage × CoreWeave Forge: SO-101 chess agent with voice — design

- Date: 2026-11-23
- Status: approved in brainstorming, pending written-spec review
- Owner: Ivo Baltruschat
- Branch: `forge/chess`, from `chore/uv-install` (which carries all Forge work to date)
- Supersedes: section 6, the Orders page and the kitting skills of [2026-11-13-lelab-forge-so101-design.md](./2026-11-13-lelab-forge-so101-design.md). Sections 3, 4.1, 4.3, 4.4, 7 and 12 of that spec stay in force.

## 1. Goal

A human plays a full game of chess against the SO-101 by voice. The human moves their own pieces by hand and talks to the agent with hold-to-talk; the agent answers aloud, reads the board with the `top` camera, asks a selectable Serverless Inference LLM for its move, and executes that move with a learned policy trained on CoreWeave Sandboxes. Every Forge product keeps the role it has in the kitting demo.

Success criteria:

1. A game runs to checkmate, stalemate or resignation without leaving the app, except for "Open in Forge" links.
2. Voice round trip (end of speech to start of the spoken reply, arm time excluded) is under 3 s on the demo Mac.
3. Every artifact (dataset, run, checkpoint, evaluation, player-model evaluation, game trace, ARIA report) is visible in Forge with lineage.
4. With Forge disabled, the fork behaves like upstream leLab.

Out of scope: German or any language other than English, hands-free microphone, wake words, electronic sensor boards, sim training, clocks and time controls, more than one board, the agent tuning its own prompts.

Research behind the choices (checked 2026-11-23): the only complete physical SO-101 chess game found is [Dev-devadath/so101-chess](https://github.com/Dev-devadath/so101-chess) (IK travel plus an ACT grasp, 247 episodes, overhead-camera square diff). [RebelDot/so101-chess-physical](https://huggingface.co/datasets/RebelDot/so101-chess-physical) records 200 real "red square to green square" episodes. No SO-101 demo with a fully learned, square-conditioned policy exists; this spec takes that route by decision and owns the risk in section 5.4. Serverless Inference serves no speech model and no `/audio` endpoint ([EMEA Inference Overview, 11 Nov 2026](https://docs.google.com/document/d/1UMwTRo7QWgEbL88lWvLsw9X7q-ZSDvIK8IxHKe0mXXk)), so speech runs on the Mac. Weave renders `wav`/`mp3` logged as `Content` ([docs](https://docs.coreweave.com/products/wandb/weave/guides/core-types/media)).

## 2. Hardware cell

| Item | Detail |
|---|---|
| Arms | SO-101 leader + follower, USB to the Mac (unchanged) |
| Host | Apple Silicon Mac running `lelab` on `:8000`; built-in or USB microphone and speaker |
| Cameras | `top` (whole board, bin and parking squares, nothing occluded when the arm is parked), `wrist` (grasp), `side` (judge). All three recorded; policies use `top` + `wrist`; judge and planner also use `side`. |
| Board | Tournament-pattern board sized to the arm's reach, about 25 to 30 cm; measured against the follower's workspace in M1 |
| Pieces | Light, similar height, grip width inside the gripper's opening; measured in M1 |
| Bin | Open box next to the board for captured pieces |
| Parking | Two marked squares next to the board holding a spare white and a spare black queen |

## 3. Forge product map

Same as the kitting spec, with these changes:

| Forge product | Role in the chess demo |
|---|---|
| Serverless Inference | Planner (vision, tool calling), the **player** LLM (text, selectable per game), the VLM judge |
| W&B Weave | Real-arm transfer evaluations; **player-model evaluation** on chess puzzles across catalogue models |
| CoreWeave Registry | Collections `so101-chess-act` and `so101-chess-smolvla`, aliases `candidate` and `production` |
| CoreWeave Agent Lens | One conversation per game, one turn per human utterance or robot move |
| W&B Models | Dataset Artifact `so101-chess:vN`, training runs, checkpoint Artifacts |

W&B project default: `so101-chess`. Not used: Serverless SFT/RL, Distillation, Dedicated Inference.

## 4. Architecture

### 4.1 Backend layout

```
lelab/
  chess/
    game.py        # GameState over python-chess: FEN, legal moves, apply, result, move -> transfers
    calibration.py # BoardCalibration: 4 corners, bin, parking -> homography and 67 polygons; load/save JSON
    vision.py      # square occupancy and colour vs a reference frame; detect_human_move; verify_squares
    overlay.py     # draw_goal_markers(frame, source, target, calibration) -> frame
    player.py      # ChessPlayer: Serverless chat completion -> legal chess.Move, retry policy
    puzzles.py     # Lichess puzzle subset loader for the Weave player evaluation
  voice/
    audio.py       # webm/opus or wav bytes -> 16 kHz mono float32 (PyAV)
    stt.py         # Transcriber protocol; MlxWhisperTranscriber; FakeTranscriber
    tts.py         # Speaker protocol; KokoroSpeaker -> wav bytes; FakeSpeaker
  agent/
    loop.py        # GameLoop: guardrails per utterance and per robot move
    planner.py     # tool schema for the chess tools; InferencePlanner unchanged
    executor.py    # binds tools to chess/, voice/ and the rollout path
    tracing.py     # Agent Lens conversation per game; audio Content on listen/say
    session.py     # GameSession state machine, speech queue, HTTP handlers
  forge/
    weave_eval.py  # + player puzzle evaluation
  utils/forge_config.py  # + player_model, stt_model, tts_voice
```

Removed on this branch: `lelab/agent/loop.py` kitting parsing (`SKILLS`, `item_counts`, `rollout_budget`), the `pick`/`verify_tray` tools, `collection_name(policy_type, skill)` per-skill logic, the Orders page, `so101-kitting` strings, kitting fixtures in tests and the kitting storyline in `README.md`. `notebooks/curate_dataset.py` keeps its checks and gains a marker-visibility check.

### 4.2 Boundaries

- `chess/` imports python-chess, numpy and OpenCV only. It does not import hardware code, the planner or Forge.
- `voice/` imports PyAV; the real models (`mlx-whisper`, `mlx-audio`) are imported lazily inside the concrete classes. Both live in the optional extra `lelab[voice]` (macOS arm64).
- The agent drives the arm only through the existing rollout path and its mutex. Recording, teleoperation, replay, inference and robot moves stay mutually exclusive.
- The planner cannot move the arm except through `play_move()`, which executes only the move the player chose and the `GameLoop` validated.

## 5. Motion: one overlay-conditioned skill

### 5.1 The skill

One policy per family. Task string, fixed: `move the piece on the red square to the green square`. `overlay.py` paints a filled red disc centred on the source square and a filled green disc on the target square into the `top` frame, disc diameter 40 % of the square's shorter side, at record time and at inference time. Markers are baked into the recorded video; the dataset holds the overlaid `top` stream, the raw `wrist` and `side` streams, and per-episode metadata `source`, `target` (square names, `bin`, `park_white`, `park_black`).

### 5.2 Transfers

`game.py` turns a `chess.Move` into an ordered list of transfers `(source, target)`:

| Move | Transfers |
|---|---|
| Simple | `(from, to)` |
| Capture | `(to, bin)`, `(from, to)` |
| En passant | `(captured_pawn_square, bin)`, `(from, to)` |
| Castling | `(king_from, king_to)`, `(rook_from, rook_to)` |
| Promotion | `(from, bin)`, `(park_<colour>, to)`; a capture adds `(to, bin)` first |

Each transfer is one rollout with a 30 s timeout. The robot's move budget is `3 × len(transfers)` rollouts. A transfer whose target square is not occupied afterwards (section 6) is retried; after the third failure the agent asks the human to finish that transfer by hand, waits for confirmation, and re-reads the board.

### 5.3 Recording

The Record page gains a **Chess transfer** mode. It samples the next `(source, target)` pair with a balanced scheme: 70 % board-to-board pairs stratified over near/mid/far rows and all files, 20 % board-to-bin, 10 % parking-to-board. It shows the overlay live, writes the fixed task string, stores `source`/`target` in the episode metadata, and logs the dataset Artifact as before. Starting target: 300 episodes.

### 5.4 Go/no-go gate (M2)

After the first 100 episodes, a short ACT run is trained and tried on 10 unseen square pairs on the real arm. Fewer than 5 of 10 reaching the target square triggers the fallback: scripted travel with LeRobot's `RobotKinematics` (placo IK, already in LeRobot) to a hover pose above the source and target, with the learned skill kept for descend, grasp, lift and place. The fallback changes `executor.py` and the recording scheme only; the rest of this spec stands.

## 6. Board reading

`calibration.py`: the operator clicks the four board corners, the bin and the two parking squares in the `top` view on Setup; a homography maps them to 64 + 3 polygons. The operator captures the empty-board reference frame under demo lighting. Saved to `~/.cache/huggingface/lerobot/garage/board_calibration.json` through `utils/config.py`.

`vision.py` classifies each polygon as empty, white or black by comparing it with the reference frame (mean absolute difference for occupancy, luminance split for colour, thresholds set in calibration and stored with it). `detect_human_move(expected_board, frame)` lists the legal moves whose resulting occupancy matches the frame. One candidate: the move. Zero or several: the agent asks by voice. The human can always state the move ("knight f3"), which the planner passes to `confirm_human_move`. `verify_squares(frame, expected)` confirms a transfer's result and the whole position after the robot's move. The existing Nemotron VL judge scores `top` + `side` on the stage ladder (`started`, `reached`, `grasped`, `delivered`) for Weave trials; it is not in the game loop's decision path.

## 7. The two LLMs

| | Planner | Player |
|---|---|---|
| Model | `FORGE_PLANNER_MODEL`, default Nemotron 3.5 Super VL | `FORGE_PLAYER_MODEL`, chosen per game on Play from the live `/v1/models` list, default DeepSeek V4.1-Flash |
| Input | Transcript, game state, tool results, frames on `look` | FEN, SAN history, legal-move list, colour |
| Output | Tool calls and `say` text | One move in SAN or UCI |
| Guardrail | `GameLoop` (section 8) | python-chess legality; illegal or unparseable → retry quoting the error → second retry → random legal move, spoken apology, `fallback=true` in the trace |

Player-model evaluation (`weave_eval.py`): a fixed set of 100 Lichess puzzles (rating 800 to 1600, bundled as CSV under `lelab/chess/data/`). Per model: legal-move rate, best-move rate, median latency. Runs from Evaluate without the arm. Results are a Weave Evaluation per model with the puzzle id, FEN, expected and returned move per row.

## 8. Agent loop

Tools exposed to the planner:

| Tool | Behaviour |
|---|---|
| `look()` | Current `top` (with overlay when a transfer is active), `side`, `wrist` frames |
| `read_board()` | Vision result: detected human move or candidate list, plus the FEN the game expects |
| `confirm_human_move(san)` | Applies the human's move if legal and consistent with the camera; otherwise returns the mismatch |
| `choose_move()` | Calls the player; returns the chosen legal move and whether the fallback fired |
| `play_move()` | Executes the chosen move's transfers; returns per-transfer stage, duration, retries |
| `say(text)` | Synthesises speech and queues it for the page |
| `ask_operator(message)` | Shows a prompt, speaks it, blocks until the operator answers (voice or click) |
| `new_game(human_colour)` | Resets the game and the board reference check |
| `resign()` | Ends the game for the robot |

Guardrails in code: turn order (the planner cannot `play_move` on the human's turn); only the move from `choose_move` can be played; rollout budget per move `3 × transfers`; 30 s per rollout; arm mutex; at most 8 planner calls per utterance; Stop aborts the rollout, the loop and the speech queue. Prompts are English; the planner is told the human speaks English.

Tracing: one Agent Lens conversation per game (`agent_name="so101-chess"`), one turn per utterance (user message = transcript) and per robot move; tool spans as today; the `listen` and `say` spans carry the audio as Weave `Content` (`wav`). Validated at the start of M3; if Agent Lens drops the audio, the clips are logged on a Weave op beside the trace.

## 9. Voice

- **Capture**: hold-to-talk button on Play (mouse, touch, or Space). The page records with `MediaRecorder` (`audio/webm;codecs=opus`) and posts the clip to `POST /voice/utterance`. The microphone is requested on first use; denied permission hides the button and leaves the text field.
- **STT**: `audio.py` decodes to 16 kHz mono with PyAV. `MlxWhisperTranscriber` runs `mlx-community/whisper-large-v3-turbo` by default (`GARAGE_STT_MODEL`), with an initial prompt listing chess vocabulary (`e4, Nf3, knight, bishop, castle, resign, takes`). Empty transcripts produce the spoken reply "I didn't catch that".
- **TTS**: `KokoroSpeaker` runs Kokoro-82M through `mlx-audio` (`GARAGE_TTS_VOICE`, default `af_heart`), returns 24 kHz `wav` bytes.
- **Delivery**: the session holds a speech queue `[{id, text, wav_url}]` exposed in `GET /play/status`; the page plays entries it has not played yet. `GET /voice/speech/{id}` serves the wav.
- **Fallback**: a text field next to the button sends the same utterance without audio. Without the `voice` extra the backend reports `voice_available=false`, the button is hidden and the text field remains.
- **Protocols**: `Transcriber.transcribe(pcm: np.ndarray, sample_rate: int) -> str` and `Speaker.speak(text: str) -> bytes`. Fakes return fixed strings and a 0.1 s silent wav.

## 10. Pages

Left nav: Setup, Record, Train, Evaluate, **Play**, Improve.

- **Setup**: adds "Board calibration": click corners, bin and parking squares on the live `top` frame, capture the reference, view the detected occupancy grid live, "Test microphone" and "Test speaker".
- **Record**: adds the Chess transfer mode (section 5.3).
- **Train**: unchanged apart from the collection names.
- **Evaluate**: the plan becomes fixed transfer pairs (default 16 board-to-board covering corners and centre, 2 board-to-bin, 2 parking-to-board) plus "Evaluate player models" with a multi-select of catalogue models.
- **Play**: board diagram from the game state, three camera tiles (`top` with overlay), hold-to-talk button and text field, dialogue log (transcripts and replies), step log, player model select, policy family / version / host, human colour, Start game, Stop, "Open trace in Agent Lens".
- **Improve**: the ARIA automation text moves from skill/stage to transfer class (board-to-board near/far, bin, parking) and stage.

Theme and shell follow section 7 of the kitting spec.

## 11. Error handling and testing

### 11.1 Failure handling

| Failure | Behaviour |
|---|---|
| Transcript empty or under 2 characters | Spoken "I didn't catch that", no planner call |
| Player returns illegal or unparseable move three times | Random legal move, spoken apology, `fallback=true` in the trace |
| Vision finds zero or several candidate human moves | Spoken question; the human states the move |
| Transfer fails three times | Agent asks the human to finish by hand, waits, re-reads the board |
| Board does not match the expected position after a move | Agent names the squares that differ, asks the human to fix, re-reads |
| Microphone permission denied or `voice` extra missing | Button hidden, text field only, badge on Play |
| STT or TTS model fails to load | `voice_available=false`, reason on Setup; game continues in text |
| Planner error or invalid tool call | Utterance dropped with a spoken "Let's try that again"; game state untouched; trace kept |
| Catalogue model list unavailable | Player select falls back to the configured default and shows the error |
| Sandbox, policy server, W&B, judge and secrets | As in the kitting spec, section 8.1 |

### 11.2 Tests

Following leLab's pattern (schemas, pure helpers, idle/mutex branches; subprocess, cloud and model happy paths are not unit-tested):

- `game.py`: move to transfers for simple, capture, en passant, castling both sides, promotion with and without capture; result detection.
- `calibration.py`: homography from four corners; polygons do not overlap; load/save round trip.
- `vision.py`: synthetic rendered boards as fixtures; occupancy and colour; `detect_human_move` with one, zero and two candidates.
- `overlay.py`: marker pixels fall inside the right polygon and nowhere else.
- `player.py`: fake client returning illegal, unparseable, then legal; retry cap; fallback flag; SAN and UCI parsing.
- `loop.py`: scripted planner and fake executor: wrong-turn `play_move` refused, budget, timeout, mutex, stop, planner-call cap.
- `voice/audio.py`: fixture webm decodes to the expected length and rate. STT and TTS through the fakes; the real models under `-m voice`, skipped off macOS arm64.
- `session.py`: speech queue ordering, status payload omits audio bytes, handlers refuse while another feature holds the arm.
- Frontend vitest: Play states (idle, human turn, robot turn, question, ended), Setup calibration steps.
- Live Forge tests `-m forge`, skipped by default.
- Manual hardware checklist per milestone.

## 12. Demo storyline

1. Setup checklist green: ports, calibration, cameras, board calibration, microphone, Forge connected.
2. Record 300 transfer episodes; dataset appears as an Artifact.
3. Curate in the Notebook; new dataset version.
4. Train ACT and SmolVLA on CoreWeave Sandboxes; compare in W&B.
5. Evaluate both on the arm; Weave comparison; ARIA report says far-row transfers fail at the grasp stage.
6. Record 60 targeted far-row episodes, retrain, re-evaluate, promote to `production`.
7. Evaluate three player models on puzzles in Weave; pick one.
8. Start a game by voice; play several moves including a capture; open the trace in Agent Lens.
9. Toggle the policy host to CoreWeave and play another move.

## 13. Milestones

| # | Deliverable | Demo at the end |
|---|---|---|
| M1 | Branch cleanup (kitting removed), `chess/game.py`, calibration, vision, overlay, Setup board calibration, Record transfer mode | Human moves detected live; episodes recorded with markers |
| M2 | Collections renamed, training on Sandboxes, Registry linking, go/no-go gate | Trained ACT reaches unseen squares |
| M3 | `voice/`, hold-to-talk, speech queue, text fallback, audio in traces | Talk to the planner, it answers aloud (no arm) |
| M4 | `player.py`, model select, puzzle evaluation in Weave | Voice-only game on the digital board; models compared in Weave |
| M5 | `play_move` transfers, Evaluate transfer plan, judge wiring | Robot plays random transfers; Weave trials |
| M6 | Full game loop, Agent Lens per game, ARIA text, storyline run, README | A game end to end |

## 14. Verify at the start of each milestone

| Milestone | Check |
|---|---|
| M1 | Board size against the follower's reach; piece grip width against the gripper opening; `top` camera sees board, bin and parking with the arm parked |
| M2 | Sandbox image still builds LeRobot with the fixed task string; dataset Artifact carries `source`/`target` metadata |
| M3 | `mlx-whisper` and `mlx-audio` install on Python 3.12 on the demo Mac; Kokoro latency for a 10-word sentence; Agent Lens accepts `wav` Content on tool spans |
| M4 | Live `/v1/models` list; tool-calling and plain-text behaviour of the chosen player defaults; per-request latency |
| M5 | Policy server round trip from Europe with overlaid frames; overlay applied on the client before frames leave the Mac |
| M6 | ARIA as an Automations action still fires on the renamed project |

## 15. Upstream sync

As in the kitting spec: new code in new files, upstream edits limited to wiring and theme, rebase on huggingface/leLab regularly, run `pytest` after each rebase and the LeRobot CLI test after any LeRobot bump.
