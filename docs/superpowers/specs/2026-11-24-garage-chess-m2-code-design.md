# Garage chess M2, code slice — design

- Date: 2026-11-24
- Status: approved in brainstorming; pending review of this file
- Owner: Ivo Baltruschat
- Branch: `forge/chess`
- Refines: sections 5.3, 5.4 and the M2 row of [2026-11-23-garage-chess-voice-design.md](./2026-11-23-garage-chess-voice-design.md)
- Follows: M1, which left a 1,000-episode dataset at `~/.cache/huggingface/lerobot/local/so101-chess-sim` (primitive pieces, randomization off, not logged to Forge)

## 1. Goal

Make the simulation trainable and evaluable on this Mac, and file the M1 dataset under a chess collection in Forge. No CoreWeave Sandbox job runs in this slice, and no success-rate target is claimed for the learned policy.

This slice is done when:

1. Registry collections are `so101-chess-act` and `so101-chess-smolvla`, and a logged dataset artifact is named `so101-chess` with `source=sim` when the dataset was generated in simulation.
2. The M1 dataset has been logged once under that artifact name.
3. New sim episodes render Staunton meshes and apply domain randomization, and the expert gate still runs without randomization.
4. `python -m lelab.sim evaluate` scores a checkpoint on the 20-transfer plan and, when Forge is configured, logs a Weave evaluation named `so101-chess-sim-eval`.
5. An ACT train on this Mac, on 30 newly generated episodes, writes a checkpoint that the eval command runs to completion.
6. Record can start "Generate in simulation" and Evaluate can start "In simulation".

## 2. Names

Only the registry collections and the dataset artifact change.

| What | Name |
|---|---|
| ACT checkpoints | `so101-chess-act` |
| SmolVLA checkpoints | `so101-chess-smolvla` |
| Dataset artifact | `so101-chess` |
| Weave evaluation | `so101-chess-sim-eval` |

`collection_name(policy_type, skill)` returns the chess collection for `act` and `smolvla`. The skill argument is accepted and ignored, so existing callers keep working, and ACT no longer requires a skill slug. One chess skill means there is no `so101-chess-act-<skill>` suffix. Unknown policy types still raise. `list_collections` lists collections whose names start with `so101-chess`, so the Setup page no longer shows `so101-kitting-*` collections.

`log_dataset_if_configured` builds a W&B artifact named `so101-chess`. Its metadata keeps `repo_id` and the episode counts, and adds `source` from `meta/garage_sim.json` when that file exists (`sim` for this dataset).

These stay as they are: the default W&B project `so101-kitting`, the ARIA automation name, the policy-server deployment `so101-kitting`, the Improve page, and the curate notebook's `so101-kitting:latest`. The notebook therefore does not download the chess dataset. Tests that assert collection or dataset-artifact names are updated. Tests that assert the project name are not.

After the rename, log the existing M1 dataset once. It is about 3.7 GB. Do not regenerate it.

## 3. Scene

### Staunton visuals

Piece visuals switch from the primitive shapes to Staunton meshes from [clarkerubber/Staunton-Pieces](https://github.com/clarkerubber/Staunton-Pieces) (MIT), at one pinned revision, downloaded into the sim asset cache the same way the arm model is. They are not committed to the repo. Each mesh is scaled so its base matches that piece type's base diameter in `SimCell`. Collision geometry stays the current convex primitives, so the expert and the judge are unchanged by the mesh swap. New generation uses the meshes by default. The M1 dataset keeps the primitive look it was recorded with.

### Domain randomization

`randomize.py` perturbs a scene before planning, and the record pass rebuilds from the same seed so the two passes see one scene:

- light direction and intensity
- board and piece colours
- table texture
- `top` and `side` camera position ±1 cm and angle ±2°
- field of view ±2°
- piece mass and friction ±30%
- actuator gain ±15%
- initial joints ±2°

The expert gate leaves randomization off, so the 97% M1 result stays comparable. Generation turns it on by default and accepts a flag to turn it off. The goal overlay is built from the camera after randomization, so the markers still sit on the source and target squares in the image the policy sees. `meta/garage_sim.json` records whether randomization was on, the seed, and `visuals` as `primitives` or `staunton`.

The M1 dataset is the randomization-off, primitive-piece arm of the later comparison. This slice does not generate the 250, 1,000, or 4,000 episode sets.

## 4. Evaluation

`python -m lelab.sim evaluate --checkpoint <path> --seeds 3` loads a local checkpoint, runs the 20-transfer plan for that many seeds in `ChessTransferEnv` with randomization on, and calls LeRobot's `predict_action` with the checkpoint's processors. Each trial records the judge's outcome, the failure class, and the duration. When Forge is configured, the trials are a Weave evaluation named `so101-chess-sim-eval`, with a `top` video per trial and a summary by transfer class and failure class. A local JSON summary is written either way.

The proof is operational, not a unit test. Generate 30 episodes with Staunton meshes and randomization on, train ACT on this Mac until it writes a checkpoint the eval command can load, and run the eval command on that checkpoint for 3 seeds. The run has to finish and write its summary. It does not have to reach 80%.

Record offers "Generate in simulation" and Evaluate offers "In simulation". Both call the same entry points as the CLI. The Mac eval uses 3 seeds. A Sandbox run with 20 seeds is out of this slice.

## 5. Out of scope

- CoreWeave Sandbox training and evaluation jobs.
- The data-size comparison (250 / 1,000 / 4,000), the randomization on/off comparison beyond recording the flag, and ACT versus SmolVLA.
- Promoting a checkpoint to the `sim-best` alias, and the 80% sim target.
- Real-arm recording, calibration, and evaluation (M3).
- Renaming the W&B project, the ARIA automation, or the policy-server deployment.

## 6. Tests

Unit tests do not start the arm, do not call W&B, and do not need a GPU. Registry and artifact tests cover the new names with the existing fakes. Randomization tests check that a seed reproduces a scene and that the gate path does not apply it. Eval tests drive a stand-in policy through the transfer plan and check the summary. Mesh download and the short ACT train are checked once by hand, the same way the M1 asset download and the 1,000-episode run were.
