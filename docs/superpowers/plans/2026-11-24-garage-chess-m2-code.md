# Chess M2 code slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the registry collections and the dataset artifact to chess, render Staunton pieces with domain randomization, and evaluate a locally trained ACT checkpoint in simulation.

**Architecture:** `collection_name` and `log_dataset_if_configured` are the only Forge names that change. Staunton meshes are visuals downloaded like the arm model; collisions stay the M1 primitives. `randomize.py` perturbs a scene from a seed before the expert plans, and the record pass rebuilds that same scene. `evaluate.py` runs a checkpoint through `ChessTransferEnv` and the existing judge.

**Tech Stack:** Python 3.12, MuJoCo, LeRobot v0.7.0 (`predict_action`, ACT training), Weights & Biases artifacts and Weave, React + Vite.

## Global Constraints

- Work on branch `forge/chess`. Do not push unless the controller asks.
- Spec: `docs/superpowers/specs/2026-11-24-garage-chess-m2-code-design.md`. Its names are exact: collections `so101-chess-act` and `so101-chess-smolvla`, dataset artifact `so101-chess`, Weave evaluation `so101-chess-sim-eval`.
- Do not rename the W&B project default `so101-kitting`, the ARIA automation, the policy-server deployment `so101-kitting`, the Improve page, or the curate notebook. (Note, 2026-11-24: `2026-11-24-garage-chess-cleanup.md` later renamed these to `so101-chess`, `so101-chess evaluation` and the `so101-chess-sim` / `so101-chess-real` dataset artifacts.)
- Do not start a Sandbox job, a 250/1,000/4,000 sweep, SmolVLA training, or a `sim-best` promotion.
- Do not start the robot arm in a unit test. Unit tests do not call W&B or the network.
- The working tree holds uncommitted camera-control, lens, and color-calibration work, plus `frontend/vite.config.ts`, `lelab/scripts/lelab.py`, `lelab/server.py`, `tests/test_scripts_lelab.py`, and `lerobot.code-workspace`. Do not commit them, do not revert them. Stage files by name. Never `git add -A` or `git commit -a`. Do not skip hooks.
- `WANDB_API_KEY` and `CWSANDBOX_API_KEY` never appear in a response, a log line, or a logger call.
- New Python modules start with the Apache-2.0 header in `tests/test_forge_config.py` lines 1–13, then a blank line.
- Run Python tests with `uv run pytest`. Frontend tests with `npm test -- --run` inside `frontend/`. Lint with `uv run ruff check` and `uv run ruff format --check` on the files you touch.
- MuJoCo rendering needs a GL context and runs outside the sandbox (`required_permissions: ["all"]`). Tests that render skip when no context exists.
- The M1 dataset at `~/.cache/huggingface/lerobot/local/so101-chess-sim` is not regenerated.

---

### Task 1: Chess registry collections

**Files:**

- Modify: `lelab/forge/registry.py` (`collection_name`, `list_collections` prefix)
- Modify: `tests/test_forge_registry.py`
- Modify: any other test whose expected value is produced by `collection_name` or `list_collections` (found by running the tests below)

**Interfaces:**

- Consumes: `collection_name(policy_type: str, skill: str | None) -> str`
- Produces: `collection_name("act", None) == "so101-chess-act"`, `collection_name("act", "bracket") == "so101-chess-act"`, `collection_name("smolvla", "bolt") == "so101-chess-smolvla"`. Unknown types still raise `ValueError`. `list_collections` keeps only names starting with `so101-chess`.

- [ ] **Step 1: Rewrite the collection tests**

In `tests/test_forge_registry.py`, replace `test_act_collection_uses_the_skill_slug` and `test_act_without_a_slug_is_rejected` with:

```python
def test_act_has_one_collection() -> None:
    assert collection_name("act", None) == "so101-chess-act"
    assert collection_name("act", "bracket") == "so101-chess-act"
    assert collection_name("act", "Pick the bolt") == "so101-chess-act"
```

Change `test_smolvla_has_one_collection` to expect `so101-chess-smolvla`. In `list_collections` fixtures in this file, rename `so101-kitting-smolvla` and `so101-kitting-act-*` to `so101-chess-smolvla` and `so101-chess-act`. Leave every `so101-kitting` string that is a W&B project, a deployment, or an automation name.

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/test_forge_registry.py -q`
Expected: FAIL because `collection_name` still returns `so101-kitting-*`.

- [ ] **Step 3: Implement**

```python
def collection_name(policy_type: str, skill: str | None) -> str:
    """One chess collection per policy family. `skill` is accepted and ignored."""
    if policy_type == "smolvla":
        return "so101-chess-smolvla"
    if policy_type == "act":
        return "so101-chess-act"
    raise ValueError(f"No registry collection for policy type {policy_type}")
```

In `list_collections`, change the prefix check from `so101-kitting` to `so101-chess`.

- [ ] **Step 4: Update the other callers' tests**

Run: `uv run pytest tests/test_forge_registry.py tests/test_forge_evaluate.py tests/test_forge_tracking.py tests/test_forge_policy_server.py -q`

Update an assertion only when the value is produced by `collection_name` or `list_collections`. Do not change `project="so101-kitting"`, `deployment=so101-kitting`, or `so101-kitting evaluation`.

- [ ] **Step 5: Commit**

```bash
git add lelab/forge/registry.py tests/test_forge_registry.py
# plus only the other test files you actually changed
git commit -m "feat: point registry collections at so101-chess"
```

---

### Task 2: Chess dataset artifact

**Files:**

- Modify: `lelab/forge/tracking.py` (`log_dataset_if_configured`)
- Modify: `tests/test_forge_tracking.py`

**Interfaces:**

- Consumes: `log_dataset_if_configured(dataset_dir: Path, repo_id: str) -> str | None`
- Produces: the artifact name is `so101-chess`. Metadata is `{"repo_id", "episode_counts", "source"}`. `source` is the `source` field of `dataset_dir/meta/garage_sim.json` when that file exists, otherwise `None`.

- [ ] **Step 1: Extend the tracking test**

In the test that asserts `run.logged.name == "so101-kitting"`, expect `so101-chess`. Add a test that writes `meta/garage_sim.json` with `{"source": "sim"}` and asserts `metadata["source"] == "sim"`. Add a test with no such file and assert `metadata["source"] is None`. Keep `created["project"] == "so101-kitting"`.

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_forge_tracking.py -q`
Expected: FAIL on the artifact name.

- [ ] **Step 3: Implement**

Read `source` inside `log_dataset_if_configured`:

```python
def _dataset_source(dataset_dir: Path) -> str | None:
    path = dataset_dir / "meta" / "garage_sim.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text()).get("source")
    except (OSError, json.JSONDecodeError):
        return None
```

Pass `metadata={"repo_id": repo_id, "episode_counts": counts, "source": _dataset_source(dataset_dir)}` and name the artifact `"so101-chess"`.

- [ ] **Step 4: Commit**

```bash
git add lelab/forge/tracking.py tests/test_forge_tracking.py
git commit -m "feat: log datasets as the so101-chess artifact"
```

---

### Task 3: Staunton visuals

**Files:**

- Create: `lelab/sim/pieces.py`
- Modify: `lelab/sim/scene.py` (`_piece`)
- Create: `tests/test_sim_pieces.py`

**Interfaces:**

- Produces: `piece_mesh_path(symbol: str) -> Path` for `pnbrqk` (case-insensitive). `PIECES_REPO` and `PIECES_REVISION` are pinned constants. `piece_visual_xml(name, symbol, mesh_file) -> str` adds a collision cylinder of the current size and a visual mesh scaled so its base diameter equals `2 * radius`. Missing meshes raise `SimUnavailable` from the download helper, not from scene compilation.

- [ ] **Step 1: Write the failing test**

```python
def test_visual_uses_a_mesh_and_collision_stays_a_cylinder() -> None:
    from lelab.sim.pieces import piece_visual_xml

    xml = piece_visual_xml("p_e2_P", "P", "pawn.obj", radius=0.0085, height=0.030)
    assert 'mesh="' in xml or "mesh=" in xml
    assert 'type="cylinder"' in xml
    assert "pawn.obj" in xml
```

Mesh files are not downloaded in the unit test. Pass a filename; the test only checks the XML.

- [ ] **Step 2: Implement**

`lelab/sim/pieces.py` maps `p n b r q k` to the filenames in [clarkerubber/Staunton-Pieces](https://github.com/clarkerubber/Staunton-Pieces). `ensure_piece_meshes()` downloads one pinned git commit of that repo into `config.SIM_ASSETS_PATH / "staunton"` once. Record the URL and the commit as constants. Do not invent a Hugging Face repo for the meshes. Scale is `cell_radius / mesh_base_radius`, with `mesh_base_radius` read from the mesh's horizontal extent (or a constant measured once and stored next to the revision). `_piece` in `scene.py` calls `piece_visual_xml` instead of emitting the visual cylinder and sphere. The collision cylinder stays as it is today, including its size and position. `compile_scene` must be given the mesh directory as an asset path; thread that through `scene_xml` / `compile_scene` the same way the arm XML is included.

- [ ] **Step 3: Run**

Run: `uv run pytest tests/test_sim_pieces.py tests/test_sim_scene.py -q`
Expected: pass. A one-off compile of a scene outside the sandbox must load the real meshes once.

- [ ] **Step 4: Commit**

```bash
git add lelab/sim/pieces.py lelab/sim/scene.py tests/test_sim_pieces.py
git commit -m "feat: render Staunton meshes while collisions stay primitive"
```

---

### Task 4: Domain randomization

**Files:**

- Create: `lelab/sim/randomize.py`
- Modify: `lelab/sim/generate.py`, `lelab/sim/__main__.py`
- Create: `tests/test_sim_randomize.py`

**Interfaces:**

- Produces: `randomize_cell(cell: SimCell, rng: random.Random) -> SimCell` and `randomize_joints(joints, rng) -> tuple[float, ...]`. Ranges are the spec's: camera position ±1 cm, camera angle ±2°, field of view ±2°, mass and friction ±30%, actuator gain ±15%, initial joints ±2°, plus light direction, light intensity, board colour, piece colour, and table colour. `generate_dataset(..., randomize: bool = True)` stores `randomization: {"enabled": bool, "seed": int}` and `visuals: "staunton"` in `garage_sim.json`. The gate does not call `randomize_cell`.

- [ ] **Step 1: Write the failing test**

```python
def test_one_seed_reproduces_the_cell_and_the_gate_does_not_randomize() -> None:
    from lelab.sim.cell import SimCell
    from lelab.sim.randomize import randomize_cell

    cell = SimCell()
    one = randomize_cell(cell, random.Random(4))
    two = randomize_cell(cell, random.Random(4))
    assert one == two
    assert one.top_camera.pos != cell.top_camera.pos
    assert abs(one.top_camera.pos[0] - cell.top_camera.pos[0]) <= 0.01 + 1e-9
```

- [ ] **Step 2: Implement**

`randomize_cell` returns a new `SimCell` (the dataclass is frozen). Perturb `top_camera` and `side_camera` position and fovy. Rotate each camera's `xyaxes` by a yaw of at most 2°. Piece mass and friction are scene-level defaults today (`PIECE_MASS` in `scene.py`); pass the randomized mass and friction into `scene_xml` rather than storing them on `SimCell` if `SimCell` has no field for them. Add the parameters as explicit arguments to `scene_xml` with the current constants as defaults, so existing callers are unchanged. Light and colours are MJCF material and light attributes already written by `scene_xml`; randomize those arguments the same way. `generate_dataset` draws one `random.Random(episode_seed)` per attempt, builds the randomized cell before `plan_attempt`, and passes that cell dict into both the physics pass and the record pass. Add `--no-randomize` to the generate CLI. The gate path in `__main__` keeps calling `load_cell()` with no randomization.

- [ ] **Step 3: Run**

Run: `uv run pytest tests/test_sim_randomize.py tests/test_sim_generate.py -q`
Expected: pass, including the existing generate tests (they may pass `randomize=False` to stay fast and deterministic).

- [ ] **Step 4: Commit**

```bash
git add lelab/sim/randomize.py lelab/sim/scene.py lelab/sim/generate.py lelab/sim/__main__.py tests/test_sim_randomize.py tests/test_sim_generate.py
git commit -m "feat: randomize the sim scene from a seed, off for the expert gate"
```

---

### Task 5: Sim evaluation

**Files:**

- Create: `lelab/sim/evaluate.py`
- Modify: `lelab/sim/__main__.py` (add `evaluate`)
- Create: `tests/test_sim_evaluate.py`

**Interfaces:**

- Produces: `evaluate_checkpoint(checkpoint: Path, *, seeds: int = 3, cell: SimCell | None = None) -> EvalReport`. `EvalReport` has `trials: list[Trial]` and `summary: dict[str, dict[str, int]]` keyed by transfer class then by `success` or a failure name. Each `Trial` has `transfer`, `seed`, `outcome` (`success` or the failure name), and `duration_s`. The CLI is `python -m lelab.sim evaluate --checkpoint <path> [--seeds 3]`. When Forge is configured, log a Weave evaluation named `so101-chess-sim-eval` with one `top` video per trial. Always write `eval_report.json` next to the checkpoint's parent, or under a `--out` directory.

- [ ] **Step 1: Write the failing test**

The test does not load a real checkpoint and does not render. Pass a callable `predict` instead of a checkpoint when `checkpoint is None` is not the design; add an internal `run_plan(world_factory, predict, seeds) -> EvalReport` that the test calls with a function returning the PARK action. Assert one trial per transfer in `TRANSFER_PLAN` per seed, and that the summary counts add up to `len(TRANSFER_PLAN) * seeds`.

- [ ] **Step 2: Implement**

`evaluate_checkpoint` loads the policy with LeRobot's checkpoint API and calls `predict_action` per step using the checkpoint's pre/post processors. Observations come from `ChessTransferEnv.observation_for_policy`. Randomization is on. Stop a trial when the env terminates or truncates. Record `info["outcome"]`. Build the per-class summary with `transfer_class`. Weave logging is skipped when Forge is not configured, and the unit test never configures it.

- [ ] **Step 3: Run**

Run: `uv run pytest tests/test_sim_evaluate.py -q`
Expected: pass without GL and without a checkpoint.

- [ ] **Step 4: Commit**

```bash
git add lelab/sim/evaluate.py lelab/sim/__main__.py tests/test_sim_evaluate.py
git commit -m "feat: evaluate a checkpoint on the sim transfer plan"
```

---

### Task 6: Record and Evaluate controls

**Files:**

- Modify: `lelab/server.py` (two routes). The file has unrelated uncommitted edits. Do not stage it yourself; leave the new routes in the working tree and name them in the report so the controller can stage those hunks.
- Modify: `frontend/src/pages/Recording.tsx`, `frontend/src/pages/Evaluate.tsx`
- Modify: their tests

**Interfaces:**

- Produces: `POST /sim-generate` with `{episodes: int, seed: int, randomize: bool}` starts generation in a background thread and returns `{started: true}`. `GET /sim-generate` returns `{running, saved, episodes, error}`. `POST /sim-evaluate` with `{checkpoint: str, seeds: int}` starts an eval and returns `{started: true}`. `GET /sim-evaluate` returns `{running, report, error}`. The Record page shows "Generate in simulation". The Evaluate page shows "In simulation" and a checkpoint path. Neither control starts the arm.

- [ ] **Step 1: Write the failing frontend tests**

Assert the Recording page renders a button named "Generate in simulation" and that clicking it posts `{episodes: 30, seed: 0, randomize: true}` to `/sim-generate`. Assert the Evaluate page renders "In simulation" and posts the checkpoint path and `seeds: 3` to `/sim-evaluate`.

- [ ] **Step 2: Implement the routes and the controls**

The handlers call `generate_dataset` and `evaluate_checkpoint` on a daemon thread, one job at a time, and refuse to start a second. They do not import MuJoCo at server import time. Follow the status-endpoint style already used by recording.

- [ ] **Step 3: Run**

Run: `cd frontend && npm test -- --run src/pages/Recording.test.tsx src/pages/Evaluate.test.tsx` (use the existing test filenames). Run the Python test that covers the new routes if one exists; add `tests/test_sim_jobs.py` for the "already running" refusal, with the heavy functions patched.

- [ ] **Step 4: Commit**

Commit the frontend files and `tests/test_sim_jobs.py` only. Leave `lelab/server.py` unstaged.

```bash
git commit -m "feat: start sim generation and sim evaluation from the app"
```

---

### Task 7: Log the dataset and prove the loop

Operational. No new code unless Task 5 cannot load a real checkpoint.

- [ ] **Step 1: Log the M1 dataset**

Run, outside the sandbox, with Forge configured: a one-off call of `log_dataset_if_configured` on `~/.cache/huggingface/lerobot/local/so101-chess-sim` with repo id `local/so101-chess-sim`. Confirm the artifact name is `so101-chess` and `source` is `sim`. Do not print credentials.

- [ ] **Step 2: Generate 30 episodes**

`uv run python -m lelab.sim generate --repo-id local/so101-chess-sim-m2 --episodes 30 --no-artifact`
Expect Staunton visuals and randomization on. Keep the display awake (`caffeinate -dimsu`).

- [ ] **Step 3: Train ACT until a checkpoint exists**

Train with LeRobot's CLI on that dataset for the smallest step count that writes a loadable checkpoint, on this Mac (CPU or MPS). Do not submit a Sandbox job.

- [ ] **Step 4: Evaluate**

`uv run python -m lelab.sim evaluate --checkpoint <that checkpoint> --seeds 3`
The command must finish and write a summary. The success rate is recorded and not required to reach 80%.

- [ ] **Step 5: Commit only code fixes**

If nothing in the tree changed, do not commit.
