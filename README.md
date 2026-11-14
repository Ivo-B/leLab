<h1 align="center">🦾 LeLab</h1>

<p align="center">
  <b>The official graphical interface for <a href="https://github.com/huggingface/lerobot">LeRobot</a>.</b>
</p>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/huggingface/leLab/blob/main/LICENSE)
[![HF Space](https://img.shields.io/badge/🤗-Open%20in%20Spaces-yellow)](https://huggingface.co/spaces/lerobot/LeLab)
[![Discord](https://img.shields.io/badge/Discord-Join_Us-5865F2?style=flat&logo=discord&logoColor=white)](https://discord.gg/q8Dzzpym3f)

</div>

**LeLab** is a web app that puts the full LeRobot workflow — calibrate, teleoperate, record, train, replay — into a single browser UI. Plug in your arm, open the app, and go. No CLI gymnastics, no keyboard prompts.

🤗 A web-native front door to LeRobot, designed so newcomers can get from "unboxing" to "training their first policy" in minutes.

🤗 Install and run everything with a single command.

## Quick Start

Grab the one-liner from the [LeLab Space](https://huggingface.co/spaces/lerobot/LeLab) — it installs and runs LeLab + LeRobot in a single command.

A page will automatically open in your browser and you are ready to go.

## What you can do

<div align="center">
  <table>
    <tr>
      <td>🎯 <b>Calibrate</b></td>
      <td>Guided web flow for both arms — no keyboard prompts.</td>
    </tr>
    <tr>
      <td>🕹️ <b>Teleoperate</b></td>
      <td>Move the leader, the follower mirrors it. Live joint streaming.</td>
    </tr>
    <tr>
      <td>📹 <b>Record</b></td>
      <td>Capture episodes into a LeRobotDataset, with cameras.</td>
    </tr>
    <tr>
      <td>🔎 <b>Browse</b></td>
      <td>Watch the episodes of a dataset on disk — every camera, frame by frame, with a joint-motion trace. No upload required.</td>
    </tr>
    <tr>
      <td>🧠 <b>Train</b></td>
      <td>Kick off a LeRobot training job, watch logs live.</td>
    </tr>
    <tr>
      <td>🤖 <b>Run inference</b></td>
      <td>Execute a trained policy on the follower.</td>
    </tr>
    <tr>
      <td>⏪ <b>Replay</b></td>
      <td>Re-run any recorded episode.</td>
    </tr>
    <tr>
      <td>☁️ <b>Upload</b></td>
      <td>Push your dataset to the <a href="https://huggingface.co/">Hugging Face Hub</a> in one click.</td>
    </tr>
  </table>
</div>

## Camera rotation

If a camera is mounted upside down or sideways, select its **Rotation** in the camera setup on the Calibration page.
Choose 0°, 90°, 180°, or 270° clockwise. For an upside-down wrist camera, choose 180°.

LeLab saves the rotation for each camera. Setup and teleoperation previews, recorded frames, and inference use the selected rotation.
Profiles without a rotation setting use 0°.

The resolution setting describes the camera's capture resolution. A 90° or 270° rotation swaps the output dimensions.
For example, a 640 × 480 capture becomes a 480 × 640 observation. The preview shows the complete rotated frame.

Before inference, select the same rotation used for the training dataset. Check it again if you change the camera binding.
Checkpoint image dimensions do not establish image orientation. For example, 0° and 180° have the same dimensions.
The setting applies to new observations; it does not change existing recordings or checkpoints.

## Resources

- **[LeRobot](https://github.com/huggingface/lerobot):** the underlying library — go here for everything beyond the UI.
- **[LeLab Space](https://huggingface.co/spaces/lerobot/LeLab):** try the UI in your browser.
- **[Discord](https://discord.gg/q8Dzzpym3f):** chat with the LeRobot community.
- **[CLAUDE.md](CLAUDE.md):** architecture rundown for contributors.

## CoreWeave Forge

The Forge storyline takes one SO-101 kitting task from a first recording to a better policy, with a W&B Report written by ARIA after each evaluation.

1. **Setup:** configure LeLab, the arm, and your Weights & Biases and CoreWeave settings.
2. **Record:** record demonstrations of each kitting skill with the leader arm.
3. **Curate:** open `notebooks/curate_dataset.py` to review episodes and drop the bad ones.
4. **Train:** train a policy on the curated dataset.
5. **Evaluate:** run the trials; each trial is scored by stage (started, reached, grasped, delivered) in a Weave evaluation.
6. **Record more and Promote:** record more demonstrations for the weakest skill, retrain, and Promote the better policy.
7. **Order:** chain the skills into one kitting order and run it.
8. **Repeat on the CoreWeave policy host:** run the same loop with the policy served from CoreWeave.

### ARIA report

Create a W&B automation named `so101-kitting evaluation`. Set its event to `A run reaches FINISHED` and its action to Trigger ARIA. The prompt below is the brief ARIA should write in the W&B Report:

> Look at the Weave evaluation that just finished in this project. Name the skill and the stage where trials failed. The stages are started, reached, grasped, and delivered. A trial with no stage needs review and counts as a failure. Propose the next experiment: which skill needs more demonstrations, and whether to retrain SmolVLA or the ACT policy for that skill. Do not change an alias and do not launch a job.

The prompt text lives in `aria_prompt()` in `lelab/improve/aria.py`, and a test keeps this README in sync with it.

## Contribute

PRs welcome. Hot-reload mode for working on the code:

```bash
lelab --dev
```

Vite on `:8080`, uvicorn `--reload` on `:8000`. Installing LeLab builds the frontend bundle, so PRs never include `frontend/dist`.

<div align="center">
<sub>Originally hacked together by <a href="https://www.linkedin.com/posts/nicolas-rabault-_lerobot-hackathon-lerobot-ugcPost-7341065019368828930-jTnl/">Team LeLab at the 2025 LeRobot Worldwide Hackathon 🏆</a>, now maintained by the <a href="https://huggingface.co/lerobot">LeRobot</a> team at <a href="https://huggingface.co">Hugging Face</a> with ❤️</sub>
</div>
