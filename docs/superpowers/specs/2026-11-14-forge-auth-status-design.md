# Forge auth status

**Date:** 2026-11-14

**Status:** Implemented locally

## Status bar

The Hugging Face chip leaves the status bar. Two chips replace it, read once from `GET /forge-status`. The five-second cell poll does not call that endpoint. The response never includes secret values.

Weights & Biases is connected when `forge_configured` is exactly true and `entity` is a non-empty string. The chip shows the entity. Otherwise the chip is a button labeled Weights & Biases. Clicking it says `Add WANDB_API_KEY and WANDB_ENTITY on Setup.` and links to `/setup`.

CoreWeave is ready when `sandbox_configured` is exactly true. The chip reads CoreWeave. Otherwise the chip is a button labeled CoreWeave. Clicking it says `Add CWSANDBOX_API_KEY and CWSANDBOX_TRAIN_IMAGE on Setup.` and links to `/setup`. A failed status request shows both as not configured. The status bar does not call the Sandbox API.

## CoreWeave jobs

The fixed `Local` slot reads the first running training job's runner: `CoreWeave`, `Hugging Face`, or `Local`. The fixed `Not connected` slot reads `GET /evaluations/current`: `Evaluation open` or `No evaluation`. Both ride on the five-second cell poll. When CoreWeave is ready and training runs are going there, its chip reads `CoreWeave · 1 run` or `CoreWeave · N runs`.

A CoreWeave job card shows `CoreWeave Sandbox`. When the job's tracking URL is on `wandb.ai`, the card also links `Open run in Weights & Biases`.

`GET /evaluations/current` returns `{"open": false}` or the open session: `policy_family`, `version`, `policy_host`, `policy_runs_on`, `trials`, `finished`, `running`. Nothing starts the Sandbox policy host yet, so `policy_runs_on` is always `this machine`. When CoreWeave is picked on Evaluate, the page says `The CoreWeave policy host is not wired yet. Trials run on this machine and are logged as local in Weave.` Weave rows record the host a trial ran on and keep the requested host as requested_policy_host.

## Online jobs

The sentence `Sign in with Hugging Face to see your cloud jobs.` is replaced by `No active cloud jobs.` Hub upload, replay from the Hub, and listed cloud jobs stay.
