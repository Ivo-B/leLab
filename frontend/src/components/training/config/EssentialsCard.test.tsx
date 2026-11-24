import { render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import EssentialsCard from "./EssentialsCard";
import { TrainingConfig } from "../types";

vi.mock("./ChessDatasetCheck", () => ({ default: () => null }));

const config: TrainingConfig = {
  target: { runner: "local" },
  dataset_repo_id: "user/chess",
  policy_type: "act",
  steps: 10000,
  batch_size: 8,
  num_workers: 4,
  log_freq: 250,
  save_freq: 1000,
  save_checkpoint: true,
  resume: false,
  trackio_enable: false,
  wandb_enable: true,
  policy_use_amp: false,
  use_policy_training_preset: true,
};

it("has no Skill field when W&B is on and the policy is ACT", () => {
  render(<EssentialsCard config={config} updateConfig={vi.fn()} datasets={[]} datasetsLoading={false} />);
  expect(screen.getByLabelText("Track with Weights & Biases")).toBeInTheDocument();
  expect(screen.queryByLabelText("Skill")).toBeNull();
});
