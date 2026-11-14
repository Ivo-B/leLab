import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiProvider } from "@/contexts/ApiContext";
import type { JobRecord } from "@/lib/jobsApi";
import JobCard from "./JobCard";

function job(overrides: Partial<JobRecord> = {}): JobRecord {
  return {
    id: "job-1",
    name: "so101-smolvla",
    state: "running",
    config: {} as JobRecord["config"],
    output_dir: "/tmp/out",
    started_at: Date.now() / 1000,
    ended_at: null,
    exit_code: null,
    error_message: null,
    metrics: { current_step: 10, total_steps: 100 } as JobRecord["metrics"],
    runner: "local",
    hf_job_id: null,
    hf_flavor: null,
    hf_repo_id: null,
    hf_job_url: null,
    tracking_url: null,
    checkpoint_count: 0,
    ...overrides,
  };
}

function renderCard(record: JobRecord) {
  return render(
    <MemoryRouter>
      <ApiProvider>
        <JobCard job={record} onStop={() => undefined} onDelete={() => undefined} onPlay={() => undefined} />
      </ApiProvider>
    </MemoryRouter>,
  );
}

describe("JobCard", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve({ ok: true, json: () => Promise.resolve([]) } as Response)));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("labels a CoreWeave run and links its Weights & Biases run", () => {
    renderCard(job({ runner: "coreweave", tracking_url: "https://wandb.ai/team/so101-kitting/runs/abc" }));
    expect(screen.getByText("CoreWeave Sandbox")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open run in Weights & Biases" })).toHaveAttribute(
      "href",
      "https://wandb.ai/team/so101-kitting/runs/abc",
    );
    expect(screen.getByRole("button", { name: "Stop job" })).toBeInTheDocument();
  });

  it("shows the CoreWeave label without a link before the run URL is known", () => {
    renderCard(job({ runner: "coreweave" }));
    expect(screen.getByText("CoreWeave Sandbox")).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Open run in Weights & Biases" })).not.toBeInTheDocument();
  });

  it("does not link a tracking URL that is not Weights & Biases", () => {
    renderCard(job({ runner: "coreweave", tracking_url: "https://huggingface.co/spaces/team/trackio" }));
    expect(screen.queryByRole("link", { name: "Open run in Weights & Biases" })).not.toBeInTheDocument();
  });

  it("does not label a local run", () => {
    renderCard(job());
    expect(screen.queryByText("CoreWeave Sandbox")).not.toBeInTheDocument();
  });
});
