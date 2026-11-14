import { useEffect, useState, type ReactNode } from "react";
import { Cable, ClipboardCheck, Cpu, Monitor } from "lucide-react";
import TourLauncher from "@/components/onboarding/TourLauncher";
import { useApi } from "@/contexts/ApiContext";
import { ForgeStatusChips } from "./ForgeStatusChips";

type Compute = "Idle" | "Local" | "CoreWeave" | "Hugging Face";

interface CellStatus {
  arm: "Arm idle" | "Arm in use";
  job: string;
  compute: Compute;
  evaluation: "No evaluation" | "Evaluation open";
  coreweaveRuns: number;
}

const IDLE: CellStatus = {
  arm: "Arm idle",
  job: "No job",
  compute: "Idle",
  evaluation: "No evaluation",
  coreweaveRuns: 0,
};

const COMPUTE: Record<string, Compute> = { coreweave: "CoreWeave", hf_cloud: "Hugging Face" };

type StatusBody = {
  teleoperation_active?: unknown;
  recording_active?: unknown;
  replay_active?: unknown;
  inference_active?: unknown;
  jobs?: unknown;
  open?: unknown;
  policy_runs_on?: unknown;
};

async function readCellStatus(baseUrl: string): Promise<CellStatus> {
  const [teleop, recording, replay, inference, jobs, evaluation] = await Promise.all(
    [
      "/teleoperation-status",
      "/recording-status",
      "/replay-status",
      "/inference-status",
      "/jobs",
      "/evaluations/current",
    ].map(
      (path): Promise<StatusBody> =>
        fetch(`${baseUrl}${path}`).then((r) => (r.ok ? r.json() : {})),
    ),
  );
  const armActive = Boolean(
    teleop.teleoperation_active ||
      recording.recording_active ||
      replay.replay_active ||
      inference.inference_active,
  );
  const jobsList = Array.isArray(jobs.jobs)
    ? (jobs.jobs as { state?: string; name?: string; runner?: string }[])
    : [];
  const runningJobs = jobsList.filter((job) => job.state === "running");
  const running = runningJobs[0];
  return {
    arm: armActive ? "Arm in use" : "Arm idle",
    job: running?.name ?? "No job",
    compute: running
      ? (running.runner && COMPUTE[running.runner]) || "Local"
      : evaluation.open === true
        ? evaluation.policy_runs_on === "this machine"
          ? "Local"
          : "CoreWeave"
        : "Idle",
    evaluation: evaluation.open === true ? "Evaluation open" : "No evaluation",
    coreweaveRuns: runningJobs.filter((job) => job.runner === "coreweave").length,
  };
}

function Slot({ icon, label }: { icon: ReactNode; label: string }) {
  return (
    <span className="flex items-center gap-2 whitespace-nowrap text-sm">
      <span aria-hidden className="text-primary">{icon}</span>
      {label}
    </span>
  );
}

export function StatusBar() {
  const { baseUrl } = useApi();
  const [status, setStatus] = useState<CellStatus>(IDLE);

  useEffect(() => {
    let cancelled = false;
    const tick = () => {
      // A synchronous throw from fetch must not escape the effect.
      Promise.resolve()
        .then(() => readCellStatus(baseUrl))
        .then((next) => { if (!cancelled) setStatus(next); })
        .catch(() => undefined);
    };
    tick();
    const id = setInterval(tick, 5000);
    return () => { cancelled = true; clearInterval(id); };
  }, [baseUrl]);

  return (
    <div className="flex min-h-12 flex-wrap items-center gap-x-6 gap-y-1 border-b border-border px-4 py-1">
      <Slot icon={<Cable size={16} />} label={status.arm} />
      <Slot icon={<Cpu size={16} />} label={status.job} />
      <Slot icon={<Monitor size={16} />} label={status.compute} />
      <Slot icon={<ClipboardCheck size={16} />} label={status.evaluation} />
      <div className="ml-auto flex flex-wrap items-center gap-2">
        <ForgeStatusChips coreweaveRuns={status.coreweaveRuns} />
        <TourLauncher variant="inline" />
      </div>
    </div>
  );
}
