import { useEffect, useState, type ReactNode } from "react";
import { Cable, Cpu, Monitor, Unplug } from "lucide-react";
import HfAuthChip from "@/components/landing/HfAuthChip";
import TourLauncher from "@/components/onboarding/TourLauncher";

interface CellStatus {
  arm: "Arm idle" | "Arm in use";
  job: string;
}

const IDLE: CellStatus = { arm: "Arm idle", job: "No job" };

async function readCellStatus(): Promise<CellStatus> {
  const [teleop, recording, replay, inference, jobs] = await Promise.all(
    ["/teleoperation-status", "/recording-status", "/replay-status", "/inference-status", "/jobs"].map(
      (path) => fetch(path).then((r) => (r.ok ? r.json() : {})),
    ),
  );
  const armActive = Boolean(
    teleop.teleoperation_active ||
      recording.recording_active ||
      replay.replay_active ||
      inference.inference_active,
  );
  const jobsList = Array.isArray(jobs.jobs)
    ? (jobs.jobs as { state?: string; name?: string }[])
    : [];
  const running = jobsList.find((job) => job.state === "running");
  return { arm: armActive ? "Arm in use" : "Arm idle", job: running?.name ?? "No job" };
}

function Slot({ icon, label }: { icon: ReactNode; label: string }) {
  return (
    <span className="flex items-center gap-2 text-sm">
      <span aria-hidden className="text-primary">{icon}</span>
      {label}
    </span>
  );
}

export function StatusBar() {
  const [status, setStatus] = useState<CellStatus>(IDLE);

  useEffect(() => {
    let cancelled = false;
    const tick = () => {
      // A synchronous throw from fetch must not escape the effect.
      Promise.resolve()
        .then(readCellStatus)
        .then((next) => { if (!cancelled) setStatus(next); })
        .catch(() => undefined);
    };
    tick();
    const id = setInterval(tick, 5000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  return (
    <div className="flex h-12 items-center gap-6 border-b border-border px-4">
      <Slot icon={<Cable size={16} />} label={status.arm} />
      <Slot icon={<Cpu size={16} />} label={status.job} />
      <Slot icon={<Monitor size={16} />} label="Local" />
      <Slot icon={<Unplug size={16} />} label="Not connected" />
      <div className="ml-auto flex items-center gap-2">
        <HfAuthChip />
        <TourLauncher variant="inline" />
      </div>
    </div>
  );
}
