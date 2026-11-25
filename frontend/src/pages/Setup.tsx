import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useApi } from "@/contexts/ApiContext";
import { useRobots } from "@/hooks/useRobots";
import ForgeCredentialsForm, {
  type ForgeStatus,
} from "@/components/setup/ForgeCredentialsForm";
import { RegistryStatus } from "@/components/setup/RegistryStatus";
import BoardCalibrationCard from "@/components/setup/BoardCalibrationCard";
import CameraPoseCard from "@/components/setup/CameraPoseCard";
import { calibrationRobotName } from "@/pages/setupLinks";

export default function Setup() {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { records, selectedName, availableNames, isLoading } = useRobots();
  const [forgeStatus, setForgeStatus] = useState<ForgeStatus | null>(null);
  const robotName = calibrationRobotName(selectedName, availableNames);
  const topCamera = (robotName ? records[robotName] : undefined)?.cameras.find((c) => c.name === "top");

  useEffect(() => {
    let cancelled = false;
    fetchWithHeaders(`${baseUrl}/forge-status`)
      .then(async (response) => {
        if (!response.ok) throw new Error(`forge-status ${response.status}`);
        return (await response.json()) as ForgeStatus;
      })
      .then((body) => {
        if (!cancelled) setForgeStatus(body);
      })
      .catch(() => {
        if (!cancelled) {
          setForgeStatus({
            forge_configured: false,
            sandbox_configured: false,
            entity: null,
            project: null,
            registry: { ready: false, collections: [] },
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders]);

  const calibrateState = robotName ? { robot_name: robotName } : undefined;
  const calibrateDetail = isLoading
    ? "Loading robots…"
    : robotName
      ? `Ports, motor range, and wrist / top / side cameras for "${robotName}".`
      : "Select or create a robot on Home first, then open Setup again.";

  return (
    <section className="mx-auto max-w-3xl p-8">
      <h1 className="font-extrabold text-3xl">Setup</h1>
      <p className="mt-2 text-muted-foreground">Work through this once per cell.</p>
      <ol className="mt-6 flex flex-col gap-3">
        <li>
          <Link
            to={robotName ? "/calibration" : "/"}
            state={calibrateState}
            className="block rounded-md border border-border p-4 hover:bg-accent"
          >
            <span className="font-bold">Calibrate arms and cameras</span>
            <span className="mt-1 block text-sm text-muted-foreground">{calibrateDetail}</span>
          </Link>
        </li>
        <li className="rounded-md border border-border p-4">
          <span className="font-bold">Board calibration</span>
          <span className="mt-1 mb-3 block text-sm text-muted-foreground">
            Mark the board, bin and parking squares on the top camera, then verify the detected position.
          </span>
          <BoardCalibrationCard robot={robotName ?? undefined} topCamera={topCamera ?? null} />
        </li>
        <li className="rounded-md border border-border p-4">
          <span className="font-bold">Camera positions for the simulation</span>
          <span className="mt-1 mb-3 block text-sm text-muted-foreground">
            Click the four board corners on each camera's straightened reference. The fit is written to the sim
            cell so simulated top and side pictures match the real ones.
          </span>
          <div className="flex flex-col gap-6">
            <CameraPoseCard robot={robotName ?? undefined} camera="side" />
            <CameraPoseCard robot={robotName ?? undefined} camera="top" />
          </div>
        </li>
        <li className="rounded-md border border-border p-4">
          <ForgeCredentialsForm
            status={forgeStatus}
            onSaved={(next) => setForgeStatus(next)}
          />
        </li>
        <li className="rounded-md border border-border p-4">
          <span className="font-bold">Model Registry</span>
          <RegistryStatus registry={forgeStatus?.registry ?? null} />
        </li>
      </ol>
    </section>
  );
}
