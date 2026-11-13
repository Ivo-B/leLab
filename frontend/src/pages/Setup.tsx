import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useApi } from "@/contexts/ApiContext";
import { useRobots } from "@/hooks/useRobots";
import ForgeCredentialsForm, {
  type ForgeStatus,
} from "@/components/setup/ForgeCredentialsForm";
import { RegistryStatus } from "@/components/setup/RegistryStatus";
import { calibrationRobotName } from "@/pages/setupLinks";

export default function Setup() {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { selectedName, availableNames, isLoading } = useRobots();
  const [forgeStatus, setForgeStatus] = useState<ForgeStatus | null>(null);
  const robotName = calibrationRobotName(selectedName, availableNames);

  const refreshForgeStatus = useCallback(() => {
    return fetchWithHeaders(`${baseUrl}/forge-status`)
      .then(async (response) => {
        if (!response.ok) throw new Error(`forge-status ${response.status}`);
        return (await response.json()) as ForgeStatus;
      })
      .then((body) => {
        setForgeStatus(body);
        return body;
      })
      .catch(() => {
        const fallback: ForgeStatus = {
          forge_configured: false,
          sandbox_configured: false,
          entity: null,
          project: null,
          registry: { ready: false, collections: [] },
        };
        setForgeStatus(fallback);
        return fallback;
      });
  }, [baseUrl, fetchWithHeaders]);

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
