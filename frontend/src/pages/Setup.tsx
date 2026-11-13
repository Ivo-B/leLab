import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useApi } from "@/contexts/ApiContext";
import { setupSandboxDetail } from "@/components/training/target";

const LINKS = [
  { href: "/calibration", title: "Calibrate the leader and follower arms", detail: "Ports, motor direction, and range." },
  { href: "/calibration#calibration-cameras", title: "Check the cameras", detail: "Wrist, top, and side. The side camera is recorded and kept out of the policy." },
  { href: "/", title: "Sign in to Hugging Face", detail: "Datasets and policies still push to the Hub." },
];

const FORGE_PENDING = setupSandboxDetail(false, false, "");
const FORGE_MISSING = FORGE_PENDING;

export default function Setup() {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [forgeDetail, setForgeDetail] = useState(FORGE_PENDING);

  useEffect(() => {
    let cancelled = false;
    fetchWithHeaders(`${baseUrl}/forge-status`)
      .then(async (response) => {
        if (!response.ok) throw new Error(`forge-status ${response.status}`);
        return (await response.json()) as {
          forge_configured: boolean;
          sandbox_configured?: boolean;
          entity: string | null;
          project: string | null;
        };
      })
      .then((body) => {
        if (cancelled) return;
        setForgeDetail(
          setupSandboxDetail(
            body.forge_configured,
            body.sandbox_configured === true,
            body.entity ?? "",
          ),
        );
      })
      .catch(() => {
        if (!cancelled) setForgeDetail(FORGE_MISSING);
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders]);

  return (
    <section className="mx-auto max-w-3xl p-8">
      <h1 className="font-extrabold text-3xl">Setup</h1>
      <p className="mt-2 text-muted-foreground">Work through this once per cell.</p>
      <ol className="mt-6 flex flex-col gap-3">
        {LINKS.map((item) => (
          <li key={item.title}>
            <Link to={item.href} className="block rounded-md border border-border p-4 hover:bg-accent">
              <span className="font-bold">{item.title}</span>
              <span className="mt-1 block text-sm text-muted-foreground">{item.detail}</span>
            </Link>
          </li>
        ))}
        <li className="rounded-md border border-border p-4">
          <span className="font-bold">Connect Forge</span>
          <span className="mt-1 block text-sm text-muted-foreground">{forgeDetail}</span>
        </li>
      </ol>
    </section>
  );
}
