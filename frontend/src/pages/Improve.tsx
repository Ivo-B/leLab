import { useEffect, useState } from "react";
import { useApi } from "@/contexts/ApiContext";
import { PlaceholderPage } from "@/pages/PlaceholderPage";

const PLACEHOLDER_BODY = "CoreWeave ARIA reports arrive in a later milestone.";
const EMPTY_STATE = "Close an evaluation, then open the project Automations tab and choose Trigger ARIA.";

// Keep these three strings in step with lelab/improve/aria.py.
const AUTOMATION_NAME = "so101-kitting evaluation";
const AUTOMATION_EVENT = "A run reaches FINISHED";
const ARIA_PROMPT =
  "Look at the Weave evaluation that just finished in this project. " +
  "Name the skill and the stage where trials failed. " +
  "The stages are started, reached, grasped, and delivered. " +
  "A trial with no stage needs review and counts as a failure. " +
  "Propose the next experiment: which skill needs more demonstrations, " +
  "and whether to retrain SmolVLA or the ACT policy for that skill. " +
  "Do not change an alias and do not launch a job.";

type ForgeState = "loading" | "off" | "on";

export default function Improve() {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [forge, setForge] = useState<ForgeState>("loading");

  useEffect(() => {
    let cancelled = false;
    fetchWithHeaders(`${baseUrl}/forge-status`)
      .then(async (response) => {
        if (!response.ok) throw new Error(`forge-status ${response.status}`);
        return (await response.json()) as { forge_configured?: unknown };
      })
      .then((body) => {
        if (!cancelled) setForge(body.forge_configured === true ? "on" : "off");
      })
      .catch(() => {
        if (!cancelled) setForge("off");
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders]);

  if (forge === "loading") return null;
  if (forge === "off") return <PlaceholderPage title="Improve" body={PLACEHOLDER_BODY} />;

  return (
    <section className="mx-auto max-w-3xl p-8">
      <h1 className="font-extrabold text-3xl">Improve</h1>
      <dl className="mt-6 flex flex-col gap-4">
        <div>
          <dt className="text-sm font-bold">Automation</dt>
          <dd className="mt-1 text-base">{AUTOMATION_NAME}</dd>
        </div>
        <div>
          <dt className="text-sm font-bold">Event</dt>
          <dd className="mt-1 text-base">{AUTOMATION_EVENT}</dd>
        </div>
        <div>
          <dt className="text-sm font-bold">Prompt</dt>
          <dd className="mt-1 text-base">{ARIA_PROMPT}</dd>
        </div>
      </dl>
      <p className="mt-8 text-base text-muted-foreground">{EMPTY_STATE}</p>
    </section>
  );
}
