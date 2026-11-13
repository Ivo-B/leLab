import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useApi } from "@/contexts/ApiContext";
import { PlaceholderPage } from "@/pages/PlaceholderPage";

const PLACEHOLDER_BODY =
  "Real-arm evaluation in W&B Weave arrives in a later milestone. Rollouts still run from the inference page.";
const EMPTY_STATE = "Choose a checkpoint, then score it on the arm.";
const SANDBOX_HINT = "Add CWSANDBOX_API_KEY and CWSANDBOX_TRAIN_IMAGE to the backend environment.";

const STAGES = ["started", "reached", "grasped", "delivered"] as const;
const SELECT_CLASS =
  "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50";

type ForgeState = "loading" | "off" | { sandbox: boolean };

interface Session {
  trials: number;
}

interface Verdict {
  stage: string | null;
  verdict_source: string;
}

async function errorText(response: Response, fallback: string): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string" && body.detail) return body.detail;
  } catch {
    // Fall through to the generic message.
  }
  return fallback;
}

function Field({ id, label, children }: { id: string; label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={id} className="text-sm font-bold">
        {label}
      </label>
      {children}
    </div>
  );
}

function verdictText(verdict: Verdict): string {
  const source = `(${verdict.verdict_source})`;
  return verdict.stage === null
    ? `Needs review: no stage recorded ${source}`
    : `Stage: ${verdict.stage} ${source}`;
}

export default function Evaluate() {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [forge, setForge] = useState<ForgeState>("loading");

  const [family, setFamily] = useState("smolvla");
  const [version, setVersion] = useState("candidate");
  const [host, setHost] = useState("local");
  const [trials, setTrials] = useState("10");
  const [skill, setSkill] = useState("");

  const [session, setSession] = useState<Session | null>(null);
  const [verdict, setVerdict] = useState<Verdict | null>(null);
  const [stage, setStage] = useState<string>(STAGES[0]);
  const [weaveUrl, setWeaveUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchWithHeaders(`${baseUrl}/forge-status`)
      .then(async (response) => {
        if (!response.ok) throw new Error(`forge-status ${response.status}`);
        return (await response.json()) as { forge_configured?: unknown; sandbox_configured?: unknown };
      })
      .then((body) => {
        if (cancelled) return;
        setForge(body.forge_configured === true ? { sandbox: body.sandbox_configured === true } : "off");
      })
      .catch(() => {
        if (!cancelled) setForge("off");
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders]);

  const post = useCallback(
    async (path: string, body: unknown): Promise<Response> =>
      fetchWithHeaders(`${baseUrl}${path}`, { method: "POST", body: JSON.stringify(body) }),
    [baseUrl, fetchWithHeaders],
  );

  const start = async () => {
    setBusy(true);
    setError(null);
    setWeaveUrl(null);
    try {
      const request: Record<string, unknown> = {
        policy_family: family,
        version,
        policy_host: host,
        trials_per_skill: Number(trials),
      };
      if (family === "act") request.skill = skill.trim();
      const response = await post("/evaluations", request);
      if (!response.ok) {
        setError(await errorText(response, "Could not start the evaluation."));
        return;
      }
      const body = (await response.json()) as { trials: number };
      setSession({ trials: body.trials });
      setVerdict(null);
    } catch {
      setError("Could not reach the backend.");
    } finally {
      setBusy(false);
    }
  };

  const override = async (next: string | null) => {
    setBusy(true);
    setError(null);
    try {
      const response = await post("/evaluations/current/override", { stage: next });
      if (!response.ok) {
        setError(await errorText(response, "Could not record the verdict."));
        return;
      }
      const body = (await response.json()) as Verdict;
      setVerdict({ stage: body.stage, verdict_source: body.verdict_source });
      if (body.stage !== null) setStage(body.stage);
    } catch {
      setError("Could not reach the backend.");
    } finally {
      setBusy(false);
    }
  };

  const close = async () => {
    setBusy(true);
    setError(null);
    try {
      const response = await post("/evaluations/current/close", {});
      if (!response.ok) {
        setError(await errorText(response, "Could not close the evaluation."));
        return;
      }
      const body = (await response.json()) as { weave_url?: string | null };
      setWeaveUrl(body.weave_url ?? null);
      setSession(null);
      setVerdict(null);
    } catch {
      setError("Could not reach the backend.");
    } finally {
      setBusy(false);
    }
  };

  if (forge === "loading") {
    return (
      <section className="mx-auto max-w-3xl p-8">
        <h1 className="font-extrabold text-3xl">Evaluate</h1>
      </section>
    );
  }
  if (forge === "off") {
    return <PlaceholderPage title="Evaluate" body={PLACEHOLDER_BODY} />;
  }

  const trialsValid = Number.isInteger(Number(trials)) && Number(trials) >= 1;
  const canStart = !busy && session === null && trialsValid && (family !== "act" || skill.trim() !== "");

  return (
    <section className="mx-auto max-w-3xl p-8">
      <h1 className="font-extrabold text-3xl">Evaluate</h1>

      <div className="mt-6 grid gap-4 sm:grid-cols-2">
        <Field id="evaluate-family" label="Policy family">
          <select
            id="evaluate-family"
            className={SELECT_CLASS}
            value={family}
            disabled={session !== null}
            onChange={(event) => setFamily(event.target.value)}
          >
            <option value="smolvla">SmolVLA</option>
            <option value="act">ACT</option>
          </select>
        </Field>
        <Field id="evaluate-version" label="Version">
          <select
            id="evaluate-version"
            className={SELECT_CLASS}
            value={version}
            disabled={session !== null}
            onChange={(event) => setVersion(event.target.value)}
          >
            <option value="candidate">candidate</option>
            <option value="production">production</option>
          </select>
        </Field>
        <Field id="evaluate-host" label="Policy host">
          <select
            id="evaluate-host"
            className={SELECT_CLASS}
            value={host}
            disabled={session !== null}
            onChange={(event) => setHost(event.target.value)}
          >
            <option value="local">Local</option>
            <option value="coreweave" disabled={!forge.sandbox}>
              CoreWeave
            </option>
          </select>
          {!forge.sandbox && <p className="text-sm text-muted-foreground">{SANDBOX_HINT}</p>}
        </Field>
        <Field id="evaluate-trials" label="Trials">
          <Input
            id="evaluate-trials"
            type="number"
            min={1}
            value={trials}
            disabled={session !== null}
            onChange={(event) => setTrials(event.target.value)}
          />
        </Field>
        {family === "act" && (
          <Field id="evaluate-skill" label="Skill">
            <Input
              id="evaluate-skill"
              value={skill}
              disabled={session !== null}
              onChange={(event) => setSkill(event.target.value)}
            />
          </Field>
        )}
      </div>

      <div className="mt-6 flex flex-wrap gap-3">
        <Button onClick={start} disabled={!canStart}>
          Start evaluation
        </Button>
        {session !== null && (
          <Button variant="outline" onClick={close} disabled={busy}>
            Close evaluation
          </Button>
        )}
      </div>

      {error !== null && (
        <p role="alert" className="mt-4 text-sm text-destructive">
          {error}
        </p>
      )}

      {session === null ? (
        <div className="mt-8">
          <p className="text-muted-foreground">{EMPTY_STATE}</p>
          {weaveUrl !== null && (
            <a
              href={weaveUrl}
              target="_blank"
              rel="noreferrer"
              className="mt-3 inline-block text-sm font-bold underline"
            >
              Open in W&B Weave
            </a>
          )}
        </div>
      ) : (
        <div className="mt-8 rounded-md border border-border p-4">
          <p className="font-bold">{`Session open: ${session.trials} trials planned.`}</p>
          <p className="mt-2 text-sm" aria-live="polite">
            {verdict === null ? "Latest trial: no verdict yet" : verdictText(verdict)}
          </p>
          <div className="mt-4 flex flex-wrap items-end gap-3">
            <div className="flex flex-col gap-1.5">
              <label htmlFor="evaluate-stage" className="text-sm font-bold">
                Stage
              </label>
              <select
                id="evaluate-stage"
                className={SELECT_CLASS}
                value={stage}
                onChange={(event) => setStage(event.target.value)}
              >
                {STAGES.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </div>
            <Button onClick={() => override(stage)} disabled={busy}>
              Confirm
            </Button>
            <Button variant="outline" onClick={() => override(null)} disabled={busy}>
              Needs review
            </Button>
          </div>
        </div>
      )}
    </section>
  );
}
