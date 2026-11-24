import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { PlaceholderPage } from "@/pages/PlaceholderPage";

const PLACEHOLDER_BODY =
  "Connect Forge (Weights & Biases) to record the operator's outcome for each transfer of the plan and log it to W&B Weave. Simulation evaluation below works without it.";
const EMPTY_STATE =
  "Choose a checkpoint, then run the 20-transfer plan on the arm and record how each transfer ends.";
const SANDBOX_HINT = "Add CWSANDBOX_API_KEY and CWSANDBOX_TRAIN_IMAGE to the backend environment.";
const COREWEAVE_HOST_NOTE =
  "The CoreWeave policy host is not wired yet. Trials run on this machine and are logged as local in Weave, with CoreWeave kept as the requested host.";

const OUTCOMES = ["success", "approach", "grasp", "drop", "place", "topple", "disturbed"] as const;
const SELECT_CLASS =
  "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50";

type ForgeState = "loading" | "off" | { sandbox: boolean };

interface Session {
  trials: number;
  finished: number;
  next_transfer: [string, string] | null;
  next_class: string | null;
}

interface Verdict {
  transfer: [string, string];
  class: string;
  outcome: string;
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

interface SimStatus {
  running: boolean;
  report: { success_rate: number; trials: unknown[] } | null;
  error: string | null;
}

const SIM_POLL_MS = 1000;

function simResultText(status: SimStatus): string | null {
  if (status.running) return "Evaluating in simulation…";
  if (status.report === null) return null;
  const percent = Math.round(status.report.success_rate * 100);
  return `Simulation: ${percent}% of ${status.report.trials.length} trials succeeded.`;
}

// Runs a checkpoint on the sim transfer plan. It never starts the arm and does not need Forge.
function SimEvaluate() {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [checkpoint, setCheckpoint] = useState("");
  const [seeds, setSeeds] = useState("3");
  const [status, setStatus] = useState<SimStatus | null>(null);
  const [startError, setStartError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const response = await fetchWithHeaders(`${baseUrl}/sim-evaluate`);
      if (response.ok) setStatus((await response.json()) as SimStatus);
    } catch {
      // The next poll tries again.
    }
  }, [baseUrl, fetchWithHeaders]);

  // Pick up a job that was started before this page opened.
  useEffect(() => {
    void refresh();
  }, [refresh]);

  const running = status?.running === true;
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => void refresh(), SIM_POLL_MS);
    return () => clearInterval(timer);
  }, [running, refresh]);

  const seedsValid = Number.isInteger(Number(seeds)) && Number(seeds) >= 1;
  const canStart = !starting && !running && checkpoint.trim() !== "" && seedsValid;

  const start = async () => {
    setStarting(true);
    setStartError(null);
    try {
      const response = await fetchWithHeaders(`${baseUrl}/sim-evaluate`, {
        method: "POST",
        body: JSON.stringify({ checkpoint: checkpoint.trim(), seeds: Number(seeds) }),
      });
      if (!response.ok) {
        setStartError(await errorText(response, "Could not start the simulation."));
        return;
      }
      setStatus({ running: true, report: null, error: null });
    } catch {
      setStartError("Could not reach the backend.");
    } finally {
      setStarting(false);
    }
  };

  const text = status === null ? null : simResultText(status);
  const error = startError ?? status?.error ?? null;

  return (
    <div className="mt-10 border-t border-border pt-6">
      <div className="grid gap-4 sm:grid-cols-2">
        <Field id="sim-checkpoint" label="Checkpoint path">
          <Input
            id="sim-checkpoint"
            placeholder="…/checkpoints/last/pretrained_model"
            value={checkpoint}
            disabled={running}
            onChange={(event) => setCheckpoint(event.target.value)}
          />
        </Field>
        <Field id="sim-seeds" label="Seeds">
          <Input
            id="sim-seeds"
            type="number"
            min={1}
            value={seeds}
            disabled={running}
            onChange={(event) => setSeeds(event.target.value)}
          />
        </Field>
      </div>
      <div className="mt-4 flex flex-wrap items-center gap-3">
        <Button variant="outline" onClick={start} disabled={!canStart}>
          In simulation
        </Button>
        {text !== null && (
          <p role="status" className="text-sm">
            {text}
          </p>
        )}
        {error !== null && (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        )}
      </div>
    </div>
  );
}

function nextText(session: Session): string {
  if (session.next_transfer === null) return `All ${session.trials} trials recorded.`;
  const [source, target] = session.next_transfer;
  return `Next: ${source} → ${target} (${session.next_class}) · ${session.finished + 1} of ${session.trials}`;
}

function verdictText(verdict: Verdict): string {
  const [source, target] = verdict.transfer;
  return `Last: ${source} → ${target} ${verdict.outcome} (${verdict.verdict_source})`;
}

export default function Evaluate() {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [forge, setForge] = useState<ForgeState>("loading");

  const [family, setFamily] = useState("smolvla");
  const [version, setVersion] = useState("candidate");
  const [host, setHost] = useState("local");
  const [rounds, setRounds] = useState("1");

  const [session, setSession] = useState<Session | null>(null);
  // A session the server opened but this page could not read yet; starting again would be refused.
  const [unread, setUnread] = useState(false);
  const [verdict, setVerdict] = useState<Verdict | null>(null);
  const [outcome, setOutcome] = useState<string>(OUTCOMES[0]);
  const [weaveUrl, setWeaveUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [promoted, setPromoted] = useState<string | null>(null);
  const { toast } = useToast();

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

  const refresh = useCallback(async (): Promise<string | null> => {
    const response = await fetchWithHeaders(`${baseUrl}/evaluations/current`);
    if (!response.ok) return await errorText(response, "Could not read the evaluation.");
    const body = (await response.json()) as Partial<Session> & { open?: boolean };
    if (body.open === true) {
      setSession({
        trials: body.trials ?? 0,
        finished: body.finished ?? 0,
        next_transfer: body.next_transfer ?? null,
        next_class: body.next_class ?? null,
      });
    } else {
      setSession(null);
      setVerdict(null);
    }
    setUnread(false);
    return null;
  }, [baseUrl, fetchWithHeaders]);

  // Show a session that was already open before this page loaded.
  useEffect(() => {
    if (forge === "loading" || forge === "off") return;
    void refresh().catch(() => undefined);
  }, [forge, refresh]);

  const start = async () => {
    setBusy(true);
    setError(null);
    setWeaveUrl(null);
    try {
      const response = await post("/evaluations", {
        policy_family: family,
        version,
        policy_host: host,
        rounds: Number(rounds),
      });
      if (!response.ok) {
        setError(await errorText(response, "Could not start the evaluation."));
        return;
      }
      setVerdict(null);
      setUnread(true);
      setError(await refresh());
    } catch {
      setError("Could not reach the backend.");
    } finally {
      setBusy(false);
    }
  };

  const reread = async () => {
    setBusy(true);
    setError(null);
    try {
      setError(await refresh());
    } catch {
      setError("Could not reach the backend.");
    } finally {
      setBusy(false);
    }
  };

  const record = async () => {
    setBusy(true);
    setError(null);
    try {
      const response = await post("/evaluations/current/trial", { outcome, duration_s: 0 });
      if (!response.ok) {
        setError(await errorText(response, "Could not record the outcome."));
        return;
      }
      setVerdict((await response.json()) as Verdict);
      setError(await refresh());
    } catch {
      setError("Could not reach the backend.");
    } finally {
      setBusy(false);
    }
  };

  const correct = async () => {
    setBusy(true);
    setError(null);
    try {
      const response = await post("/evaluations/current/override", { outcome });
      if (!response.ok) {
        setError(await errorText(response, "Could not correct the outcome."));
        return;
      }
      setVerdict((await response.json()) as Verdict);
      setError(await refresh());
    } catch {
      setError("Could not reach the backend.");
    } finally {
      setBusy(false);
    }
  };

  const promote = async () => {
    setBusy(true);
    setError(null);
    setPromoted(null);
    try {
      const response = await post("/registry/promote", { policy_family: family });
      if (!response.ok) {
        setError(await errorText(response, "Could not promote the candidate."));
        return;
      }
      const body = (await response.json()) as { collection: string; version: string; to_alias?: string };
      const message = `Promoted ${body.collection} ${body.version} to ${body.to_alias ?? "production"}.`;
      setPromoted(message);
      toast({ title: "Promoted to production", description: `${body.collection} ${body.version}` });
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
    return (
      <>
        <PlaceholderPage title="Evaluate" body={PLACEHOLDER_BODY} />
        <div className="mx-auto max-w-3xl px-8 pb-8">
          <SimEvaluate />
        </div>
      </>
    );
  }

  const roundsValid =
    rounds.trim() !== "" && Number.isInteger(Number(rounds)) && Number(rounds) >= 1 && Number(rounds) <= 5;
  const canPromote = !busy;
  const canStart = !busy && session === null && !unread && roundsValid;

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
          {host === "coreweave" && <p className="text-sm text-muted-foreground">{COREWEAVE_HOST_NOTE}</p>}
        </Field>
        <Field id="evaluate-rounds" label="Rounds">
          <Input
            id="evaluate-rounds"
            type="number"
            min={1}
            max={5}
            value={rounds}
            disabled={session !== null}
            onChange={(event) => setRounds(event.target.value)}
          />
        </Field>
      </div>

      <div className="mt-6 flex flex-wrap gap-3">
        <Button onClick={start} disabled={!canStart}>
          Start evaluation
        </Button>
        <Button variant="outline" onClick={promote} disabled={!canPromote}>
          Promote candidate to production
        </Button>
        {session !== null && (
          <Button variant="outline" onClick={close} disabled={busy}>
            Close evaluation
          </Button>
        )}
        {session === null && unread && (
          <Button variant="outline" onClick={reread} disabled={busy}>
            Refresh
          </Button>
        )}
      </div>

      {error !== null && (
        <p role="alert" className="mt-4 text-sm text-destructive">
          {error}
        </p>
      )}
      {promoted !== null && (
        <p role="status" className="mt-4 text-sm">
          {promoted}
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
          <p className="font-bold">{nextText(session)}</p>
          <p className="mt-2 text-sm" aria-live="polite">
            {verdict !== null
              ? verdictText(verdict)
              : session.finished > 0
                ? "Latest trial: recorded before this page opened"
                : "Latest trial: nothing recorded yet"}
          </p>
          <div className="mt-4 flex flex-wrap items-end gap-3">
            <div className="flex flex-col gap-1.5">
              <label htmlFor="evaluate-outcome" className="text-sm font-bold">
                Outcome
              </label>
              <select
                id="evaluate-outcome"
                className={SELECT_CLASS}
                value={outcome}
                onChange={(event) => setOutcome(event.target.value)}
              >
                {OUTCOMES.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </div>
            <Button onClick={record} disabled={busy || session.next_transfer === null}>
              Record outcome
            </Button>
            <Button variant="outline" onClick={correct} disabled={busy || (verdict === null && session.finished === 0)}>
              Correct last
            </Button>
          </div>
        </div>
      )}

      <SimEvaluate />
    </section>
  );
}
