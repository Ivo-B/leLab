import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useApi } from "@/contexts/ApiContext";
import { PlaceholderPage } from "@/pages/PlaceholderPage";

const PLACEHOLDER_BODY = "The order-picking agent arrives in a later milestone.";
const EMPTY_STATE = "Type an order, then the agent picks each part into the tray.";
const SANDBOX_HINT = "Add CWSANDBOX_API_KEY and CWSANDBOX_TRAIN_IMAGE to the backend environment.";
const POLL_MS = 2000;
const TILES = ["Wrist", "Top", "Side"] as const;
const SELECT_CLASS =
  "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50";

type ForgeState = "loading" | "off" | { sandbox: boolean };

interface Order {
  status: string;
  counts: Record<string, number>;
  traceUrl: string | null;
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

function toCounts(value: unknown): Record<string, number> {
  const counts: Record<string, number> = {};
  if (typeof value === "object" && value !== null) {
    for (const [name, count] of Object.entries(value)) {
      if (typeof count === "number") counts[name] = count;
    }
  }
  return counts;
}

export default function Orders() {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [forge, setForge] = useState<ForgeState>("loading");

  const [text, setText] = useState("");
  const [family, setFamily] = useState("smolvla");
  const [host, setHost] = useState("local");

  const [order, setOrder] = useState<Order | null>(null);
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

  const running = order?.status === "running";

  // Poll the order only while it is running; a finished order stops the timer.
  useEffect(() => {
    if (!running) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const response = await fetchWithHeaders(`${baseUrl}/orders/current`);
        if (!response.ok) return;
        const body = (await response.json()) as {
          status?: unknown;
          counts?: unknown;
          trace_url?: unknown;
        };
        if (cancelled || typeof body.status !== "string" || body.status === "idle") return;
        const status = body.status;
        const traceUrl = typeof body.trace_url === "string" && body.trace_url !== "" ? body.trace_url : null;
        setOrder((previous) => ({
          status,
          counts: body.counts === undefined ? (previous?.counts ?? {}) : toCounts(body.counts),
          traceUrl,
        }));
      } catch {
        // Keep the last known state; the next tick retries.
      }
    };
    void poll();
    const timer = setInterval(() => void poll(), POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [running, baseUrl, fetchWithHeaders]);

  const post = useCallback(
    async (path: string, body: unknown): Promise<Response> =>
      fetchWithHeaders(`${baseUrl}${path}`, { method: "POST", body: JSON.stringify(body) }),
    [baseUrl, fetchWithHeaders],
  );

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      const response = await post("/orders", {
        text: text.trim(),
        policy_family: family,
        policy_host: host,
      });
      if (!response.ok) {
        setError(await errorText(response, "Could not start the order."));
        return;
      }
      const body = (await response.json()) as { status?: unknown; counts?: unknown };
      setOrder({
        status: typeof body.status === "string" ? body.status : "running",
        counts: toCounts(body.counts),
        traceUrl: null,
      });
    } catch {
      setError("Could not reach the backend.");
    } finally {
      setBusy(false);
    }
  };

  const stop = async () => {
    setBusy(true);
    setError(null);
    try {
      const response = await post("/orders/current/stop", {});
      if (!response.ok) setError(await errorText(response, "Could not stop the order."));
    } catch {
      setError("Could not reach the backend.");
    } finally {
      setBusy(false);
    }
  };

  if (forge === "loading") {
    return (
      <section className="mx-auto max-w-3xl p-8">
        <h1 className="font-extrabold text-3xl">Orders</h1>
      </section>
    );
  }
  if (forge === "off") {
    return <PlaceholderPage title="Orders" body={PLACEHOLDER_BODY} />;
  }

  const canStart = !busy && !running && text.trim() !== "";

  return (
    <section className="mx-auto max-w-3xl p-8">
      <h1 className="font-extrabold text-3xl">Orders</h1>

      <div className="mt-6 grid gap-4 sm:grid-cols-2">
        <div className="sm:col-span-2">
          <Field id="orders-text" label="Order">
            <Input
              id="orders-text"
              value={text}
              disabled={running}
              onChange={(event) => setText(event.target.value)}
            />
          </Field>
        </div>
        <Field id="orders-family" label="Policy family">
          <select
            id="orders-family"
            className={SELECT_CLASS}
            value={family}
            disabled={running}
            onChange={(event) => setFamily(event.target.value)}
          >
            <option value="smolvla">SmolVLA</option>
            <option value="act">ACT</option>
          </select>
        </Field>
        <Field id="orders-host" label="Policy host">
          <select
            id="orders-host"
            className={SELECT_CLASS}
            value={host}
            disabled={running}
            onChange={(event) => setHost(event.target.value)}
          >
            <option value="local">Local</option>
            <option value="coreweave" disabled={!forge.sandbox}>
              CoreWeave
            </option>
          </select>
          {!forge.sandbox && <p className="text-sm text-muted-foreground">{SANDBOX_HINT}</p>}
        </Field>
      </div>

      <div className="mt-6 flex flex-wrap gap-3">
        <Button onClick={start} disabled={!canStart}>
          Start order
        </Button>
        {running && (
          <Button variant="outline" onClick={stop} disabled={busy}>
            Stop
          </Button>
        )}
      </div>

      {error !== null && (
        <p role="alert" className="mt-4 text-sm text-destructive">
          {error}
        </p>
      )}

      {order === null ? (
        <p className="mt-8 text-muted-foreground">{EMPTY_STATE}</p>
      ) : (
        <div className="mt-8 rounded-md border border-border p-4">
          <p className="font-bold" aria-live="polite">{`Status: ${order.status}`}</p>
          <ul className="mt-2 text-sm">
            {Object.entries(order.counts).map(([name, count]) => (
              <li key={name}>{`${name}: ${count}`}</li>
            ))}
          </ul>
          {order.traceUrl !== null && (
            <a
              href={order.traceUrl}
              target="_blank"
              rel="noreferrer"
              className="mt-3 inline-block text-sm font-bold underline"
            >
              Open trace in Agent Lens
            </a>
          )}
        </div>
      )}

      <div className="mt-8 grid gap-4 sm:grid-cols-3">
        {TILES.map((name) => (
          <div
            key={name}
            role="group"
            aria-label={name}
            className="flex aspect-video flex-col items-center justify-center rounded-md border border-border p-2"
          >
            <p className="text-sm font-bold">{name}</p>
            <p className="text-sm text-muted-foreground">No frame yet</p>
          </div>
        ))}
      </div>
    </section>
  );
}
