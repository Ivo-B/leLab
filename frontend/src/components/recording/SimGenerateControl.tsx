import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { useApi } from "@/contexts/ApiContext";

// The same values the CLI generates with; the dataset is a simulated one, so the arm is never started.
const REQUEST = { episodes: 30, seed: 0, randomize: true };
const POLL_MS = 1000;

interface GenerateStatus {
  running: boolean;
  saved: number;
  episodes: number;
  error: string | null;
}

async function errorText(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string" && body.detail) return body.detail;
  } catch {
    // Fall through to the generic message.
  }
  return "Could not start the simulation.";
}

function statusText(status: GenerateStatus): string | null {
  if (status.running) return `Generating ${status.saved}/${status.episodes} episodes in simulation…`;
  if (status.episodes > 0 && status.error === null) return `Saved ${status.saved} simulated episodes.`;
  return null;
}

export default function SimGenerateControl() {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [status, setStatus] = useState<GenerateStatus | null>(null);
  const [startError, setStartError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  const refresh = useCallback(async (): Promise<GenerateStatus | null> => {
    try {
      const response = await fetchWithHeaders(`${baseUrl}/sim-generate`);
      if (!response.ok) return null;
      const next = (await response.json()) as GenerateStatus;
      setStatus(next);
      return next;
    } catch {
      return null;
    }
  }, [baseUrl, fetchWithHeaders]);

  // Pick up a job that was started before this page opened.
  useEffect(() => {
    void refresh();
  }, [refresh]);

  const running = status?.running === true;
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => void refresh(), POLL_MS);
    return () => clearInterval(timer);
  }, [running, refresh]);

  const start = async () => {
    setStarting(true);
    setStartError(null);
    try {
      const response = await fetchWithHeaders(`${baseUrl}/sim-generate`, {
        method: "POST",
        body: JSON.stringify(REQUEST),
      });
      if (!response.ok) {
        setStartError(await errorText(response));
        return;
      }
      setStatus({ running: true, saved: 0, episodes: REQUEST.episodes, error: null });
    } catch {
      setStartError("Could not reach the backend.");
    } finally {
      setStarting(false);
    }
  };

  const text = status === null ? null : statusText(status);
  const error = startError ?? status?.error ?? null;

  return (
    <div className="mt-3 flex flex-shrink-0 items-center gap-3">
      <Button
        variant="outline"
        onClick={start}
        disabled={starting || running}
        className="border-gray-600 bg-gray-800 text-white hover:bg-gray-700 hover:text-white disabled:opacity-40"
      >
        Generate in simulation
      </Button>
      {text !== null && (
        <p role="status" className="text-sm text-gray-400">
          {text}
        </p>
      )}
      {error !== null && (
        <p role="alert" className="text-sm text-red-400">
          {error}
        </p>
      )}
    </div>
  );
}
