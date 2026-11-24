import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useApi } from "@/contexts/ApiContext";

const POLL_MS = 1000;
const NAME_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/;

interface GenerateStatus {
  running: boolean;
  saved: number;
  episodes: number;
  error: string | null;
  repo_id: string | null;
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
  if (status.episodes > 0 && status.error === null) {
    if (status.repo_id !== null) return `Saved ${status.saved} simulated episodes to ${status.repo_id}.`;
    return `Saved ${status.saved} simulated episodes.`;
  }
  return null;
}

// Generates a simulated dataset on the backend. It never starts the arm.
export default function SimGenerateControl() {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [episodes, setEpisodes] = useState("30");
  const [seed, setSeed] = useState("0");
  const [randomize, setRandomize] = useState(true);
  const [name, setName] = useState("");
  const [status, setStatus] = useState<GenerateStatus | null>(null);
  const [startError, setStartError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const response = await fetchWithHeaders(`${baseUrl}/sim-generate`);
      if (response.ok) setStatus((await response.json()) as GenerateStatus);
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
    const timer = setInterval(() => void refresh(), POLL_MS);
    return () => clearInterval(timer);
  }, [running, refresh]);

  const episodesValid = Number.isInteger(Number(episodes)) && Number(episodes) >= 1;
  const seedValid = Number.isInteger(Number(seed)) && seed.trim() !== "";
  const nameValid = name.trim() === "" || NAME_PATTERN.test(name.trim());
  const canStart = !starting && !running && episodesValid && seedValid && nameValid;

  const start = async () => {
    setStarting(true);
    setStartError(null);
    const request = {
      episodes: Number(episodes),
      seed: Number(seed),
      randomize,
      ...(name.trim() ? { name: name.trim() } : {}),
    };
    try {
      const response = await fetchWithHeaders(`${baseUrl}/sim-generate`, {
        method: "POST",
        body: JSON.stringify(request),
      });
      if (!response.ok) {
        setStartError(await errorText(response));
        return;
      }
      const started = (await response.json()) as { repo_id?: string };
      setStatus({
        running: true,
        saved: 0,
        episodes: request.episodes,
        error: null,
        repo_id: started.repo_id ?? null,
      });
    } catch {
      setStartError("Could not reach the backend.");
    } finally {
      setStarting(false);
    }
  };

  const text = status === null ? null : statusText(status);
  const error = startError ?? status?.error ?? null;

  return (
    <div className="mt-6">
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="flex flex-col gap-1.5">
          <label htmlFor="sim-episodes" className="text-sm font-bold">
            Episodes
          </label>
          <Input
            id="sim-episodes"
            type="number"
            min={1}
            value={episodes}
            disabled={running}
            onChange={(event) => setEpisodes(event.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1.5">
          <label htmlFor="sim-seed" className="text-sm font-bold">
            Seed
          </label>
          <Input
            id="sim-seed"
            type="number"
            value={seed}
            disabled={running}
            onChange={(event) => setSeed(event.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1.5 sm:col-span-2">
          <label htmlFor="sim-name" className="text-sm font-bold">
            Dataset name (optional)
          </label>
          <Input
            id="sim-name"
            value={name}
            placeholder="so101-chess-sim-30ep-s0"
            disabled={running}
            onChange={(event) => setName(event.target.value)}
          />
        </div>
      </div>
      <label htmlFor="sim-randomize" className="mt-4 flex items-center gap-2 text-sm font-bold">
        <input
          id="sim-randomize"
          type="checkbox"
          checked={randomize}
          disabled={running}
          onChange={(event) => setRandomize(event.target.checked)}
        />
        Randomize the scene
      </label>
      <div className="mt-4 flex flex-wrap items-center gap-3">
        <Button onClick={start} disabled={!canStart}>
          Generate in simulation
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
