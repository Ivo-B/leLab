import { useEffect, useState } from "react";
import { useApi } from "@/contexts/ApiContext";

interface ChessCheck {
  checked: boolean;
  problems: string[];
}

// Warns, never blocks: a dataset that fails the check can still be trained, it just is not chess transfer data.
export default function ChessDatasetCheck({ repoId }: { repoId: string }) {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [check, setCheck] = useState<ChessCheck | null>(null);

  useEffect(() => {
    let cancelled = false;
    setCheck(null);
    if (!repoId) return;
    fetchWithHeaders(`${baseUrl}/datasets/chess-check?repo_id=${encodeURIComponent(repoId)}`)
      .then((response) => (response.ok ? response.json() : null))
      .then((body) => {
        if (!cancelled) setCheck(body as ChessCheck | null);
      })
      .catch(() => {
        if (!cancelled) setCheck(null);
      });
    return () => {
      cancelled = true;
    };
  }, [repoId, baseUrl, fetchWithHeaders]);

  if (!check?.checked) return null;
  if (check.problems.length === 0) {
    return <p className="mt-2 text-sm text-muted-foreground">Chess transfer data: markers, sidecar and task string present.</p>;
  }
  return (
    <div role="alert" className="mt-2 text-sm text-amber-400">
      <p className="font-bold">Not chess transfer data:</p>
      <ul className="list-disc pl-5">
        {check.problems.map((problem) => (
          <li key={problem}>{problem}</li>
        ))}
      </ul>
    </div>
  );
}
