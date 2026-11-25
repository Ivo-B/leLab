import { useEffect, useRef, useState } from "react";
import { useApi } from "@/contexts/ApiContext";

interface ChessCheck {
  checked: boolean;
  problems: string[];
  cameras?: string[];
}

/** What a chess policy reads unless told otherwise; `side` is recorded for experiments. */
const DEFAULT_POLICY_CAMERAS = ["top", "wrist"];

/**
 * The `policy_cameras` to send: nothing when the selection is every camera of the dataset (LeRobot then reads them
 * all by itself), else the selected cameras in the dataset's order.
 */
function policyCamerasRequest(selected: string[], datasetCameras: string[]): string[] | undefined {
  const chosen = datasetCameras.filter((camera) => selected.includes(camera));
  return chosen.length === datasetCameras.length ? undefined : chosen;
}

interface ChessDatasetCheckProps {
  repoId: string;
  /** Called with the cameras the policy should read, or undefined to let LeRobot use the whole dataset. */
  onPolicyCameras?: (cameras: string[] | undefined) => void;
}

// Warns, never blocks: a dataset that fails the check can still be trained, it just is not chess transfer data.
export default function ChessDatasetCheck({ repoId, onPolicyCameras }: ChessDatasetCheckProps) {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [check, setCheck] = useState<ChessCheck | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const report = useRef(onPolicyCameras);
  report.current = onPolicyCameras;

  useEffect(() => {
    let cancelled = false;
    setCheck(null);
    report.current?.(undefined);
    if (!repoId) return;
    fetchWithHeaders(`${baseUrl}/datasets/chess-check?repo_id=${encodeURIComponent(repoId)}`)
      .then((response) => (response.ok ? response.json() : null))
      .then((body) => {
        if (cancelled) return;
        const result = body as ChessCheck | null;
        setCheck(result);
        if (result?.checked && result.problems.length === 0 && result.cameras) {
          const initial = result.cameras.filter((camera) => DEFAULT_POLICY_CAMERAS.includes(camera));
          setSelected(initial);
          report.current?.(policyCamerasRequest(initial, result.cameras));
        }
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
    const cameras = check.cameras ?? [];
    return (
      <div className="mt-2 text-sm text-muted-foreground">
        <p>Chess transfer data: markers, sidecar and task string present.</p>
        {cameras.length > 1 ? (
          <fieldset className="mt-2 flex flex-wrap items-center gap-4">
            <legend className="sr-only">Cameras the policy reads</legend>
            <span>Policy cameras:</span>
            {cameras.map((camera) => {
              const checked = selected.includes(camera);
              return (
                <label key={camera} className="flex items-center gap-1">
                  <input
                    type="checkbox"
                    checked={checked}
                    // A policy needs at least one picture.
                    disabled={checked && selected.length === 1}
                    onChange={() => {
                      const next = checked ? selected.filter((c) => c !== camera) : [...selected, camera];
                      setSelected(next);
                      report.current?.(policyCamerasRequest(next, cameras));
                    }}
                  />
                  {camera}
                </label>
              );
            })}
          </fieldset>
        ) : null}
      </div>
    );
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
