import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useApi } from "@/contexts/ApiContext";

const WANDB_HINT = "Add WANDB_API_KEY and WANDB_ENTITY on Setup.";
const COREWEAVE_HINT = "Add CWSANDBOX_API_KEY and CWSANDBOX_TRAIN_IMAGE on Setup.";

type ForgeBody = {
  forge_configured?: unknown;
  sandbox_configured?: unknown;
  entity?: unknown;
};

type Which = "wandb" | "coreweave";

function Dot({ ready }: { ready: boolean }) {
  return (
    <span
      aria-hidden
      className={`h-2 w-2 rounded-full ${ready ? "bg-emerald-400" : "bg-amber-400"}`}
    />
  );
}

const chipClass =
  "inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs";

function runsLabel(count: number): string {
  if (count <= 0) return "CoreWeave";
  return `CoreWeave · ${count} ${count === 1 ? "run" : "runs"}`;
}

export function ForgeStatusChips({ coreweaveRuns = 0 }: { coreweaveRuns?: number }) {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [body, setBody] = useState<ForgeBody | "error" | null>(null);
  const [open, setOpen] = useState<Which | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchWithHeaders(`${baseUrl}/forge-status`)
      .then(async (response) => {
        if (!response.ok) throw new Error(`forge-status ${response.status}`);
        return (await response.json()) as ForgeBody;
      })
      .then((next) => {
        if (!cancelled) setBody(next);
      })
      .catch(() => {
        if (!cancelled) setBody("error");
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders]);

  if (body === null) {
    return <span className="text-xs text-muted-foreground">Checking Forge…</span>;
  }

  const status = body === "error" ? {} : body;
  const entity = typeof status.entity === "string" ? status.entity : "";
  const wandbReady = status.forge_configured === true && entity.length > 0;
  const coreweaveReady = status.sandbox_configured === true;

  return (
    <>
      {wandbReady ? (
        <span
          className={`${chipClass} border-border bg-card text-foreground`}
          aria-label={`Weights & Biases connected as ${entity}`}
        >
          <Dot ready />
          {entity}
        </span>
      ) : (
        <button
          type="button"
          onClick={() => setOpen("wandb")}
          className={`${chipClass} border-amber-500/60 bg-amber-500/10 text-foreground hover:bg-amber-500/20`}
          aria-label="Weights & Biases not configured"
        >
          <Dot ready={false} />
          Weights & Biases
        </button>
      )}
      {coreweaveReady ? (
        <span
          className={`${chipClass} border-border bg-card text-foreground`}
          aria-label={coreweaveRuns > 0 ? runsLabel(coreweaveRuns) : "CoreWeave ready"}
        >
          <Dot ready />
          {runsLabel(coreweaveRuns)}
        </span>
      ) : (
        <button
          type="button"
          onClick={() => setOpen("coreweave")}
          className={`${chipClass} border-amber-500/60 bg-amber-500/10 text-foreground hover:bg-amber-500/20`}
          aria-label="CoreWeave not configured"
        >
          <Dot ready={false} />
          CoreWeave
        </button>
      )}
      <Dialog open={open !== null} onOpenChange={(next) => { if (!next) setOpen(null); }}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{open === "coreweave" ? "CoreWeave" : "Weights & Biases"}</DialogTitle>
            <DialogDescription>{open === "coreweave" ? COREWEAVE_HINT : WANDB_HINT}</DialogDescription>
          </DialogHeader>
          <Link to="/setup" className="text-sm font-medium text-primary underline">
            Open Setup
          </Link>
        </DialogContent>
      </Dialog>
    </>
  );
}
