import { useEffect, useState } from "react";
import { CircleAlert, CircleCheck } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
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

const chipClass = "inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs";
const readyClass = `${chipClass} border-border bg-card text-foreground`;
const missingClass = `${chipClass} border-border bg-card text-muted-foreground hover:text-foreground`;

function runsLabel(count: number): string {
  if (count <= 0) return "CoreWeave";
  return `CoreWeave · ${count} ${count === 1 ? "run" : "runs"}`;
}

export function ForgeStatusChips({ coreweaveRuns = 0 }: { coreweaveRuns?: number }) {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { pathname } = useLocation();
  const [body, setBody] = useState<ForgeBody | "error" | null>(null);
  const [open, setOpen] = useState<Which | null>(null);

  // Ask again on every route change so keys saved on Setup show up without a reload.
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
  }, [baseUrl, fetchWithHeaders, pathname]);

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
          role="status"
          className={readyClass}
          aria-label={`Weights & Biases connected as ${entity}`}
        >
          <CircleCheck size={14} aria-hidden />
          {entity}
        </span>
      ) : (
        <button
          type="button"
          onClick={() => setOpen("wandb")}
          className={missingClass}
          aria-label="Weights & Biases not configured"
        >
          <CircleAlert size={14} aria-hidden />
          Weights & Biases · not set up
        </button>
      )}
      {coreweaveReady ? (
        <span
          role="status"
          className={readyClass}
          aria-label={coreweaveRuns > 0 ? runsLabel(coreweaveRuns) : "CoreWeave ready"}
        >
          <CircleCheck size={14} aria-hidden />
          {runsLabel(coreweaveRuns)}
        </span>
      ) : (
        <button
          type="button"
          onClick={() => setOpen("coreweave")}
          className={missingClass}
          aria-label="CoreWeave not configured"
        >
          <CircleAlert size={14} aria-hidden />
          CoreWeave · not set up
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
