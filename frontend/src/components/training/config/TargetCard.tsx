import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ConfigComponentProps } from "../types";
import { RunnerFlavor } from "@/lib/jobsApi";
import { COREWEAVE_NEEDS_WANDB, targetFooter, targetFromSelection } from "../target";

interface TargetCardProps extends ConfigComponentProps {
  authenticated: boolean;
  flavors: RunnerFlavor[];
  loading: boolean;
  sandboxConfigured: boolean;
}

const formatHourly = (unitCostUsd: number, unitLabel: string): string => {
  const hourly = unitLabel === "minute" ? unitCostUsd * 60 : unitCostUsd;
  return `$${hourly.toFixed(2)}/hr`;
};

const formatFlavorLine = (f: RunnerFlavor): string => {
  const accel = f.accelerator ? f.accelerator : f.cpu;
  return `${f.pretty_name} · ${accel} · ${formatHourly(f.unit_cost_usd, f.unit_label)}`;
};

const TargetCard: React.FC<TargetCardProps> = ({
  config,
  updateConfig,
  authenticated,
  flavors,
  loading,
  sandboxConfigured,
}) => {
  const target = config.target;
  const value =
    target.runner === "local"
      ? "local"
      : target.runner === "coreweave"
      ? "coreweave"
      : `hf:${target.flavor ?? ""}`;

  const handleChange = (v: string) => {
    if (v === "local" || v === "coreweave" || v.startsWith("hf:")) {
      updateConfig("target", targetFromSelection(v));
    }
  };

  const hint =
    target.runner === "coreweave"
      ? "RTX PRO 6000 on CoreWeave. The Sandbox is deleted when the job stops."
      : !sandboxConfigured
      ? "Add CWSANDBOX_API_KEY and CWSANDBOX_TRAIN_IMAGE to the backend environment."
      : !config.wandb_enable
      ? COREWEAVE_NEEDS_WANDB
      : null;

  return (
    <Card className="bg-slate-800/50 border-slate-700 rounded-xl">
      <CardHeader>
        <CardTitle className="text-white">Compute target</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div>
          <Label className="text-slate-300">Run training on</Label>
          <Select value={value} onValueChange={handleChange}>
            <SelectTrigger className="bg-slate-900 border-slate-600 text-white rounded-lg mt-1">
              <SelectValue placeholder={loading ? "Loading…" : "Select target"} />
            </SelectTrigger>
            <SelectContent className="bg-slate-800 border-slate-600 text-white">
              <SelectItem value="local">Local — your machine (free)</SelectItem>
              {flavors.map((f) => (
                <SelectItem
                  key={f.name}
                  value={`hf:${f.name}`}
                  disabled={!authenticated}
                >
                  {formatFlavorLine(f)}
                  {!authenticated && (
                    <span className="text-amber-300 ml-2 text-xs">
                      log in to HF
                    </span>
                  )}
                </SelectItem>
              ))}
              <SelectItem
                value="coreweave"
                disabled={!sandboxConfigured || !config.wandb_enable}
              >
                CoreWeave Sandbox
              </SelectItem>
            </SelectContent>
          </Select>
          {hint && <p className="text-xs text-slate-500 mt-1">{hint}</p>}
          <p className="text-xs text-slate-500 mt-1">
            {targetFooter(target.runner)}
          </p>
        </div>
      </CardContent>
    </Card>
  );
};

export default TargetCard;
