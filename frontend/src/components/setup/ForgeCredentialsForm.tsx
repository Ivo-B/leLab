import { FormEvent, type ReactNode, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useApi } from "@/contexts/ApiContext";
import { setupSandboxDetail } from "@/components/training/target";
import { cn } from "@/lib/utils";
import type { RegistryStatusData } from "./RegistryStatus";

export type ForgeStatus = {
  forge_configured: boolean;
  sandbox_configured?: boolean;
  entity: string | null;
  project: string | null;
  sandbox_image?: string | null;
  registry?: RegistryStatusData;
  fields?: {
    api_key?: boolean;
    entity?: boolean;
    project?: boolean;
    sandbox_api_key?: boolean;
    sandbox_image?: boolean;
  };
};

type Props = {
  status: ForgeStatus | null;
  onSaved: (status: ForgeStatus) => void;
};

function FieldLabel({
  htmlFor,
  children,
  isSet,
}: {
  htmlFor: string;
  children: ReactNode;
  isSet: boolean;
}) {
  return (
    <div className="flex items-center justify-between gap-2">
      <Label htmlFor={htmlFor}>{children}</Label>
      {isSet ? (
        <span className="rounded-sm bg-emerald-500/15 px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-emerald-700 dark:text-emerald-300">
          Set
        </span>
      ) : (
        <span className="rounded-sm bg-amber-500/15 px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-amber-800 dark:text-amber-200">
          Missing
        </span>
      )}
    </div>
  );
}

function fieldClass(isSet: boolean) {
  return cn(
    isSet && "border-emerald-500/60 bg-emerald-500/5 focus-visible:ring-emerald-500/40",
  );
}

export default function ForgeCredentialsForm({ status, onSaved }: Props) {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [apiKey, setApiKey] = useState("");
  const [entity, setEntity] = useState("");
  const [project, setProject] = useState("so101-kitting");
  const [sandboxApiKey, setSandboxApiKey] = useState("");
  const [sandboxImage, setSandboxImage] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fields = status?.fields;
  const apiKeySet = fields?.api_key === true;
  const entitySet = fields?.entity === true;
  const projectSet = fields?.project === true;
  const sandboxKeySet = fields?.sandbox_api_key === true;
  const sandboxImageSet = fields?.sandbox_image === true;

  useEffect(() => {
    if (!status) return;
    setEntity(status.entity ?? "");
    setProject(status.project ?? "so101-kitting");
    setSandboxImage(status.sandbox_image ?? "");
  }, [status]);

  const detail = status
    ? setupSandboxDetail(
        status.forge_configured,
        status.sandbox_configured === true,
        status.entity ?? "",
      )
    : setupSandboxDetail(false, false, "");

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setSaving(true);
    setError(null);
    const body: Record<string, string> = {};
    if (apiKey.trim()) body.api_key = apiKey.trim();
    if (entity.trim()) body.entity = entity.trim();
    if (project.trim()) body.project = project.trim();
    if (sandboxApiKey.trim()) body.sandbox_api_key = sandboxApiKey.trim();
    if (sandboxImage.trim()) body.sandbox_image = sandboxImage.trim();
    try {
      const response = await fetchWithHeaders(`${baseUrl}/forge-credentials`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        const message =
          typeof payload.detail === "string"
            ? payload.detail
            : "Could not save Forge credentials";
        throw new Error(message);
      }
      setApiKey("");
      setSandboxApiKey("");
      onSaved(payload as ForgeStatus);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-3">
      <span className="font-bold">Connect Forge</span>
      <p className="text-sm text-muted-foreground">{detail}</p>
      <form className="grid gap-3 sm:grid-cols-2" onSubmit={onSubmit}>
        <div className="space-y-1.5 sm:col-span-2">
          <FieldLabel htmlFor="wandb-api-key" isSet={apiKeySet}>
            W&amp;B API key
          </FieldLabel>
          <Input
            id="wandb-api-key"
            type="password"
            autoComplete="off"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            className={fieldClass(apiKeySet)}
            placeholder={apiKeySet ? "•••••••• (leave blank to keep)" : "WANDB_API_KEY"}
          />
        </div>
        <div className="space-y-1.5">
          <FieldLabel htmlFor="wandb-entity" isSet={entitySet}>
            W&amp;B entity
          </FieldLabel>
          <Input
            id="wandb-entity"
            value={entity}
            onChange={(e) => setEntity(e.target.value)}
            className={fieldClass(entitySet)}
            placeholder="WANDB_ENTITY"
          />
        </div>
        <div className="space-y-1.5">
          <FieldLabel htmlFor="wandb-project" isSet={projectSet}>
            W&amp;B project
          </FieldLabel>
          <Input
            id="wandb-project"
            value={project}
            onChange={(e) => setProject(e.target.value)}
            className={fieldClass(projectSet)}
            placeholder="so101-kitting"
          />
        </div>
        <div className="space-y-1.5 sm:col-span-2">
          <FieldLabel htmlFor="cwsandbox-api-key" isSet={sandboxKeySet}>
            CoreWeave Sandbox API key
          </FieldLabel>
          <Input
            id="cwsandbox-api-key"
            type="password"
            autoComplete="off"
            value={sandboxApiKey}
            onChange={(e) => setSandboxApiKey(e.target.value)}
            className={fieldClass(sandboxKeySet)}
            placeholder={sandboxKeySet ? "•••••••• (leave blank to keep)" : "CWSANDBOX_API_KEY"}
          />
        </div>
        <div className="space-y-1.5 sm:col-span-2">
          <FieldLabel htmlFor="cwsandbox-image" isSet={sandboxImageSet}>
            Sandbox train image
          </FieldLabel>
          <Input
            id="cwsandbox-image"
            value={sandboxImage}
            onChange={(e) => setSandboxImage(e.target.value)}
            className={fieldClass(sandboxImageSet)}
            placeholder="CWSANDBOX_TRAIN_IMAGE"
          />
        </div>
        {error ? <p className="text-sm text-destructive sm:col-span-2">{error}</p> : null}
        <div className="sm:col-span-2">
          <Button type="submit" disabled={saving}>
            {saving ? "Saving…" : "Save to .env"}
          </Button>
        </div>
      </form>
    </div>
  );
}
