export function targetFromSelection(value: string): {
  runner: "local" | "hf_cloud" | "coreweave";
  flavor?: string;
} {
  if (value === "local") return { runner: "local" };
  if (value === "coreweave") return { runner: "coreweave" };
  return { runner: "hf_cloud", flavor: value.slice("hf:".length) };
}

export const COREWEAVE_NEEDS_WANDB =
  "CoreWeave Sandbox training needs Weights & Biases tracking, so the checkpoint is saved before the Sandbox is deleted.";

export function targetFooter(runner: "local" | "hf_cloud" | "coreweave"): string {
  if (runner === "coreweave") return "The checkpoint is logged to Weights & Biases.";
  return "Cost shown is per running hour. Final policy uploads to your HF account when training completes.";
}

export function setupSandboxDetail(
  forgeConfigured: boolean,
  sandboxConfigured: boolean,
  entity: string,
): string {
  if (!forgeConfigured) {
    return "Add WANDB_API_KEY and WANDB_ENTITY to the backend environment.";
  }
  if (!sandboxConfigured) {
    return `Weights & Biases is connected as ${entity}. Add CWSANDBOX_API_KEY and CWSANDBOX_TRAIN_IMAGE to run training on a GPU Sandbox.`;
  }
  return `Weights & Biases is connected as ${entity}. GPU Sandbox training is available.`;
}
