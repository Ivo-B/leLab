import { describe, expect, it } from "vitest";
import { setupSandboxDetail, targetFromSelection } from "./target";

describe("targetFromSelection", () => {
  it("maps coreweave without a flavor", () => {
    expect(targetFromSelection("coreweave")).toEqual({ runner: "coreweave" });
  });
  it("maps local", () => {
    expect(targetFromSelection("local")).toEqual({ runner: "local" });
  });
  it("maps an HF flavor", () => {
    expect(targetFromSelection("hf:a10g-small")).toEqual({
      runner: "hf_cloud",
      flavor: "a10g-small",
    });
  });
});

describe("setupSandboxDetail", () => {
  const forgeOff = setupSandboxDetail(false, false, "");
  const sandboxOff = setupSandboxDetail(true, false, "acme");
  const ready = setupSandboxDetail(true, true, "acme");

  it("asks for the W&B env when Forge is not configured", () => {
    expect(forgeOff).toBe("Add WANDB_API_KEY and WANDB_ENTITY to the backend environment.");
  });
  it("asks for the Sandbox env when only Forge is configured", () => {
    expect(sandboxOff).toBe(
      "Weights & Biases is connected as acme. Add CWSANDBOX_API_KEY and CWSANDBOX_TRAIN_IMAGE to run training on a GPU Sandbox.",
    );
  });
  it("says Sandbox training is available when both are configured", () => {
    expect(ready).toBe("Weights & Biases is connected as acme. GPU Sandbox training is available.");
  });
  it("never mentions key values", () => {
    for (const s of [forgeOff, sandboxOff, ready]) {
      expect(s).not.toContain("API key values");
      expect(s).not.toContain("local-test-key");
    }
  });
});
