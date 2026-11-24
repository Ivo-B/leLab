import { describe, expect, it } from "vitest";
import { trackingPayload } from "./tracking";

describe("trackingPayload", () => {
  it("never sends a skill", () => {
    expect(trackingPayload({ trackio_enable: false, wandb_enable: true })).toEqual({ tracking: "wandb" });
    expect(trackingPayload({ trackio_enable: true, wandb_enable: false })).toEqual({ tracking: "trackio" });
    expect(trackingPayload({ trackio_enable: false, wandb_enable: false })).toEqual({ tracking: "off" });
  });

  it("prefers Weights & Biases when both switches are on", () => {
    expect(trackingPayload({ trackio_enable: true, wandb_enable: true }).tracking).toBe("wandb");
  });
});
