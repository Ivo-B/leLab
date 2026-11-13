import { describe, expect, it } from "vitest";
import { trackingPayload } from "./tracking";

const base = {
  trackio_enable: false,
  wandb_enable: false,
  skill: "",
};

describe("trackingPayload", () => {
  it("leaves tracking off by default", () => {
    expect(trackingPayload(base)).toEqual({ tracking: "off" });
  });

  it("sends the skill slug with Weights & Biases", () => {
    expect(
      trackingPayload({ ...base, wandb_enable: true, skill: "bracket" }),
    ).toEqual({
      tracking: "wandb",
      skill: "bracket",
    });
  });

  it("prefers Weights & Biases when both switches are on", () => {
    expect(
      trackingPayload({
        ...base,
        trackio_enable: true,
        wandb_enable: true,
        skill: "bolt",
      }).tracking,
    ).toBe("wandb");
  });

  it("omits a blank skill", () => {
    expect(
      trackingPayload({ ...base, wandb_enable: true, skill: "  " }),
    ).toEqual({ tracking: "wandb" });
  });
});
