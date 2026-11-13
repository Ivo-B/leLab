import { describe, expect, it } from "vitest";
import { isSkillSlug, trackingPayload } from "./tracking";

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

describe("isSkillSlug", () => {
  it("accepts bolt, bracket, and bolt-2", () => {
    expect(isSkillSlug("bolt")).toBe(true);
    expect(isSkillSlug("bracket")).toBe(true);
    expect(isSkillSlug("bolt-2")).toBe(true);
  });

  it("rejects Bolt, bolt_v1, bolt--gear, and a blank string", () => {
    expect(isSkillSlug("Bolt")).toBe(false);
    expect(isSkillSlug("bolt_v1")).toBe(false);
    expect(isSkillSlug("bolt--gear")).toBe(false);
    expect(isSkillSlug("")).toBe(false);
  });
});
