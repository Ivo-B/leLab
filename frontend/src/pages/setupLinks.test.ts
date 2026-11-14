import { describe, expect, it } from "vitest";
import { calibrationRobotName } from "./setupLinks";

describe("calibrationRobotName", () => {
  it("uses the selected robot when it still exists", () => {
    expect(calibrationRobotName("my_arm", ["other", "my_arm"])).toBe("my_arm");
  });

  it("falls back to the only robot when nothing is selected", () => {
    expect(calibrationRobotName(null, ["my_arm"])).toBe("my_arm");
  });

  it("returns null when selection is missing and several robots exist", () => {
    expect(calibrationRobotName(null, ["a", "b"])).toBeNull();
    expect(calibrationRobotName("gone", ["a", "b"])).toBeNull();
  });

  it("returns null when there are no robots", () => {
    expect(calibrationRobotName(null, [])).toBeNull();
  });
});
