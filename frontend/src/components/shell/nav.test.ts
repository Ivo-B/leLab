import { describe, expect, it } from "vitest";
import { NAV } from "./nav";

function active(pathname: string): string[] {
  return NAV.filter((item) => item.isActive(pathname)).map((item) => item.label);
}

describe("NAV", () => {
  it("lists the loop in order", () => {
    expect(NAV.map((item) => item.label)).toEqual([
      "Setup",
      "Record",
      "Train",
      "Evaluate",
      "Orders",
      "Improve",
    ]);
  });

  it("marks exactly one item for each existing route", () => {
    expect(active("/")).toEqual(["Record"]);
    expect(active("/recording")).toEqual(["Record"]);
    expect(active("/edit-dataset")).toEqual(["Record"]);
    expect(active("/calibration")).toEqual(["Setup"]);
    expect(active("/training")).toEqual(["Train"]);
    expect(active("/training/job-1")).toEqual(["Train"]);
    expect(active("/inference")).toEqual(["Evaluate"]);
    expect(active("/evaluate")).toEqual(["Evaluate"]);
    expect(active("/orders")).toEqual(["Orders"]);
    expect(active("/improve")).toEqual(["Improve"]);
    expect(active("/nope")).toEqual([]);
  });
});
