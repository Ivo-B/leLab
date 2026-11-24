import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import RecordSourceBanner from "./RecordSourceBanner";

describe("RecordSourceBanner", () => {
  it("says simulation never moves the arm and names the sim artifact", () => {
    render(<RecordSourceBanner source="sim" />);
    const banner = screen.getByRole("note");
    expect(banner).toHaveTextContent("Simulation");
    expect(banner).toHaveTextContent("The arm does not move.");
    expect(banner).toHaveTextContent("so101-chess-sim");
  });

  it("says teleoperation moves the real arm and is not co-training data yet", () => {
    render(<RecordSourceBanner source="teleop" />);
    const banner = screen.getByRole("note");
    expect(banner).toHaveTextContent("Teleoperation");
    expect(banner).toHaveTextContent("The real arm moves.");
    expect(banner).toHaveTextContent("so101-chess-real");
    expect(banner).toHaveTextContent("no red/green markers or transfer sidecar");
  });

  it("says chess transfer mode records co-training data with markers and a sidecar", () => {
    render(<RecordSourceBanner source="teleop" chessTransfers />);
    const banner = screen.getByRole("note");
    expect(banner).toHaveTextContent("Chess transfer mode.");
    expect(banner).toHaveTextContent(
      "The real arm moves. The top camera shows the red/green markers, each episode's transfer is saved in the sidecar, and the dataset is logged to W&B as so101-chess-real when Forge is connected. This is chess co-training data.",
    );
    expect(banner).not.toHaveTextContent("not chess co-training data yet");
  });
});
