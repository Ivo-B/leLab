import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { BrandMark } from "./BrandMark";

describe("BrandMark", () => {
  afterEach(() => {
    document.documentElement.classList.remove("light", "dark");
  });

  it("shows Garage beside the dark CoreWeave mark and falls back to G", () => {
    render(<BrandMark />);
    expect(screen.getByText("Garage")).toBeInTheDocument();
    expect(screen.getByText("A LeLab demo on CoreWeave")).toBeInTheDocument();
    const img = screen.getByRole("img", { name: "CoreWeave" });
    expect(img).toHaveAttribute("src", "/brand/cw-dark.png");
    fireEvent.error(img);
    expect(screen.getByTestId("brand-fallback")).toHaveTextContent("G");
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });

  it("uses the blue C on a light background", async () => {
    document.documentElement.classList.add("light");
    render(<BrandMark />);
    await waitFor(() =>
      expect(screen.getByRole("img", { name: "CoreWeave" })).toHaveAttribute("src", "/brand/cw-light.png"),
    );
  });

  it("hides the wordmark when iconOnly is set", () => {
    render(<BrandMark iconOnly />);
    expect(screen.queryByText("Garage")).not.toBeInTheDocument();
    expect(screen.queryByText("A LeLab demo on CoreWeave")).not.toBeInTheDocument();
  });
});
