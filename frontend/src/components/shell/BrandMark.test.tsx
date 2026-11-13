import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { BrandMark } from "./BrandMark";

describe("BrandMark", () => {
  it("shows the wordmark and swaps to the letter mark when the logo file is missing", () => {
    render(<BrandMark />);
    expect(screen.getByText("LeLab")).toBeInTheDocument();
    const img = screen.getByRole("img", { name: "LeLab" });
    expect(img).toHaveAttribute("src", "/brand/mark.svg");
    fireEvent.error(img);
    expect(screen.getByTestId("brand-fallback")).toHaveTextContent("L");
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });

  it("hides the wordmark when iconOnly is set", () => {
    render(<BrandMark iconOnly />);
    expect(screen.queryByText("LeLab")).not.toBeInTheDocument();
  });
});
