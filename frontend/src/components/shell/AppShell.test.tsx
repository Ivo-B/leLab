import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { AppShell } from "./AppShell";

describe("AppShell", () => {
  it("highlights Train and shows the status bar", () => {
    render(
      <MemoryRouter initialEntries={["/training"]}>
        <Routes>
          <Route element={<AppShell />}>
            <Route path="/training" element={<p>Training page</p>} />
          </Route>
        </Routes>
      </MemoryRouter>,
    );
    const train = screen.getByRole("link", { name: "Train" });
    expect(train.className).toContain("bg-primary");
    expect(screen.getByRole("link", { name: "Record" }).className).not.toContain("bg-primary");
    expect(screen.getByText("Local")).toBeInTheDocument();
    expect(screen.getByText("Not connected")).toBeInTheDocument();
    expect(screen.getByText("Training page")).toBeInTheDocument();
  });
});
