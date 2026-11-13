import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it } from "vitest";
import { ApiProvider } from "@/contexts/ApiContext";
import { HfAuthProvider } from "@/contexts/HfAuthContext";
import { OnboardingProvider } from "@/contexts/OnboardingContext";
import { AppShell } from "./AppShell";

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <ApiProvider>
        <HfAuthProvider>
          <OnboardingProvider>
            <Routes>
              <Route element={<AppShell />}>
                <Route path="/training" element={<p>Training page</p>} />
                <Route path="/calibration" element={<p>Calibration page</p>} />
                <Route path="/recording" element={<p>Recording page</p>} />
                <Route path="/inference" element={<p>Inference page</p>} />
              </Route>
            </Routes>
          </OnboardingProvider>
        </HfAuthProvider>
      </ApiProvider>
    </MemoryRouter>,
  );
}

describe("AppShell", () => {
  beforeEach(() => {
    // A first visit opens the onboarding welcome dialog, which hides the nav.
    localStorage.setItem("lelab:onboarding-v1", "dismissed");
  });

  it("highlights Train and shows the status bar", () => {
    renderAt("/training");
    const train = screen.getByRole("link", { name: "Train" });
    expect(train.className).toContain("bg-primary");
    expect(screen.getByRole("link", { name: "Record" }).className).not.toContain("bg-primary");
    expect(screen.getByText("Local")).toBeInTheDocument();
    expect(screen.getByText("Not connected")).toBeInTheDocument();
    expect(screen.getByText("Training page")).toBeInTheDocument();
  });

  it("marks Setup active on /calibration", () => {
    renderAt("/calibration");
    const setup = screen.getByRole("link", { name: "Setup" });
    expect(setup.className).toContain("bg-primary");
    expect(setup).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("link", { name: "Record" }).className).not.toContain("bg-primary");
  });

  it("marks Record active on /recording", () => {
    renderAt("/recording");
    const record = screen.getByRole("link", { name: "Record" });
    expect(record.className).toContain("bg-primary");
    expect(record).toHaveAttribute("aria-current", "page");
  });

  it("marks Evaluate active on /inference", () => {
    renderAt("/inference");
    const evaluate = screen.getByRole("link", { name: "Evaluate" });
    expect(evaluate.className).toContain("bg-primary");
    expect(evaluate).toHaveAttribute("aria-current", "page");
  });
});
