import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it } from "vitest";
import { ApiProvider } from "@/contexts/ApiContext";
import { HfAuthProvider } from "@/contexts/HfAuthContext";
import { OnboardingProvider } from "@/contexts/OnboardingContext";
import { AppShell } from "./AppShell";

describe("AppShell", () => {
  beforeEach(() => {
    // A first visit opens the onboarding welcome dialog, which hides the nav.
    localStorage.setItem("lelab:onboarding-v1", "dismissed");
  });

  it("highlights Train and shows the status bar", () => {
    render(
      <MemoryRouter initialEntries={["/training"]}>
        <ApiProvider>
          <HfAuthProvider>
            <OnboardingProvider>
              <Routes>
                <Route element={<AppShell />}>
                  <Route path="/training" element={<p>Training page</p>} />
                </Route>
              </Routes>
            </OnboardingProvider>
          </HfAuthProvider>
        </ApiProvider>
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
