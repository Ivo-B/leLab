import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import Landing from "./Landing";

vi.mock("@/hooks/useRobots", () => ({
  useRobots: () => ({
    selectedName: null,
    selectedRecord: null,
    availableNames: [],
    isLoading: false,
    selectRobot: vi.fn(),
    createRobot: vi.fn(),
    deleteRobot: vi.fn(),
  }),
}));
vi.mock("@/hooks/useDatasets", () => ({ useDatasets: () => ({ datasets: [], loading: false }) }));
vi.mock("@/hooks/useRecording", () => ({
  useRecording: () => ({ openForNew: vi.fn(), modalProps: {} }),
}));
vi.mock("@/components/landing/RobotConfigManager", () => ({ default: () => null }));
vi.mock("@/components/landing/RecordingModal", () => ({ default: () => null }));
vi.mock("@/components/landing/DatasetPicker", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));
vi.mock("@/components/jobs/JobsSection", () => ({ default: () => null }));
vi.mock("@/components/landing/UsageInstructionsModal", () => ({ default: () => null }));
vi.mock("@/components/Footer", () => ({ default: () => null }));

function renderLanding(fetchMock: ReturnType<typeof vi.fn>) {
  vi.stubGlobal("fetch", fetchMock);
  return render(
    <MemoryRouter initialEntries={["/"]}>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/recording" element={<p>Recording page</p>} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("Landing: generate in simulation", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("offers a Generate in simulation link that points at /recording", () => {
    renderLanding(vi.fn());
    const link = screen.getByRole("link", { name: "Generate in simulation" });
    expect(link).toHaveAttribute("href", "/recording");
  });

  it("navigates to /recording without starting a recording", () => {
    const fetchMock = vi.fn();
    renderLanding(fetchMock);
    fireEvent.click(screen.getByRole("link", { name: "Generate in simulation" }));
    expect(screen.getByText("Recording page")).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
