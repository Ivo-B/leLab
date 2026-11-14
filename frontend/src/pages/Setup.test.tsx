import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiProvider } from "@/contexts/ApiContext";
import Setup from "./Setup";

const useRobotsMock = vi.fn();

vi.mock("@/hooks/useRobots", () => ({
  useRobots: () => useRobotsMock(),
}));

function renderSetup() {
  return render(
    <MemoryRouter>
      <ApiProvider>
        <Setup />
      </ApiProvider>
    </MemoryRouter>,
  );
}

describe("Setup calibration links", () => {
  beforeEach(() => {
    useRobotsMock.mockReset();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          forge_configured: false,
          sandbox_configured: false,
          entity: null,
          project: "so101-kitting",
          registry: { ready: false, collections: [] },
        }),
      }),
    );
  });

  it("opens calibration with the selected robot in link state", () => {
    useRobotsMock.mockReturnValue({
      selectedName: "my_arm",
      availableNames: ["my_arm"],
      isLoading: false,
    });
    renderSetup();
    const link = screen.getByRole("link", {
      name: /Calibrate arms and cameras/,
    });
    expect(link).toHaveAttribute("href", "/calibration");
    expect(screen.getByText(/for "my_arm"/)).toBeInTheDocument();
  });

  it("sends the user home when no robot is available", () => {
    useRobotsMock.mockReturnValue({
      selectedName: null,
      availableNames: [],
      isLoading: false,
    });
    renderSetup();
    const link = screen.getByRole("link", {
      name: /Calibrate arms and cameras/,
    });
    expect(link).toHaveAttribute("href", "/");
    expect(link.textContent).toMatch(/Select or create a robot on Home first/);
  });

  it("shows an interactive Connect Forge form", async () => {
    useRobotsMock.mockReturnValue({
      selectedName: "my_arm",
      availableNames: ["my_arm"],
      isLoading: false,
    });
    renderSetup();
    expect(await screen.findByLabelText(/W&B API key/)).toBeInTheDocument();
    expect(screen.getByLabelText(/CoreWeave Sandbox API key/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Save to \.env/ })).toBeInTheDocument();
  });

  it("shows Model Registry instead of Hugging Face on Setup", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          forge_configured: true,
          sandbox_configured: false,
          entity: "coreweave-demo",
          project: "so101-kitting",
          registry: {
            ready: true,
            collections: [
              {
                name: "so101-kitting-smolvla",
                aliases: ["candidate"],
                linked: true,
              },
            ],
          },
        }),
      }),
    );
    useRobotsMock.mockReturnValue({
      selectedName: "my_arm",
      availableNames: ["my_arm"],
      isLoading: false,
    });
    renderSetup();
    expect(await screen.findByText("Model Registry")).toBeInTheDocument();
    expect(screen.queryByText(/Sign in to Hugging Face/)).not.toBeInTheDocument();
    expect(screen.getByText(/so101-kitting-smolvla/)).toBeInTheDocument();
  });

  it("highlights fields that are already set", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          forge_configured: true,
          sandbox_configured: false,
          entity: "coreweave-demo",
          project: "so101-kitting",
          sandbox_image: null,
          fields: {
            api_key: true,
            entity: true,
            project: true,
            sandbox_api_key: false,
            sandbox_image: false,
          },
        }),
      }),
    );
    useRobotsMock.mockReturnValue({
      selectedName: "my_arm",
      availableNames: ["my_arm"],
      isLoading: false,
    });
    renderSetup();
    expect(await screen.findAllByText("Set")).toHaveLength(3);
    expect(screen.getAllByText("Missing")).toHaveLength(2);
  });
});
