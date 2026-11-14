import { fireEvent, render, screen } from "@testing-library/react";
import { Link, MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiProvider } from "@/contexts/ApiContext";
import { ForgeStatusChips } from "./ForgeStatusChips";

const BASE = "http://localhost:8000";

function respond(body: unknown, ok = true) {
  return Promise.resolve({ ok, json: () => Promise.resolve(body) } as Response);
}

function renderChips() {
  return render(
    <MemoryRouter>
      <ApiProvider>
        <ForgeStatusChips />
      </ApiProvider>
    </MemoryRouter>,
  );
}

describe("ForgeStatusChips", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shows the entity and CoreWeave when both are configured", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        respond({ forge_configured: true, sandbox_configured: true, entity: "ivos-playground" }),
      ),
    );
    renderChips();
    expect(await screen.findByLabelText("Weights & Biases connected as ivos-playground")).toHaveTextContent(
      "ivos-playground",
    );
    expect(screen.getByLabelText("CoreWeave ready")).toHaveTextContent("CoreWeave");
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("counts running CoreWeave runs on the ready chip", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => respond({ forge_configured: true, sandbox_configured: true, entity: "team" })),
    );
    render(
      <MemoryRouter>
        <ApiProvider>
          <ForgeStatusChips coreweaveRuns={1} />
        </ApiProvider>
      </MemoryRouter>,
    );
    expect(await screen.findByLabelText("CoreWeave · 1 run")).toHaveTextContent("CoreWeave · 1 run");
  });

  it("opens the Setup hint for a missing Weights & Biases key", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => respond({ forge_configured: false, sandbox_configured: true })),
    );
    renderChips();
    fireEvent.click(await screen.findByRole("button", { name: "Weights & Biases not configured" }));
    expect(screen.getByText("Add WANDB_API_KEY and WANDB_ENTITY on Setup.")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open Setup" })).toHaveAttribute("href", "/setup");
  });

  it("opens the Setup hint for a missing CoreWeave key", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => respond({ forge_configured: true, sandbox_configured: false, entity: "ivos-playground" })),
    );
    renderChips();
    fireEvent.click(await screen.findByRole("button", { name: "CoreWeave not configured" }));
    expect(
      screen.getByText("Add CWSANDBOX_API_KEY and CWSANDBOX_TRAIN_IMAGE on Setup."),
    ).toBeInTheDocument();
  });

  it("treats a failed status request as not configured", async () => {
    vi.stubGlobal("fetch", vi.fn((_url: string) => respond({}, false)));
    renderChips();
    expect(await screen.findByRole("button", { name: "Weights & Biases not configured" })).toHaveTextContent(
      "Weights & Biases · not set up",
    );
    expect(screen.getByRole("button", { name: "CoreWeave not configured" })).toHaveTextContent(
      "CoreWeave · not set up",
    );
  });

  it("asks again after a route change so saved keys show up", async () => {
    const fetchMock = vi
      .fn((_url: string) => respond({ forge_configured: false, sandbox_configured: false }))
      .mockImplementationOnce((_url: string) => respond({ forge_configured: false, sandbox_configured: false }))
      .mockImplementation((_url: string) =>
        respond({ forge_configured: true, sandbox_configured: true, entity: "team" }),
      );
    vi.stubGlobal("fetch", fetchMock);
    render(
      <MemoryRouter>
        <ApiProvider>
          <ForgeStatusChips />
          <Link to="/next">Next page</Link>
        </ApiProvider>
      </MemoryRouter>,
    );
    expect(await screen.findByRole("button", { name: "Weights & Biases not configured" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("link", { name: "Next page" }));

    expect(await screen.findByLabelText("Weights & Biases connected as team")).toBeInTheDocument();
    expect(screen.getByLabelText("CoreWeave ready")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(String(fetchMock.mock.calls[0][0])).toBe(`${BASE}/forge-status`);
  });

  it("marks a ready chip as a status", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => respond({ forge_configured: true, sandbox_configured: true, entity: "team" })),
    );
    renderChips();
    expect(await screen.findByRole("status", { name: "CoreWeave ready" })).toBeInTheDocument();
  });
});
