import { act, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiProvider } from "@/contexts/ApiContext";
import Improve from "./Improve";

const BASE = "http://localhost:8000";
const PLACEHOLDER = "CoreWeave ARIA reports arrive in a later milestone.";
const NAME = "so101-chess evaluation";
const EVENT = "A run reaches FINISHED";
const PROMPT = "P";
const EMPTY = "Close an evaluation, then open the project Automations tab and choose Trigger ARIA.";
const AUTOMATION = { name: NAME, event: EVENT, prompt: PROMPT };

type Json = Record<string, unknown>;

function respond(body: unknown, ok = true, status = 200) {
  return Promise.resolve({ ok, status, json: () => Promise.resolve(body) } as Response);
}

function stubFetch(forge: Json, automation: Json | null = AUTOMATION) {
  const fetchMock = vi.fn((url: string) => {
    if (url === `${BASE}/forge-status`) return respond(forge);
    if (url === `${BASE}/improve/automation` && automation) return respond(automation);
    return respond({}, false, 404);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function renderPage() {
  return render(
    <MemoryRouter>
      <ApiProvider>
        <Improve />
      </ApiProvider>
    </MemoryRouter>,
  );
}

describe("Improve", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shows the placeholder sentence and no automation name when forge is not configured", async () => {
    stubFetch({ forge_configured: false });
    renderPage();
    expect(await screen.findByText(PLACEHOLDER)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Improve" })).toBeInTheDocument();
    expect(screen.queryByText(NAME)).not.toBeInTheDocument();
  });

  it("treats a truthy non-true forge_configured as off", async () => {
    stubFetch({ forge_configured: "yes" });
    renderPage();
    expect(await screen.findByText(PLACEHOLDER)).toBeInTheDocument();
    expect(screen.queryByText(NAME)).not.toBeInTheDocument();
  });

  it("treats a failed forge-status request as off", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => respond({}, false, 500)),
    );
    renderPage();
    expect(await screen.findByText(PLACEHOLDER)).toBeInTheDocument();
    expect(screen.queryByText(NAME)).not.toBeInTheDocument();
  });

  it("shows the automation name, event, prompt, and empty state when forge is on", async () => {
    stubFetch({ forge_configured: true });
    renderPage();
    expect(await screen.findByText(NAME)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Improve" })).toBeInTheDocument();
    expect(screen.getByText(EVENT)).toBeInTheDocument();
    expect(screen.getByText(PROMPT)).toBeInTheDocument();
    expect(screen.getByText(EMPTY)).toBeInTheDocument();
    expect(screen.queryByText(PLACEHOLDER)).not.toBeInTheDocument();
  });

  it("shows only the empty state when the automation request fails", async () => {
    const fetchMock = stubFetch({ forge_configured: true }, null);
    renderPage();
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(`${BASE}/improve/automation`, expect.anything()));
    await act(async () => {});
    expect(screen.getByText(EMPTY)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Improve" })).toBeInTheDocument();
    expect(screen.queryByText("Automation")).not.toBeInTheDocument();
  });

  it("does not ask for the automation when forge is off", async () => {
    const fetchMock = stubFetch({ forge_configured: false });
    renderPage();
    await screen.findByText(PLACEHOLDER);
    expect(fetchMock.mock.calls.map((call) => String(call[0]))).toEqual([`${BASE}/forge-status`]);
  });

  it("links to nothing on wandb.ai or aria", async () => {
    stubFetch({ forge_configured: true });
    renderPage();
    const page = (await screen.findByRole("heading", { name: "Improve" })).closest("section") as HTMLElement;
    for (const link of page.querySelectorAll("a")) {
      const href = link.getAttribute("href") ?? "";
      expect(href).not.toContain("wandb.ai");
      expect(href).not.toContain("aria");
    }
    expect(page.querySelector('a[href*="wandb.ai"]')).toBeNull();
    expect(page.querySelector('a[href*="aria"]')).toBeNull();
  });

  it("asks for forge-status and the automation once and does not poll", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      const fetchMock = stubFetch({ forge_configured: true });
      renderPage();
      await screen.findByText(NAME);
      await vi.advanceTimersByTimeAsync(10_000);
      expect(fetchMock.mock.calls.map((call) => String(call[0]))).toEqual([
        `${BASE}/forge-status`,
        `${BASE}/improve/automation`,
      ]);
    } finally {
      vi.useRealTimers();
    }
  });
});
