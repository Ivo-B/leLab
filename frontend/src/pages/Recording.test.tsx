import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiProvider } from "@/contexts/ApiContext";
import Recording from "./Recording";

const BASE = "http://localhost:8000";

type Json = Record<string, unknown>;

function respond(body: unknown, ok = true, status = 200) {
  return Promise.resolve({ ok, status, json: () => Promise.resolve(body) } as Response);
}

function stubFetch(simPost: () => Promise<Response>) {
  const fetchMock = vi.fn((url: string, init?: RequestInit) => {
    const method = init?.method ?? "GET";
    if (url === `${BASE}/sim-generate` && method === "POST") return simPost();
    if (url === `${BASE}/sim-generate`) {
      return respond({ running: false, saved: 0, episodes: 0, error: null });
    }
    return respond({}, false, 404);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function posted(fetchMock: ReturnType<typeof vi.fn>, path: string): Json[] {
  return fetchMock.mock.calls
    .filter(
      (call) =>
        String((call as unknown[])[0]) === `${BASE}${path}` &&
        ((call as unknown[])[1] as RequestInit | undefined)?.method === "POST",
    )
    .map((call) => JSON.parse(String(((call as unknown[])[1] as RequestInit).body ?? "{}")) as Json);
}

// No router state: the page was opened without a recording config.
function renderPage() {
  return render(
    <MemoryRouter initialEntries={["/recording"]}>
      <ApiProvider>
        <Recording />
      </ApiProvider>
    </MemoryRouter>,
  );
}

describe("Recording: generate in simulation", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shows the control without a recording config and never starts the arm", async () => {
    const fetchMock = stubFetch(() => respond({ started: true }));
    renderPage();

    const button = await screen.findByRole("button", { name: "Generate in simulation" });
    expect(screen.getByLabelText("Episodes")).toHaveValue(30);
    expect(screen.getByLabelText("Seed")).toHaveValue(0);
    expect(screen.getByLabelText("Randomize the scene")).toBeChecked();

    fireEvent.click(button);
    await waitFor(() =>
      expect(posted(fetchMock, "/sim-generate")).toEqual([{ episodes: 30, seed: 0, randomize: true }]),
    );
    expect(await screen.findByText("Generating 0/30 episodes in simulation…")).toBeInTheDocument();
    expect(button).toBeDisabled();

    const urls = fetchMock.mock.calls.map((call) => String((call as unknown[])[0]));
    expect(urls.filter((url) => url.includes("/start-recording"))).toEqual([]);
    expect(urls.every((url) => url.endsWith("/sim-generate"))).toBe(true);
  });

  it("posts the episodes, seed and randomization the user chose", async () => {
    const fetchMock = stubFetch(() => respond({ started: true }));
    renderPage();
    await screen.findByRole("button", { name: "Generate in simulation" });

    fireEvent.change(screen.getByLabelText("Episodes"), { target: { value: "250" } });
    fireEvent.change(screen.getByLabelText("Seed"), { target: { value: "7" } });
    fireEvent.click(screen.getByLabelText("Randomize the scene"));
    fireEvent.click(screen.getByRole("button", { name: "Generate in simulation" }));

    await waitFor(() =>
      expect(posted(fetchMock, "/sim-generate")).toEqual([{ episodes: 250, seed: 7, randomize: false }]),
    );
  });

  it("disables the button for an invalid episode count", async () => {
    stubFetch(() => respond({ started: true }));
    renderPage();
    fireEvent.change(await screen.findByLabelText("Episodes"), { target: { value: "0" } });
    expect(screen.getByRole("button", { name: "Generate in simulation" })).toBeDisabled();
  });

  it("shows the server error when the start is refused", async () => {
    stubFetch(() => respond({ detail: "A simulation job is already running" }, false, 409));
    renderPage();

    fireEvent.click(await screen.findByRole("button", { name: "Generate in simulation" }));

    expect(await screen.findByText("A simulation job is already running")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Generate in simulation" })).toBeEnabled();
  });
});
