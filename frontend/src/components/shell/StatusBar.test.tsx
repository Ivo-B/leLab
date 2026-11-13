import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiProvider } from "@/contexts/ApiContext";
import { HfAuthProvider } from "@/contexts/HfAuthContext";
import { OnboardingProvider } from "@/contexts/OnboardingContext";
import { StatusBar } from "./StatusBar";

const BASE = "http://localhost:8000";
const STATUS_PATHS = [
  "/teleoperation-status",
  "/recording-status",
  "/replay-status",
  "/inference-status",
  "/jobs",
];

function respond(body: unknown) {
  return Promise.resolve({ ok: true, json: () => Promise.resolve(body) } as Response);
}

function statusCalls(fetchMock: ReturnType<typeof vi.fn>): string[] {
  return fetchMock.mock.calls
    .map((call) => String((call as unknown[])[0]))
    .filter((url) => STATUS_PATHS.some((path) => url.endsWith(path)));
}

function renderBar() {
  return render(
    <MemoryRouter>
      <ApiProvider>
        <HfAuthProvider>
          <OnboardingProvider>
            <StatusBar />
          </OnboardingProvider>
        </HfAuthProvider>
      </ApiProvider>
    </MemoryRouter>,
  );
}

describe("StatusBar", () => {
  beforeEach(() => {
    // A first visit opens the onboarding welcome dialog.
    localStorage.setItem("lelab:onboarding-v1", "dismissed");
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("requests every status endpoint under the API base URL", async () => {
    const fetchMock = vi.fn(() => respond({}));
    vi.stubGlobal("fetch", fetchMock);
    renderBar();

    await waitFor(() => expect(statusCalls(fetchMock)).toHaveLength(STATUS_PATHS.length));
    expect(statusCalls(fetchMock).sort()).toEqual(STATUS_PATHS.map((path) => `${BASE}${path}`).sort());
  });

  it.each([
    ["/teleoperation-status", { teleoperation_active: true }],
    ["/recording-status", { recording_active: true }],
    ["/replay-status", { replay_active: true }],
    ["/inference-status", { inference_active: true }],
  ])("shows Arm in use when %s reports activity", async (path, body) => {
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) => respond(url === `${BASE}${path}` ? body : {})),
    );
    renderBar();
    expect(await screen.findByText("Arm in use")).toBeInTheDocument();
  });

  it("shows the name of the first running job", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) =>
        respond(
          url === `${BASE}/jobs`
            ? {
                jobs: [
                  { name: "finished-job", state: "completed" },
                  { name: "so101-act", state: "running" },
                  { name: "later-job", state: "running" },
                ],
              }
            : {},
        ),
      ),
    );
    renderBar();
    expect(await screen.findByText("so101-act")).toBeInTheDocument();
  });

  it("keeps the previous labels when a later poll rejects", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let failing = false;
    const fetchMock = vi.fn((url: string) => {
      if (failing) return Promise.reject(new Error("offline"));
      if (url === `${BASE}/teleoperation-status`) return respond({ teleoperation_active: true });
      if (url === `${BASE}/jobs`) return respond({ jobs: [{ name: "so101-act", state: "running" }] });
      return respond({});
    });
    vi.stubGlobal("fetch", fetchMock);
    renderBar();
    expect(await screen.findByText("Arm in use")).toBeInTheDocument();
    expect(screen.getByText("so101-act")).toBeInTheDocument();

    failing = true;
    const before = statusCalls(fetchMock).length;
    await vi.advanceTimersByTimeAsync(5100);

    expect(statusCalls(fetchMock).length).toBeGreaterThan(before);
    expect(screen.getByText("Arm in use")).toBeInTheDocument();
    expect(screen.getByText("so101-act")).toBeInTheDocument();
  });
});
