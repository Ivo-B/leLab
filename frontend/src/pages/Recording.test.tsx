import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiProvider } from "@/contexts/ApiContext";
import Recording from "./Recording";

const BASE = "http://localhost:8000";

const RECORDING_CONFIG = {
  leader_port: "/dev/leader",
  follower_port: "/dev/follower",
  leader_config: "leader",
  follower_config: "follower",
  dataset_repo_id: "local/demo",
  single_task: "move a piece",
  num_episodes: 5,
  episode_time_s: 30,
  reset_time_s: 10,
  fps: 30,
  video: true,
  push_to_hub: false,
  resume: false,
  streaming_encoding: true,
};

const RECORDING_STATUS = {
  recording_active: true,
  current_phase: "recording",
  current_episode: 1,
  total_episodes: 5,
  phase_elapsed_seconds: 1,
  session_elapsed_seconds: 1,
  cameras: [],
  available_controls: { stop_recording: true, exit_early: true, rerecord_episode: true },
};

type Json = Record<string, unknown>;

function respond(body: unknown, ok = true, status = 200) {
  return Promise.resolve({ ok, status, json: () => Promise.resolve(body) } as Response);
}

function stubFetch(simPost: () => Promise<Response>) {
  const fetchMock = vi.fn((url: string, init?: RequestInit) => {
    const method = init?.method ?? "GET";
    if (url === `${BASE}/start-recording`) return respond({ success: true });
    if (url === `${BASE}/recording-status`) return respond(RECORDING_STATUS);
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

function renderPage() {
  return render(
    <MemoryRouter initialEntries={[{ pathname: "/recording", state: { recordingConfig: RECORDING_CONFIG } }]}>
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

  it("posts the default request to /sim-generate and starts nothing on the arm", async () => {
    const fetchMock = stubFetch(() => respond({ started: true }));
    renderPage();

    const button = await screen.findByRole("button", { name: "Generate in simulation" });
    fireEvent.click(button);

    await waitFor(() =>
      expect(posted(fetchMock, "/sim-generate")).toEqual([{ episodes: 30, seed: 0, randomize: true }]),
    );
    expect(await screen.findByText("Generating 0/30 episodes in simulation…")).toBeInTheDocument();
    expect(button).toBeDisabled();
    // The only arm session is the one the page opened itself, and it opened exactly one.
    expect(posted(fetchMock, "/start-recording")).toHaveLength(1);
  });

  it("shows the server error when the start is refused", async () => {
    stubFetch(() => respond({ detail: "A simulation job is already running" }, false, 409));
    renderPage();

    fireEvent.click(await screen.findByRole("button", { name: "Generate in simulation" }));

    expect(await screen.findByText("A simulation job is already running")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Generate in simulation" })).toBeEnabled();
  });
});
