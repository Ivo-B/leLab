import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiProvider } from "@/contexts/ApiContext";
import Evaluate from "./Evaluate";

const BASE = "http://localhost:8000";
const PLACEHOLDER =
  "Real-arm evaluation in W&B Weave arrives in a later milestone. Rollouts still run from the inference page.";
const EMPTY = "Choose a checkpoint, then score it on the arm.";
const HINT = "Add CWSANDBOX_API_KEY and CWSANDBOX_TRAIN_IMAGE to the backend environment.";

type Json = Record<string, unknown>;

// @testing-library/user-event is not installed, so drive the DOM with fireEvent.
const userEvent = {
  click: async (element: HTMLElement) => {
    fireEvent.click(element);
  },
  selectOptions: async (element: HTMLElement, value: string) => {
    fireEvent.change(element, { target: { value } });
  },
  type: async (element: HTMLElement, text: string) => {
    fireEvent.change(element, { target: { value: (element as HTMLInputElement).value + text } });
  },
  clear: async (element: HTMLElement) => {
    fireEvent.change(element, { target: { value: "" } });
  },
};

function respond(body: unknown, ok = true, status = 200) {
  return Promise.resolve({ ok, status, json: () => Promise.resolve(body) } as Response);
}

function stubFetch(forge: Json, routes: Record<string, (body: Json) => Promise<Response>> = {}) {
  const fetchMock = vi.fn((url: string, init?: RequestInit) => {
    if (url === `${BASE}/forge-status`) return respond(forge);
    const path = url.slice(BASE.length);
    const handler = routes[path];
    if (handler) return handler(init?.body ? (JSON.parse(String(init.body)) as Json) : {});
    return respond({}, false, 404);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function posted(fetchMock: ReturnType<typeof vi.fn>, path: string): Json[] {
  return fetchMock.mock.calls
    .filter((call) => String((call as unknown[])[0]) === `${BASE}${path}`)
    .map((call) => JSON.parse(String(((call as unknown[])[1] as RequestInit).body ?? "{}")) as Json);
}

function renderPage() {
  return render(
    <MemoryRouter>
      <ApiProvider>
        <Evaluate />
      </ApiProvider>
    </MemoryRouter>,
  );
}

const FORGE_ON = { forge_configured: true, sandbox_configured: true };

describe("Evaluate", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shows the placeholder sentence when forge is not configured", async () => {
    stubFetch({ forge_configured: false });
    renderPage();
    expect(await screen.findByText(PLACEHOLDER)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Start evaluation" })).not.toBeInTheDocument();
  });

  it("treats a truthy non-true forge_configured as off", async () => {
    stubFetch({ forge_configured: "yes" });
    renderPage();
    expect(await screen.findByText(PLACEHOLDER)).toBeInTheDocument();
  });

  it("shows the controls and the empty state when forge is configured", async () => {
    stubFetch(FORGE_ON);
    renderPage();
    expect(await screen.findByRole("button", { name: "Start evaluation" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Evaluate" })).toBeInTheDocument();
    expect(screen.getByText(EMPTY)).toBeInTheDocument();
    expect(screen.getByLabelText("Policy family")).toBeInTheDocument();
    expect(screen.getByLabelText("Trials")).toHaveValue(10);
    const version = screen.getByLabelText("Version");
    expect(version).toHaveDisplayValue("candidate");
    expect(screen.getByRole("option", { name: "production" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Confirm" })).not.toBeInTheDocument();
    expect(screen.queryByText(PLACEHOLDER)).not.toBeInTheDocument();
  });

  it("disables CoreWeave and shows the hint when the sandbox is not configured", async () => {
    stubFetch({ forge_configured: true, sandbox_configured: false });
    renderPage();
    await screen.findByRole("button", { name: "Start evaluation" });
    expect(screen.getByRole("option", { name: "CoreWeave" })).toBeDisabled();
    expect(screen.getByText(HINT)).toBeInTheDocument();
  });

  it("disables CoreWeave unless sandbox_configured is strictly true", async () => {
    stubFetch({ forge_configured: true, sandbox_configured: "yes" });
    renderPage();
    await screen.findByRole("button", { name: "Start evaluation" });
    expect(screen.getByRole("option", { name: "CoreWeave" })).toBeDisabled();
  });

  it("enables CoreWeave and hides the hint when the sandbox is configured", async () => {
    stubFetch(FORGE_ON);
    renderPage();
    await screen.findByRole("button", { name: "Start evaluation" });
    expect(screen.getByRole("option", { name: "CoreWeave" })).toBeEnabled();
    expect(screen.queryByText(HINT)).not.toBeInTheDocument();
  });

  it("starts a SmolVLA evaluation without a skill", async () => {
    const fetchMock = stubFetch(FORGE_ON, {
      "/evaluations": () =>
        respond({ policy_family: "smolvla", version: "candidate", policy_host: "local", trials: 30 }),
    });
    renderPage();
    await userEvent.click(await screen.findByRole("button", { name: "Start evaluation" }));

    await waitFor(() =>
      expect(posted(fetchMock, "/evaluations")).toEqual([
        { policy_family: "smolvla", version: "candidate", policy_host: "local", trials_per_skill: 10 },
      ]),
    );
    expect(await screen.findByText("Session open: 30 trials planned.")).toBeInTheDocument();
    expect(screen.queryByText(EMPTY)).not.toBeInTheDocument();
  });

  it("asks for a skill for ACT and sends it", async () => {
    const fetchMock = stubFetch(FORGE_ON, {
      "/evaluations": () =>
        respond({ policy_family: "act", version: "production", policy_host: "coreweave", trials: 5 }),
    });
    renderPage();
    await screen.findByRole("button", { name: "Start evaluation" });
    expect(screen.queryByLabelText("Skill")).not.toBeInTheDocument();

    await userEvent.selectOptions(screen.getByLabelText("Policy family"), "act");
    await userEvent.selectOptions(screen.getByLabelText("Version"), "production");
    await userEvent.selectOptions(screen.getByLabelText("Policy host"), "coreweave");
    await userEvent.type(screen.getByLabelText("Skill"), "bolt");
    const trials = screen.getByLabelText("Trials");
    await userEvent.clear(trials);
    await userEvent.type(trials, "5");
    await userEvent.click(screen.getByRole("button", { name: "Start evaluation" }));

    await waitFor(() =>
      expect(posted(fetchMock, "/evaluations")).toEqual([
        {
          policy_family: "act",
          version: "production",
          policy_host: "coreweave",
          trials_per_skill: 5,
          skill: "bolt",
        },
      ]),
    );
  });

  it("shows the server error when the start is refused", async () => {
    stubFetch(FORGE_ON, {
      "/evaluations": () => respond({ detail: "An evaluation session is already open" }, false, 409),
    });
    renderPage();
    await userEvent.click(await screen.findByRole("button", { name: "Start evaluation" }));
    expect(await screen.findByText("An evaluation session is already open")).toBeInTheDocument();
    expect(screen.getByText(EMPTY)).toBeInTheDocument();
  });

  async function openSession(routes: Record<string, (body: Json) => Promise<Response>> = {}) {
    const fetchMock = stubFetch(FORGE_ON, {
      "/evaluations": () =>
        respond({ policy_family: "smolvla", version: "candidate", policy_host: "local", trials: 30 }),
      ...routes,
    });
    renderPage();
    await userEvent.click(await screen.findByRole("button", { name: "Start evaluation" }));
    await screen.findByRole("button", { name: "Needs review" });
    return fetchMock;
  }

  it("posts a null stage and says Needs review", async () => {
    const fetchMock = await openSession({
      "/evaluations/current/override": () =>
        respond({ skill: "bolt", stage: null, verdict_source: "operator" }),
    });
    await userEvent.click(screen.getByRole("button", { name: "Needs review" }));
    await waitFor(() =>
      expect(posted(fetchMock, "/evaluations/current/override")).toEqual([{ stage: null }]),
    );
    expect(await screen.findByText("Needs review: no stage recorded (operator)")).toBeInTheDocument();
  });

  it("posts the chosen stage on Confirm and shows the stage word", async () => {
    const fetchMock = await openSession({
      "/evaluations/current/override": (body) =>
        respond({ skill: "bolt", stage: body.stage, verdict_source: "operator" }),
    });
    await userEvent.selectOptions(screen.getByLabelText("Stage"), "grasped");
    await userEvent.click(screen.getByRole("button", { name: "Confirm" }));
    await waitFor(() =>
      expect(posted(fetchMock, "/evaluations/current/override")).toEqual([{ stage: "grasped" }]),
    );
    expect(await screen.findByText("Stage: grasped (operator)")).toBeInTheDocument();
  });

  it("shows the server error when an override is refused", async () => {
    await openSession({
      "/evaluations/current/override": () => respond({ detail: "No finished trial to override" }, false, 400),
    });
    await userEvent.click(screen.getByRole("button", { name: "Confirm" }));
    expect(await screen.findByText("No finished trial to override")).toBeInTheDocument();
  });

  it("closes the session, shows the Weave link, and returns to the empty state", async () => {
    const fetchMock = await openSession({
      "/evaluations/current/close": () => respond({ weave_url: "https://wandb.ai/acme/forge/weave" }),
    });
    await userEvent.click(screen.getByRole("button", { name: "Close evaluation" }));
    await waitFor(() => expect(posted(fetchMock, "/evaluations/current/close")).toHaveLength(1));
    const link = await screen.findByRole("link", { name: "Open in W&B Weave" });
    expect(link).toHaveAttribute("href", "https://wandb.ai/acme/forge/weave");
    expect(screen.getByText(EMPTY)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Confirm" })).not.toBeInTheDocument();
  });
});
