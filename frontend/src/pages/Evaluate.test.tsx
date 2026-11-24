import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiProvider } from "@/contexts/ApiContext";
import Evaluate from "./Evaluate";

const BASE = "http://localhost:8000";
const PLACEHOLDER =
  "Connect Forge (Weights & Biases) to record the operator's outcome for each transfer of the plan and log it to W&B Weave. Simulation evaluation below works without it.";
const EMPTY = "Choose a checkpoint, then run the 20-transfer plan on the arm and record how each transfer ends.";
const HINT = "Add CWSANDBOX_API_KEY and CWSANDBOX_TRAIN_IMAGE to the backend environment.";
const HOST_NOTE =
  "The CoreWeave policy host is not wired yet. Trials run on this machine and are logged as local in Weave, with CoreWeave kept as the requested host.";

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
    expect(screen.getByLabelText("Rounds")).toHaveValue(1);
    expect(screen.queryByLabelText("Trials")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Skill")).not.toBeInTheDocument();
    const version = screen.getByLabelText("Version");
    expect(version).toHaveDisplayValue("candidate");
    expect(screen.getByRole("option", { name: "production" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Record outcome" })).not.toBeInTheDocument();
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

  it("says the CoreWeave policy host is not wired when CoreWeave is picked", async () => {
    stubFetch(FORGE_ON);
    renderPage();
    await screen.findByRole("button", { name: "Start evaluation" });
    expect(screen.queryByText(HOST_NOTE)).not.toBeInTheDocument();
    await userEvent.selectOptions(screen.getByLabelText("Policy host"), "coreweave");
    expect(screen.getByText(HOST_NOTE)).toBeInTheDocument();
  });

  const OPEN = {
    open: true,
    policy_family: "smolvla",
    version: "candidate",
    policy_host: "local",
    trials: 20,
    finished: 2,
    next_transfer: ["e2", "e3"],
    next_class: "near",
  };

  it("starts an evaluation with rounds and no skill field for either family", async () => {
    let started = false;
    const fetchMock = stubFetch(FORGE_ON, {
      "/evaluations": () => {
        started = true;
        return respond({ policy_family: "smolvla", version: "candidate", policy_host: "local", trials: 20 });
      },
      "/evaluations/current": () =>
        respond(started ? { ...OPEN, finished: 0, next_transfer: ["a1", "c3"] } : { open: false }),
    });
    renderPage();
    await screen.findByRole("button", { name: "Start evaluation" });
    expect(screen.queryByLabelText("Skill")).not.toBeInTheDocument();
    await userEvent.selectOptions(screen.getByLabelText("Policy family"), "act");
    expect(screen.queryByLabelText("Skill")).not.toBeInTheDocument();
    await userEvent.selectOptions(screen.getByLabelText("Policy family"), "smolvla");
    await userEvent.click(screen.getByRole("button", { name: "Start evaluation" }));

    await waitFor(() =>
      expect(posted(fetchMock, "/evaluations")).toEqual([
        { policy_family: "smolvla", version: "candidate", policy_host: "local", rounds: 1 },
      ]),
    );
    expect(await screen.findByText("Next: a1 → c3 (near) · 1 of 20")).toBeInTheDocument();
    expect(screen.queryByText(EMPTY)).not.toBeInTheDocument();
  });

  it("sends the chosen family, version, host and rounds", async () => {
    const fetchMock = stubFetch(FORGE_ON, {
      "/evaluations": () =>
        respond({ policy_family: "act", version: "production", policy_host: "coreweave", trials: 60 }),
    });
    renderPage();
    await screen.findByRole("button", { name: "Start evaluation" });
    await userEvent.selectOptions(screen.getByLabelText("Policy family"), "act");
    await userEvent.selectOptions(screen.getByLabelText("Version"), "production");
    await userEvent.selectOptions(screen.getByLabelText("Policy host"), "coreweave");
    const rounds = screen.getByLabelText("Rounds");
    await userEvent.clear(rounds);
    await userEvent.type(rounds, "3");
    await userEvent.click(screen.getByRole("button", { name: "Start evaluation" }));

    await waitFor(() =>
      expect(posted(fetchMock, "/evaluations")).toEqual([
        { policy_family: "act", version: "production", policy_host: "coreweave", rounds: 3 },
      ]),
    );
  });

  it("only starts with a whole number of rounds from 1 to 5", async () => {
    stubFetch(FORGE_ON);
    renderPage();
    const button = await screen.findByRole("button", { name: "Start evaluation" });
    const rounds = screen.getByLabelText("Rounds");
    for (const bad of ["0", "6", "1.5", ""]) {
      await userEvent.clear(rounds);
      if (bad !== "") await userEvent.type(rounds, bad);
      expect(button).toBeDisabled();
    }
    await userEvent.clear(rounds);
    await userEvent.type(rounds, "5");
    expect(button).toBeEnabled();
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

  it("offers a refresh instead of a second start when the session cannot be read after starting", async () => {
    let readable = false;
    const fetchMock = stubFetch(FORGE_ON, {
      "/evaluations": () =>
        respond({ policy_family: "smolvla", version: "candidate", policy_host: "local", trials: 20 }),
      "/evaluations/current": () =>
        readable
          ? respond({ ...OPEN, finished: 0, next_transfer: ["a1", "c3"] })
          : respond({ detail: "Could not read the evaluation." }, false, 500),
    });
    renderPage();
    await userEvent.click(await screen.findByRole("button", { name: "Start evaluation" }));

    expect(await screen.findByText("Could not read the evaluation.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start evaluation" })).toBeDisabled();
    readable = true;
    await userEvent.click(screen.getByRole("button", { name: "Refresh" }));

    expect(await screen.findByText("Next: a1 → c3 (near) · 1 of 20")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Refresh" })).not.toBeInTheDocument();
    expect(posted(fetchMock, "/evaluations")).toHaveLength(1);
  });

  it("shows an already-open session when the page loads", async () => {
    stubFetch(FORGE_ON, { "/evaluations/current": () => respond(OPEN) });
    renderPage();
    expect(await screen.findByText("Next: e2 → e3 (near) · 3 of 20")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Record outcome" })).toBeEnabled();
    expect(screen.queryByText(EMPTY)).not.toBeInTheDocument();
  });

  it("keeps no session when the page loads and none is open", async () => {
    stubFetch(FORGE_ON, { "/evaluations/current": () => respond({ open: false }) });
    renderPage();
    await screen.findByRole("button", { name: "Start evaluation" });
    expect(await screen.findByText(EMPTY)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start evaluation" })).toBeEnabled();
  });

  it("says every trial is recorded and disables Record outcome at the end of the plan", async () => {
    stubFetch(FORGE_ON, {
      "/evaluations/current": () =>
        respond({ ...OPEN, finished: 20, next_transfer: null, next_class: null }),
    });
    renderPage();
    expect(await screen.findByText("All 20 trials recorded.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Record outcome" })).toBeDisabled();
  });

  it("offers the seven outcomes", async () => {
    stubFetch(FORGE_ON, { "/evaluations/current": () => respond(OPEN) });
    renderPage();
    const select = await screen.findByLabelText("Outcome");
    const names = Array.from(select.querySelectorAll("option")).map((option) => option.textContent);
    expect(names).toEqual(["success", "approach", "grasp", "drop", "place", "topple", "disturbed"]);
  });

  describe("recording", () => {
    let current: Json;
    async function openSession(routes: Record<string, (body: Json) => Promise<Response>> = {}) {
      current = OPEN;
      const fetchMock = stubFetch(FORGE_ON, {
        "/evaluations/current": () => respond(current),
        ...routes,
      });
      renderPage();
      await screen.findByText("Next: e2 → e3 (near) · 3 of 20");
      return fetchMock;
    }

    const TRIAL = { index: 3, transfer: ["e2", "e3"], class: "near", outcome: "grasp", verdict_source: "operator" };

    it("posts the chosen outcome, shows it, and re-reads the session", async () => {
      const fetchMock = await openSession({
        "/evaluations/current/trial": () => {
          current = { ...OPEN, finished: 3, next_transfer: ["e3", "e4"], next_class: "mid" };
          return respond(TRIAL);
        },
      });
      await userEvent.selectOptions(screen.getByLabelText("Outcome"), "grasp");
      await userEvent.click(screen.getByRole("button", { name: "Record outcome" }));

      await waitFor(() =>
        expect(posted(fetchMock, "/evaluations/current/trial")).toEqual([{ outcome: "grasp", duration_s: 0 }]),
      );
      expect(await screen.findByText("Next: e3 → e4 (mid) · 4 of 20")).toBeInTheDocument();
      expect(screen.getByText("Last: e2 → e3 grasp (operator)")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Correct last" })).toBeEnabled();
    });

    it("shows the server error when recording is refused", async () => {
      await openSession({
        "/evaluations/current/trial": () => respond({ detail: "Every planned trial has been run" }, false, 400),
      });
      await userEvent.click(screen.getByRole("button", { name: "Record outcome" }));
      expect(await screen.findByText("Every planned trial has been run")).toBeInTheDocument();
    });

    it("offers Correct last for a session that already has finished trials", async () => {
      const fetchMock = await openSession({
        "/evaluations/current/override": (body) =>
          respond({ ...TRIAL, transfer: ["d2", "d4"], outcome: body.outcome, verdict_source: "override" }),
      });
      expect(screen.getByText("Latest trial: recorded before this page opened")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Correct last" })).toBeEnabled();
      await userEvent.selectOptions(screen.getByLabelText("Outcome"), "topple");
      await userEvent.click(screen.getByRole("button", { name: "Correct last" }));

      await waitFor(() =>
        expect(posted(fetchMock, "/evaluations/current/override")).toEqual([{ outcome: "topple" }]),
      );
      expect(await screen.findByText("Last: d2 → d4 topple (override)")).toBeInTheDocument();
    });

    it("clears the session panel when the server reports no open session", async () => {
      await openSession({
        "/evaluations/current/trial": () => {
          current = { open: false };
          return respond(TRIAL);
        },
      });
      await userEvent.click(screen.getByRole("button", { name: "Record outcome" }));

      expect(await screen.findByText(EMPTY)).toBeInTheDocument();
      expect(screen.queryByRole("button", { name: "Record outcome" })).not.toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Start evaluation" })).toBeEnabled();
    });

    it("posts the selected outcome to the override route on Correct last", async () => {
      const fetchMock = await openSession({
        "/evaluations/current/trial": () => respond(TRIAL),
        "/evaluations/current/override": (body) => respond({ ...TRIAL, outcome: body.outcome, verdict_source: "override" }),
      });
      await userEvent.click(screen.getByRole("button", { name: "Record outcome" }));
      await screen.findByText("Last: e2 → e3 grasp (operator)");
      await userEvent.selectOptions(screen.getByLabelText("Outcome"), "success");
      await userEvent.click(screen.getByRole("button", { name: "Correct last" }));

      await waitFor(() =>
        expect(posted(fetchMock, "/evaluations/current/override")).toEqual([{ outcome: "success" }]),
      );
      expect(await screen.findByText("Last: e2 → e3 success (override)")).toBeInTheDocument();
    });

    it("shows the server error when a correction is refused", async () => {
      await openSession({
        "/evaluations/current/trial": () => respond(TRIAL),
        "/evaluations/current/override": () => respond({ detail: "No recorded trial to override" }, false, 400),
      });
      await userEvent.click(screen.getByRole("button", { name: "Record outcome" }));
      await screen.findByText("Last: e2 → e3 grasp (operator)");
      await userEvent.click(screen.getByRole("button", { name: "Correct last" }));
      expect(await screen.findByText("No recorded trial to override")).toBeInTheDocument();
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
      expect(screen.queryByRole("button", { name: "Record outcome" })).not.toBeInTheDocument();
    });
  });

  it("promotes candidate to production for the selected family", async () => {
    const fetchMock = stubFetch(FORGE_ON, {
      "/registry/promote": () =>
        respond({ collection: "so101-chess-act", to_alias: "production", version: "v1" }),
    });
    renderPage();
    await screen.findByRole("button", { name: "Start evaluation" });
    await userEvent.selectOptions(screen.getByLabelText("Policy family"), "act");
    await userEvent.click(screen.getByRole("button", { name: "Promote candidate to production" }));

    await waitFor(() => expect(posted(fetchMock, "/registry/promote")).toEqual([{ policy_family: "act" }]));
    expect(fetchMock).toHaveBeenCalledWith(
      `${BASE}/registry/promote`,
      expect.objectContaining({ method: "POST" }),
    );
    expect(await screen.findByText("Promoted so101-chess-act v1 to production.")).toBeInTheDocument();
  });

  it("shows the server error when promotion is refused", async () => {
    stubFetch(FORGE_ON, {
      "/registry/promote": () => respond({ detail: "Registry needs WANDB_API_KEY and WANDB_ENTITY" }, false, 400),
    });
    renderPage();
    await userEvent.click(await screen.findByRole("button", { name: "Promote candidate to production" }));
    expect(await screen.findByText("Registry needs WANDB_API_KEY and WANDB_ENTITY")).toBeInTheDocument();
  });

  it("offers no promote control when forge is off", async () => {
    stubFetch({ forge_configured: false });
    renderPage();
    await screen.findByText(PLACEHOLDER);
    expect(
      screen.queryByRole("button", { name: "Promote candidate to production" }),
    ).not.toBeInTheDocument();
  });

  describe("in simulation", () => {
    const IDLE = { running: false, report: null, error: null };

    it("posts the checkpoint path and 3 seeds, and never touches the arm routes", async () => {
      const fetchMock = stubFetch(FORGE_ON, {
        "/sim-evaluate": (body) => respond(Object.keys(body).length ? { started: true } : IDLE),
      });
      renderPage();
      const button = await screen.findByRole("button", { name: "In simulation" });
      expect(button).toBeDisabled();

      await userEvent.type(screen.getByLabelText("Checkpoint path"), "/runs/act/pretrained_model");
      expect(button).toBeEnabled();
      await userEvent.click(button);

      await waitFor(() =>
        expect(posted(fetchMock, "/sim-evaluate").filter((body) => "checkpoint" in body)).toEqual([
          { checkpoint: "/runs/act/pretrained_model", seeds: 3 },
        ]),
      );
      expect(await screen.findByText("Evaluating in simulation…")).toBeInTheDocument();
      expect(button).toBeDisabled();
      expect(posted(fetchMock, "/evaluations")).toEqual([]);
    });

    it("is offered without Forge and shows a finished report", async () => {
      stubFetch(
        { forge_configured: false },
        { "/sim-evaluate": () => respond({ running: false, report: { success_rate: 0.5, trials: [{}, {}] }, error: null }) },
      );
      renderPage();
      expect(await screen.findByText(PLACEHOLDER)).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "In simulation" })).toBeInTheDocument();
      expect(await screen.findByText("Simulation: 50% of 2 trials succeeded.")).toBeInTheDocument();
    });

    it("shows the server error when the start is refused", async () => {
      stubFetch(FORGE_ON, {
        "/sim-evaluate": (body) =>
          Object.keys(body).length
            ? respond({ detail: "A simulation job is already running" }, false, 409)
            : respond(IDLE),
      });
      renderPage();
      await userEvent.type(await screen.findByLabelText("Checkpoint path"), "/runs/act");
      await userEvent.click(screen.getByRole("button", { name: "In simulation" }));
      expect(await screen.findByText("A simulation job is already running")).toBeInTheDocument();
    });
  });
});
