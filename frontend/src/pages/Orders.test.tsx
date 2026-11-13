import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiProvider } from "@/contexts/ApiContext";
import Orders from "./Orders";

const BASE = "http://localhost:8000";
const PLACEHOLDER = "The order-picking agent arrives in a later milestone.";
const EMPTY = "Type an order, then the agent picks each part into the tray.";
const HINT = "Add CWSANDBOX_API_KEY and CWSANDBOX_TRAIN_IMAGE to the backend environment.";

type Json = Record<string, unknown>;

// @testing-library/user-event is not installed, so drive the DOM with fireEvent.
const userEvent = {
  click: async (element: HTMLElement) => {
    fireEvent.click(element);
  },
  type: async (element: HTMLElement, text: string) => {
    fireEvent.change(element, { target: { value: (element as HTMLInputElement).value + text } });
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
    if (path === "/orders/current") return respond({ status: "idle" });
    return respond({}, false, 404);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function posted(fetchMock: ReturnType<typeof vi.fn>, path: string): Json[] {
  return fetchMock.mock.calls
    .filter((call) => {
      const [url, init] = call as [string, RequestInit | undefined];
      return String(url) === `${BASE}${path}` && init?.method === "POST";
    })
    .map((call) => JSON.parse(String(((call as unknown[])[1] as RequestInit).body ?? "{}")) as Json);
}

function renderPage() {
  return render(
    <MemoryRouter>
      <ApiProvider>
        <Orders />
      </ApiProvider>
    </MemoryRouter>,
  );
}

const FORGE_ON = { forge_configured: true, sandbox_configured: true };

async function startOrder(routes: Record<string, (body: Json) => Promise<Response>> = {}) {
  const fetchMock = stubFetch(FORGE_ON, {
    "/orders": () => respond({ status: "running", counts: { bolt: 1 } }),
    "/orders/current": () => respond({ status: "running", counts: { bolt: 1 }, trace_url: null }),
    ...routes,
  });
  renderPage();
  await userEvent.type(await screen.findByLabelText("Order"), "one bolt");
  await userEvent.click(screen.getByRole("button", { name: "Start order" }));
  return fetchMock;
}

describe("Orders", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shows the placeholder sentence when forge is not configured", async () => {
    stubFetch({ forge_configured: false });
    renderPage();
    expect(await screen.findByText(PLACEHOLDER)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Start order" })).not.toBeInTheDocument();
  });

  it("treats a truthy non-true forge_configured as off", async () => {
    stubFetch({ forge_configured: "yes" });
    renderPage();
    expect(await screen.findByText(PLACEHOLDER)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Start order" })).not.toBeInTheDocument();
  });

  it("shows the controls and the empty state when forge is configured", async () => {
    stubFetch(FORGE_ON);
    renderPage();
    expect(await screen.findByRole("button", { name: "Start order" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Orders" })).toBeInTheDocument();
    expect(screen.getByText(EMPTY)).toBeInTheDocument();
    expect(screen.getByLabelText("Policy family")).toHaveDisplayValue("SmolVLA");
    expect(screen.getByLabelText("Policy host")).toHaveDisplayValue("Local");
    expect(screen.queryByText(PLACEHOLDER)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Stop" })).not.toBeInTheDocument();
  });

  it("disables CoreWeave and shows the hint unless the sandbox is strictly configured", async () => {
    stubFetch({ forge_configured: true, sandbox_configured: "yes" });
    renderPage();
    await screen.findByRole("button", { name: "Start order" });
    expect(screen.getByRole("option", { name: "CoreWeave" })).toBeDisabled();
    expect(screen.getByText(HINT)).toBeInTheDocument();
  });

  it("enables CoreWeave and hides the hint when the sandbox is configured", async () => {
    stubFetch(FORGE_ON);
    renderPage();
    await screen.findByRole("button", { name: "Start order" });
    expect(screen.getByRole("option", { name: "CoreWeave" })).toBeEnabled();
    expect(screen.queryByText(HINT)).not.toBeInTheDocument();
  });

  it("keeps Start order disabled until the order has text", async () => {
    stubFetch(FORGE_ON);
    renderPage();
    const button = await screen.findByRole("button", { name: "Start order" });
    expect(button).toBeDisabled();
    await userEvent.type(screen.getByLabelText("Order"), "one bolt");
    expect(button).toBeEnabled();
  });

  it("posts the order, then shows the count, the status word and Stop", async () => {
    const fetchMock = await startOrder();

    await waitFor(() =>
      expect(posted(fetchMock, "/orders")).toEqual([
        { text: "one bolt", policy_family: "smolvla", policy_host: "local" },
      ]),
    );
    expect(await screen.findByText("bolt: 1")).toBeInTheDocument();
    expect(screen.getByText("Status: running")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Stop" })).toBeInTheDocument();
    expect(screen.queryByText(EMPTY)).not.toBeInTheDocument();
  });

  it("posts to the stop route when Stop is clicked", async () => {
    const fetchMock = await startOrder({
      "/orders/current/stop": () => respond({ status: "running" }),
    });
    await userEvent.click(await screen.findByRole("button", { name: "Stop" }));
    await waitFor(() => expect(posted(fetchMock, "/orders/current/stop")).toHaveLength(1));
  });

  it("shows the server error when the start is refused", async () => {
    await startOrder({ "/orders": () => respond({ detail: "An order is already running" }, false, 409) });
    expect(await screen.findByText("An order is already running")).toBeInTheDocument();
    expect(screen.getByText(EMPTY)).toBeInTheDocument();
  });

  it("links to the trace only when trace_url is a non-empty string", async () => {
    await startOrder({
      "/orders/current": () =>
        respond({ status: "done", counts: { bolt: 1 }, trace_url: "https://example.test/trace" }),
    });
    const link = await screen.findByRole("link", { name: "Open trace in Agent Lens" });
    expect(link).toHaveAttribute("href", "https://example.test/trace");
    expect(screen.getByText("Status: done")).toBeInTheDocument();
  });

  it("shows no trace link when trace_url is null", async () => {
    await startOrder({
      "/orders/current": () => respond({ status: "done", counts: { bolt: 1 }, trace_url: null }),
    });
    expect(await screen.findByText("Status: done")).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Open trace in Agent Lens" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Stop" })).not.toBeInTheDocument();
  });

  it("labels the three camera tiles and says No frame yet in each", async () => {
    stubFetch(FORGE_ON);
    renderPage();
    await screen.findByRole("button", { name: "Start order" });
    for (const name of ["Wrist", "Top", "Side"]) {
      const tile = screen.getByRole("group", { name });
      expect(tile).toHaveTextContent("No frame yet");
    }
    expect(screen.getAllByText("No frame yet")).toHaveLength(3);
  });
});
