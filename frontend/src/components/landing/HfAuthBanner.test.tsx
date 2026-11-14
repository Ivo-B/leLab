import { act, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiProvider } from "@/contexts/ApiContext";
import { HfAuthProvider } from "@/contexts/HfAuthContext";
import HfAuthBanner from "./HfAuthBanner";

const BASE = "http://localhost:8000";
const DEFAULT_TITLE = "Hugging Face access required for cloud training";
const OWNER_TITLE = "Log in to Hugging Face to set the repository owner";

function stubAuthStatus(body: unknown) {
  const fetchMock = vi.fn((url: unknown) => {
    if (String(url) === `${BASE}/hf-auth-status`) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) } as Response);
    }
    return Promise.reject(new Error(`unexpected fetch: ${String(url)}`));
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function renderBanner(title?: string) {
  return render(
    <ApiProvider>
      <HfAuthProvider>
        <HfAuthBanner title={title} />
      </HfAuthProvider>
    </ApiProvider>,
  );
}

describe("HfAuthBanner", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shows the default title when unauthenticated", async () => {
    stubAuthStatus({ authenticated: false, login_command: "hf auth login" });
    renderBanner();

    expect(await screen.findByText(DEFAULT_TITLE)).toBeTruthy();
  });

  it("shows the title prop instead of the default when unauthenticated", async () => {
    stubAuthStatus({ authenticated: false, login_command: "hf auth login" });
    renderBanner(OWNER_TITLE);

    expect(await screen.findByText(OWNER_TITLE)).toBeTruthy();
    expect(screen.queryByText(DEFAULT_TITLE)).toBeNull();
    expect(screen.getByText(/so datasets and trained policies can upload/)).toBeTruthy();
  });

  it("renders nothing when authenticated", async () => {
    const fetchMock = stubAuthStatus({ authenticated: true, username: "ivo", orgs: [] });
    const { container } = renderBanner();

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    // Let the provider's fetch and json promises settle so the status leaves "loading".
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(screen.queryByText(DEFAULT_TITLE)).toBeNull();
    expect(container.firstChild).toBeNull();
  });
});
