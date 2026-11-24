import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ChessDatasetCheck from "./ChessDatasetCheck";

const fetcher = vi.fn();
vi.mock("@/contexts/ApiContext", () => ({
  useApi: () => ({ baseUrl: "http://api", fetchWithHeaders: fetcher }),
}));

function respond(body: unknown) {
  fetcher.mockResolvedValueOnce({ ok: true, status: 200, json: () => Promise.resolve(body) } as Response);
}

describe("ChessDatasetCheck", () => {
  beforeEach(() => fetcher.mockReset());

  it("says a clean dataset is chess transfer data", async () => {
    respond({ checked: true, repo_id: "local/sim", problems: [], mismatches: [], message: null });
    render(<ChessDatasetCheck repoId="local/sim" />);
    expect(await screen.findByText("Chess transfer data: markers, sidecar and task string present.")).toBeInTheDocument();
    expect(fetcher).toHaveBeenCalledWith("http://api/datasets/chess-check?repo_id=local%2Fsim");
  });

  it("lists the problems of a dataset that is not chess transfer data", async () => {
    respond({ checked: true, repo_id: "local/real", problems: ["2 of 2 episodes have no transfer"], mismatches: [], message: null });
    render(<ChessDatasetCheck repoId="local/real" />);
    expect(await screen.findByRole("alert")).toHaveTextContent("2 of 2 episodes have no transfer");
  });

  it("shows nothing for a dataset that is not local", async () => {
    respond({ checked: false, repo_id: "a/b", problems: [], mismatches: [], message: "not local" });
    const { container } = render(<ChessDatasetCheck repoId="a/b" />);
    await waitFor(() => expect(fetcher).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
  });
});
