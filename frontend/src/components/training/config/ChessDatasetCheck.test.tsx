import { fireEvent, render, screen, waitFor } from "@testing-library/react";
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

  describe("policy cameras", () => {
    const sim = { checked: true, repo_id: "local/sim", problems: [], mismatches: [], cameras: ["top", "wrist", "side"] };

    it("preselects top and wrist and reports them, because side is only recorded", async () => {
      respond(sim);
      const onPolicyCameras = vi.fn();
      render(<ChessDatasetCheck repoId="local/sim" onPolicyCameras={onPolicyCameras} />);
      expect(await screen.findByLabelText("top")).toBeChecked();
      expect(screen.getByLabelText("wrist")).toBeChecked();
      expect(screen.getByLabelText("side")).not.toBeChecked();
      expect(onPolicyCameras).toHaveBeenLastCalledWith(["top", "wrist"]);
    });

    it("sends nothing when every camera of the dataset is selected", async () => {
      respond(sim);
      const onPolicyCameras = vi.fn();
      render(<ChessDatasetCheck repoId="local/sim" onPolicyCameras={onPolicyCameras} />);
      fireEvent.click(await screen.findByLabelText("side"));
      expect(onPolicyCameras).toHaveBeenLastCalledWith(undefined);
      fireEvent.click(screen.getByLabelText("top"));
      expect(onPolicyCameras).toHaveBeenLastCalledWith(["wrist", "side"]);
    });

    it("keeps the last camera selected", async () => {
      respond({ ...sim, cameras: ["top", "wrist"] });
      render(<ChessDatasetCheck repoId="local/sim" onPolicyCameras={vi.fn()} />);
      fireEvent.click(await screen.findByLabelText("top"));
      expect(screen.getByLabelText("wrist")).toBeDisabled();
    });

    it("sends nothing for a two-camera dataset read in full, and offers no choice for one camera", async () => {
      respond({ ...sim, cameras: ["top", "wrist"] });
      const onPolicyCameras = vi.fn();
      const { unmount } = render(<ChessDatasetCheck repoId="local/real" onPolicyCameras={onPolicyCameras} />);
      await screen.findByLabelText("top");
      expect(onPolicyCameras).toHaveBeenLastCalledWith(undefined);
      unmount();

      respond({ ...sim, cameras: ["top"] });
      render(<ChessDatasetCheck repoId="local/one" onPolicyCameras={onPolicyCameras} />);
      await screen.findByText(/Chess transfer data/);
      expect(screen.queryByLabelText("top")).toBeNull();
    });
  });
});
