import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiProvider } from "@/contexts/ApiContext";
import CameraPoseCard from "./CameraPoseCard";

const BASE = "http://localhost:8000";
const PNG = new Blob([new Uint8Array([137, 80, 78, 71])], { type: "image/png" });

function fitBody(overrides: Record<string, unknown> = {}) {
  return {
    camera: "side",
    pos: [0.105, -0.3, 0.32],
    xyaxes: [1, 0, 0, 0, 1, 0],
    fovy: 47.5,
    rms_px: 0.4,
    fov_fitted: true,
    warnings: [],
    applied: false,
    ...overrides,
  };
}

/** Serves the references and the fit; `fit` is the body the next camera-pose POST returns. */
function stubFetch(camera: "top" | "side", fit: Record<string, unknown> = fitBody()) {
  const fetchMock = vi.fn((url: string, init?: RequestInit) => {
    if (url === `${BASE}/robots/arm/camera-reference/${camera}`) {
      return Promise.resolve({ ok: true, status: 200, blob: () => Promise.resolve(PNG) } as Response);
    }
    if (url === `${BASE}/chess/camera-pose` && init?.method === "POST") {
      const sent = JSON.parse(String(init.body));
      return Promise.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ ...fit, camera, applied: sent.apply }),
      } as Response);
    }
    if (url === `${BASE}/chess/board-calibration`) {
      return Promise.resolve({
        ok: true,
        status: 200,
        json: () =>
          Promise.resolve({
            robot: "arm",
            image_size: [1280, 960],
            board_corners: [[100, 100], [900, 100], [900, 800], [100, 800]],
          }),
      } as Response);
    }
    return Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve({}) } as Response);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function mountImage(img: HTMLImageElement) {
  Object.defineProperty(img, "naturalWidth", { value: 640, configurable: true });
  Object.defineProperty(img, "naturalHeight", { value: 480, configurable: true });
  img.getBoundingClientRect = () => ({ left: 0, top: 0, width: 640, height: 480, right: 640, bottom: 480, x: 0, y: 0, toJSON: () => ({}) });
}

async function renderCard(camera: "top" | "side") {
  render(
    <ApiProvider>
      <CameraPoseCard robot="arm" camera={camera} />
    </ApiProvider>,
  );
  const img = (await screen.findByAltText(`${camera} camera reference`)) as HTMLImageElement;
  mountImage(img);
  return img;
}

const CLICKS: Array<[number, number]> = [[100, 80], [540, 90], [500, 400], [120, 380]];

function clickAll(img: HTMLImageElement) {
  for (const [x, y] of CLICKS) fireEvent.click(img, { clientX: x, clientY: y });
}

function posts(fetchMock: ReturnType<typeof stubFetch>) {
  return fetchMock.mock.calls
    .filter(([url, init]) => url === `${BASE}/chess/camera-pose` && (init as RequestInit)?.method === "POST")
    .map(([, init]) => JSON.parse(String((init as RequestInit).body)));
}

beforeEach(() => {
  URL.createObjectURL = vi.fn(() => "blob:ref");
  URL.revokeObjectURL = vi.fn();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("CameraPoseCard", () => {
  it("fits the four clicked corners and shows the result", async () => {
    const fetchMock = stubFetch("side");
    const img = await renderCard("side");
    const fit = screen.getByRole("button", { name: "Fit" });
    expect(fit).toBeDisabled();
    expect(screen.getByText("Click a1")).toBeInTheDocument();

    clickAll(img);
    expect(fit).toBeEnabled();
    fireEvent.click(fit);

    await screen.findByText(/fitted/);
    expect(posts(fetchMock)).toEqual([
      { camera: "side", image_size: [640, 480], board_corners: CLICKS, fovy: null, apply: false },
    ]);
    expect(screen.getByText(/Position 105 -300 320 mm/)).toBeInTheDocument();
    expect(screen.getByText(/error 0.4 px/)).toBeInTheDocument();
  });

  it("writes only after a fit, and not when the fit is too far from the clicks", async () => {
    const fetchMock = stubFetch("side");
    const img = await renderCard("side");
    const write = screen.getByRole("button", { name: "Write to sim cell" });
    expect(write).toBeDisabled();

    clickAll(img);
    fireEvent.click(screen.getByRole("button", { name: "Fit" }));
    await waitFor(() => expect(write).toBeEnabled());
    fireEvent.click(write);
    await screen.findByText(/Written to the sim cell/);
    expect(posts(fetchMock).map((p) => p.apply)).toEqual([false, true]);

    vi.unstubAllGlobals();
    stubFetch("side", fitBody({ rms_px: 3.5, warnings: ["check the click order"] }));
    fireEvent.click(screen.getByRole("button", { name: "Reset points" }));
    clickAll(img);
    fireEvent.click(screen.getByRole("button", { name: "Fit" }));
    await screen.findByText("check the click order");
    expect(write).toBeDisabled();
  });

  it("fills the top camera's corners from the saved board calibration", async () => {
    const fetchMock = stubFetch("top");
    await renderCard("top");
    fireEvent.click(screen.getByRole("button", { name: "Use saved board corners" }));

    const fit = screen.getByRole("button", { name: "Fit" });
    await waitFor(() => expect(fit).toBeEnabled());
    fireEvent.click(fit);
    await screen.findByText(/fitted/);
    // The calibration was clicked on a 1280x960 reference; this one is 640x480.
    expect(posts(fetchMock)[0].board_corners).toEqual([[50, 50], [450, 50], [450, 400], [50, 400]]);
  });

  it("offers saved corners for the top camera only", async () => {
    stubFetch("side");
    await renderCard("side");
    expect(screen.queryByRole("button", { name: "Use saved board corners" })).toBeNull();
  });

  it("explains when the reference is missing", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve({}) } as Response)),
    );
    render(
      <ApiProvider>
        <CameraPoseCard robot="arm" camera="side" />
      </ApiProvider>,
    );
    expect(await screen.findByText(/Capture the side camera reference on the Calibration page/)).toBeInTheDocument();
  });
});
