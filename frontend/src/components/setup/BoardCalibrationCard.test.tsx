import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiProvider } from "@/contexts/ApiContext";
import type { CameraConfig } from "@/lib/cameraConfig";
import BoardCalibrationCard from "./BoardCalibrationCard";

const captureCorrectedFrame = vi.hoisted(() => vi.fn());
vi.mock("@/lib/cameraReferenceApi", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/cameraReferenceApi")>()),
  captureCorrectedFrame,
}));
const liveCameras = vi.hoisted(() => ({ value: [] as Array<{ index: number; name: string; deviceId: string; available: boolean }> }));
vi.mock("@/hooks/useAvailableCameras", () => ({
  useAvailableCameras: () => ({ cameras: [], isLoading: false, refresh: () => Promise.resolve(liveCameras.value) }),
}));

const BASE = "http://localhost:8000";
const PNG = new Blob([new Uint8Array([137, 80, 78, 71])], { type: "image/png" });
const CALIBRATION = {
  robot: "arm",
  image_size: [640, 480],
  board_corners: [[0, 0], [1, 0], [1, 1], [0, 1]],
  bin: [[0, 0], [1, 0], [1, 1], [0, 1]],
  park_white: [[0, 0], [1, 0], [1, 1], [0, 1]],
  park_black: [[0, 0], [1, 0], [1, 1], [0, 1]],
  pixel_diff_threshold: 40,
  occupied_fraction: 0.15,
  colour_threshold: 128,
};
const TOP: CameraConfig = {
  id: "top-id",
  name: "top",
  type: "opencv",
  camera_index: 2,
  width: 640,
  height: 480,
  rotation: 0,
  device_id: "top-device",
  color_correction: false,
};

function stubFetch(calibration: unknown | null) {
  const fetchMock = vi.fn((url: string, init?: RequestInit) => {
    if (url === `${BASE}/robots/arm/camera-reference/top`) {
      return Promise.resolve({ ok: true, status: 200, blob: () => Promise.resolve(PNG) } as Response);
    }
    if (url === `${BASE}/chess/board-calibration` && (!init || !init.method || init.method === "GET")) {
      return calibration
        ? Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(calibration) } as Response)
        : Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve({}) } as Response);
    }
    if (url === `${BASE}/chess/board-calibration` && init?.method === "PUT") {
      return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ status: "success" }) } as Response);
    }
    if (url === `${BASE}/chess/board-state` && init?.method === "POST") {
      return Promise.resolve({
        ok: true,
        status: 200,
        json: () =>
          Promise.resolve({
            robot: "arm",
            squares: { a1: "white", h8: "black" },
            extras: { bin: "empty", park_white: "empty", park_black: "empty" },
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

/** A camera that opens and plays at `width`×`height`; returns the getUserMedia and track-stop mocks. */
function stubCamera(width: number, height: number) {
  const stop = vi.fn();
  const stream = { getTracks: () => [{ stop }], getVideoTracks: () => [{ label: "Top Cam" }] };
  const getUserMedia = vi.fn(() => Promise.resolve(stream));
  vi.stubGlobal("navigator", { ...navigator, mediaDevices: { getUserMedia } });
  vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue(undefined);
  vi.spyOn(HTMLVideoElement.prototype, "videoWidth", "get").mockReturnValue(width);
  vi.spyOn(HTMLVideoElement.prototype, "videoHeight", "get").mockReturnValue(height);
  return { getUserMedia, stop };
}

async function renderReady(topCamera: CameraConfig | null) {
  render(
    <ApiProvider>
      <BoardCalibrationCard robot="arm" topCamera={topCamera} />
    </ApiProvider>,
  );
  mountImage((await screen.findByAltText("top camera reference")) as HTMLImageElement);
  const check = screen.getByRole("button", { name: "Check position" });
  await waitFor(() => expect(check).toBeEnabled());
  return check;
}

function boardStatePosts(fetchMock: ReturnType<typeof stubFetch>) {
  return fetchMock.mock.calls.filter(
    ([url, init]) => url === `${BASE}/chess/board-state` && (init as RequestInit)?.method === "POST",
  );
}

beforeEach(() => {
  // jsdom has no object URLs; the card only needs a string for <img src>.
  URL.createObjectURL = vi.fn(() => "blob:ref");
  URL.revokeObjectURL = vi.fn();
  captureCorrectedFrame.mockReset();
  liveCameras.value = [];
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("BoardCalibrationCard", () => {
  it("collects sixteen clicks in order and saves them", async () => {
    const fetchMock = stubFetch(null);
    render(
      <ApiProvider>
        <BoardCalibrationCard robot="arm" topCamera={null} />
      </ApiProvider>,
    );
    const img = (await screen.findByAltText("top camera reference")) as HTMLImageElement;
    mountImage(img);
    expect(screen.getByText(/Click a1/)).toBeInTheDocument();
    expect(screen.getByText(/Click on the straightened reference; recapture it after changing the lens/)).toBeInTheDocument();
    const save = screen.getByRole("button", { name: "Save calibration" });
    expect(save).toBeDisabled();

    const points: Array<[number, number]> = [];
    for (let i = 0; i < 16; i++) points.push([20 + i * 30, 40 + (i % 4) * 50]);
    for (const [x, y] of points) fireEvent.click(img, { clientX: x, clientY: y });

    expect(screen.getByText(/All 16 points set/)).toBeInTheDocument();
    expect(save).toBeEnabled();
    fireEvent.click(save);

    await waitFor(() => expect(screen.getByText(/Saved/)).toBeInTheDocument());
    const put = fetchMock.mock.calls.find(([, init]) => (init as RequestInit)?.method === "PUT");
    expect(put).toBeDefined();
    const body = JSON.parse(String((put![1] as RequestInit).body));
    expect(body.robot).toBe("arm");
    expect(body.image_size).toEqual([640, 480]);
    expect(body.board_corners).toEqual(points.slice(0, 4));
    expect(body.bin).toEqual(points.slice(4, 8));
    expect(body.park_white).toEqual(points.slice(8, 12));
    expect(body.park_black).toEqual(points.slice(12, 16));
  });

  it("explains when the top reference is missing", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve({}) } as Response)),
    );
    render(
      <ApiProvider>
        <BoardCalibrationCard robot="arm" topCamera={null} />
      </ApiProvider>,
    );
    expect(await screen.findByText(/Capture the top camera reference/)).toBeInTheDocument();
  });

  it("checks the position at the camera's size with a frame corrected like the top reference", async () => {
    const fetchMock = stubFetch(CALIBRATION);
    const { getUserMedia, stop } = stubCamera(640, 480);
    const frame = new Blob(["frame"], { type: "image/png" });
    captureCorrectedFrame.mockResolvedValue(frame);

    fireEvent.click(await renderReady(TOP));

    await waitFor(() => expect(screen.getByTitle("a1: white")).toBeInTheDocument());
    expect(screen.getByTitle("h8: black")).toBeInTheDocument();
    expect(getUserMedia).toHaveBeenCalledWith({
      video: { deviceId: { exact: "top-device" }, width: { ideal: 640 }, height: { ideal: 480 } },
    });
    expect(captureCorrectedFrame).toHaveBeenCalledWith(
      expect.any(HTMLVideoElement),
      BASE,
      expect.any(Function),
      { gain: false, lens: true },
    );
    expect((boardStatePosts(fetchMock)[0][1] as RequestInit).body).toBe(frame);
    expect(stop).toHaveBeenCalled();
  });

  it("opens the live camera matching the saved index when the saved device id is unknown here", async () => {
    stubFetch(CALIBRATION);
    liveCameras.value = [
      { index: 0, name: "Wrist", deviceId: "live-wrist", available: true },
      { index: 2, name: "Top Cam", deviceId: "live-top", available: true },
    ];
    const { getUserMedia } = stubCamera(640, 480);
    captureCorrectedFrame.mockResolvedValue(new Blob(["frame"]));

    fireEvent.click(await renderReady({ ...TOP, device_id: "other-browser-id" }));

    await waitFor(() => expect(screen.getByTitle("a1: white")).toBeInTheDocument());
    expect(getUserMedia).toHaveBeenCalledWith({
      video: { deviceId: { exact: "live-top" }, width: { ideal: 640 }, height: { ideal: 480 } },
    });
  });

  it("refuses a frame whose size differs from the reference and posts nothing", async () => {
    const fetchMock = stubFetch(CALIBRATION);
    stubCamera(1280, 720);
    captureCorrectedFrame.mockResolvedValue(new Blob(["frame"]));

    fireEvent.click(await renderReady(TOP));

    expect(
      await screen.findByText(
        "The top camera delivered 1280×720 but the reference is 640×480; recapture the reference or fix the camera size.",
      ),
    ).toBeInTheDocument();
    expect(captureCorrectedFrame).not.toHaveBeenCalled();
    expect(boardStatePosts(fetchMock)).toEqual([]);
  });

  it("names the error when the camera fails without a message", async () => {
    stubFetch(CALIBRATION);
    const getUserMedia = vi.fn(() => Promise.reject(Object.assign(new Error(""), { name: "OverconstrainedError" })));
    vi.stubGlobal("navigator", { ...navigator, mediaDevices: { getUserMedia } });

    fireEvent.click(await renderReady(TOP));

    expect(await screen.findByText("OverconstrainedError")).toBeInTheDocument();
  });

  it("does not treat another robot's calibration as this one's", async () => {
    stubFetch({ ...CALIBRATION, robot: "other_arm" });
    render(
      <ApiProvider>
        <BoardCalibrationCard robot="arm" topCamera={TOP} />
      </ApiProvider>,
    );
    expect(
      await screen.findByText("The saved board calibration belongs to other_arm; calibrate this one."),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Check position" })).toBeDisabled();
  });
});
