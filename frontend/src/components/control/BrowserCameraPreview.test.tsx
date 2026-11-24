// @vitest-environment jsdom
import React, { act } from "react";
import { createRoot, Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import BrowserCameraPreview from "./BrowserCameraPreview";
import CameraRotationSelect from "./CameraRotationSelect";
vi.mock("@/hooks/useCameraStream", () => ({ useCameraStream: () => ({ videoRef: React.createRef<HTMLVideoElement>(), hasError: false, label: "USB Camera" }) }));
const savedGain = vi.hoisted(() => ({ value: null as { red: number; green: number; blue: number } | null }));
vi.mock("@/hooks/useCameraGain", () => ({ useCameraGain: () => savedGain.value }));
const savedLens = vi.hoisted(() => ({ value: null as { fov: number; bend: number; zoom: number } | null }));
vi.mock("@/hooks/useCameraLens", () => ({ useCameraLens: () => savedLens.value }));
const renderer = vi.hoisted(() => ({ create: vi.fn() }));
vi.mock("@/lib/lensRenderer", () => ({ createLensRenderer: renderer.create }));
let root: Root;
let container: HTMLDivElement;
beforeEach(() => { Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true }); container = document.createElement("div"); root = createRoot(container); savedGain.value = null; savedLens.value = null; renderer.create.mockReset(); });
afterEach(() => act(() => root.unmount()));
test("the Rotation control exposes four clockwise values and emits a number", () => {
  const change = vi.fn();
  act(() => root.render(<CameraRotationSelect rotation={180} onChange={change} />));
  const select = container.querySelector("select")!;
  expect(container.querySelector("label")!.htmlFor).toBe(select.id);
  expect(select.value).toBe("180");
  expect(Array.from(select.options).map((option) => option.text)).toEqual(["0° clockwise", "90° clockwise", "180° clockwise", "270° clockwise"]);
  act(() => { select.value = "90"; select.dispatchEvent(new Event("change", { bubbles: true })); });
  expect(change).toHaveBeenCalledWith(90);
});
test("a quarter turn fills the rotated recorded frame", () => {
  act(() => root.render(<BrowserCameraPreview deviceId="wrist" rotation={90} width={640} height={480} />));
  const video = container.querySelector("video")!, wrapper = video.parentElement!;
  expect(wrapper.style.aspectRatio).toBe("480 / 640");
  expect(video.style.objectFit).toBe("cover");
  expect(video.style.maxWidth).toBe("none");
  expect(video.style.transform).toBe("translate(-50%, -50%) rotate(90deg)");
  expect(parseFloat(video.style.width)).toBeCloseTo(640 / 480 * 100);
  expect(parseFloat(video.style.height)).toBeCloseTo(480 / 640 * 100);
});
test("a stream at another size keeps the recorded aspect ratio", () => {
  act(() => root.render(<BrowserCameraPreview deviceId="A" rotation={0} width={640} height={480} />));
  const video = container.querySelector("video")!;
  Object.defineProperty(video, "videoWidth", { value: 1280 });
  Object.defineProperty(video, "videoHeight", { value: 720 });
  act(() => video.dispatchEvent(new Event("loadedmetadata")));
  expect(video.parentElement!.style.aspectRatio).toBe("640 / 480");
  expect(video.style.width).toBe("100%");
  expect(video.style.objectFit).toBe("cover");
});
test("the reference ghost sits over the video with its rotation and opacity", () => {
  act(() => root.render(<BrowserCameraPreview deviceId="wrist" rotation={90} ghostSrc="blob:ref" ghostOpacity={0.3} />));
  const video = container.querySelector("video")!, ghost = container.querySelector("img")!;
  expect(ghost.getAttribute("src")).toBe("blob:ref");
  expect(ghost.style.opacity).toBe("0.3");
  expect(ghost.style.transform).toBe(video.style.transform);
  expect(ghost.style.width).toBe(video.style.width);
  act(() => root.render(<BrowserCameraPreview deviceId="wrist" />));
  expect(container.querySelector("img")).toBeNull();
});
test("a camera with a saved gain previews through the same colour matrix the recorder uses", () => {
  savedGain.value = { red: 1.4464, green: 1, blue: 1.7868 };
  act(() => root.render(<BrowserCameraPreview deviceId="top" rotation={0} />));
  const video = container.querySelector("video")!;
  const filter = container.querySelector("filter")!;
  expect(video.style.filter).toContain(`#${filter.id}`);
  expect(filter.getAttribute("color-interpolation-filters")).toBe("sRGB");
  expect(filter.querySelector("feColorMatrix")!.getAttribute("values")).toBe(
    "1.4464 0 0 0 0 0 1 0 0 0 0 0 1.7868 0 0 0 0 0 1 0",
  );
});

test("a camera with colour correction switched off previews unfiltered", () => {
  savedGain.value = { red: 1.4464, green: 1, blue: 1.7868 };
  act(() => root.render(<BrowserCameraPreview deviceId="top" colorCorrection={false} />));
  expect(container.querySelector("video")!.style.filter).toBe("");
  expect(container.querySelector("filter")).toBeNull();
});

test("a camera without a saved gain previews unfiltered", () => {
  act(() => root.render(<BrowserCameraPreview deviceId="top" rotation={0} />));
  expect(container.querySelector("video")!.style.filter).toBe("");
  expect(container.querySelector("filter")).toBeNull();
});

test("the reference ghost is not filtered, since the saved reference is already corrected", () => {
  savedGain.value = { red: 1.4464, green: 1, blue: 1.7868 };
  act(() => root.render(<BrowserCameraPreview deviceId="top" ghostSrc="blob:ref" />));
  expect(container.querySelector("img")!.style.filter).toBe("");
});

test("a saved lens is drawn through the renderer while the stream's video stays in the page, hidden", () => {
  const draw = vi.fn(), dispose = vi.fn();
  renderer.create.mockReturnValue({ draw, dispose });
  savedLens.value = { fov: 135, bend: 0, zoom: 0.9 };
  const frames: FrameRequestCallback[] = [];
  vi.spyOn(window, "requestAnimationFrame").mockImplementation((callback) => frames.push(callback));
  vi.spyOn(window, "cancelAnimationFrame").mockImplementation(() => undefined);
  act(() => root.render(<BrowserCameraPreview deviceId="A" width={640} height={480} />));
  const video = container.querySelector("video")!;
  expect(container.querySelector('[data-testid="lens-canvas"]')).not.toBeNull();
  expect(video.style.opacity).toBe("0");
  expect(video.style.pointerEvents).toBe("none");
  frames.shift()!(0); // no frame yet: nothing is drawn
  expect(draw).not.toHaveBeenCalled();
  Object.defineProperty(video, "readyState", { value: 4 });
  frames.shift()!(0);
  expect(draw).toHaveBeenCalledWith(video, { fov: 135, bend: 0, zoom: 0.9 });
  act(() => root.unmount());
  expect(dispose).toHaveBeenCalled();
  root = createRoot(container);
});
test("without a saved lens, or with the switch off, the plain video is shown", () => {
  act(() => root.render(<BrowserCameraPreview deviceId="A" width={640} height={480} />));
  expect(container.querySelector('[data-testid="lens-canvas"]')).toBeNull();
  expect(container.querySelector("video")!.style.opacity).toBe("");
  savedLens.value = { fov: 135, bend: 0, zoom: 0.9 };
  act(() => root.render(<BrowserCameraPreview deviceId="A" width={640} height={480} lensCorrection={false} />));
  expect(container.querySelector('[data-testid="lens-canvas"]')).toBeNull();
  expect(renderer.create).not.toHaveBeenCalled();
});
test("a browser without WebGL falls back to the plain video and says why", () => {
  renderer.create.mockReturnValue(null);
  savedLens.value = { fov: 135, bend: 0, zoom: 0.9 };
  act(() => root.render(<BrowserCameraPreview deviceId="A" width={640} height={480} />));
  expect(container.querySelector('[data-testid="lens-canvas"]')).toBeNull();
  expect(container.querySelector("video")!.style.opacity).toBe("");
  expect(container.textContent).toContain("Lens preview needs WebGL");
});
