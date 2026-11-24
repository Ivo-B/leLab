// @vitest-environment jsdom
import React, { act } from "react";
import { createRoot, Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import CameraConfiguration from "./CameraConfiguration";
import { CameraConfig, normalizeCameraConfig, serializeCameras } from "@/lib/cameraConfig";
vi.mock("@/hooks/useAvailableCameras", () => ({ useAvailableCameras: () => ({ cameras: [], isLoading: false, refresh: vi.fn() }) }));
vi.mock("@/hooks/use-toast", () => ({ useToast: () => ({ toast: vi.fn() }) }));
const reference = { url: null as string | null, error: null, save: vi.fn(), clear: vi.fn() };
vi.mock("@/hooks/useCameraReference", () => ({ useCameraReference: () => reference }));
vi.mock("@/hooks/useCameraStream", () => ({ useCameraStream: () => ({ videoRef: React.createRef<HTMLVideoElement>(), hasError: false }) }));
const savedLens = vi.hoisted(() => ({ value: null as { fov: number; bend: number; zoom: number } | null }));
vi.mock("@/hooks/useCameraLens", () => ({ useCameraLens: () => savedLens.value }));
let root: Root;
let container: HTMLDivElement;
const camera: CameraConfig = { id: "wrist", name: "wrist", device_id: "usb-wrist", type: "opencv", width: 640, height: 480 };
beforeEach(() => { Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true }); container = document.createElement("div"); root = createRoot(container); savedLens.value = null; });
afterEach(() => act(() => root.unmount()));
test("the colour correction switch is on until turned off", () => {
  const change = vi.fn();
  act(() => root.render(<CameraConfiguration cameras={[camera]} onCamerasChange={change} />));
  const toggle = container.querySelector<HTMLButtonElement>('[role="switch"]')!;
  expect(toggle.getAttribute("data-state")).toBe("checked");
  act(() => toggle.click());
  expect(change).toHaveBeenCalledWith([{ ...camera, color_correction: false }]);
});
test("the lens correction switch is on until turned off, and leaves the colour switch alone", () => {
  const change = vi.fn();
  act(() => root.render(<CameraConfiguration cameras={[camera]} onCamerasChange={change} />));
  const toggle = container.querySelector<HTMLButtonElement>('[role="switch"][aria-label="Lens correction"]')!;
  expect(toggle.getAttribute("data-state")).toBe("checked");
  expect(container.textContent).toContain("On · no saved lens for this camera");
  act(() => toggle.click());
  expect(change).toHaveBeenCalledWith([{ ...camera, lens_correction: false }]);
});
test("a saved lens is named in the status and an off switch says off", () => {
  savedLens.value = { fov: 135, bend: 0, zoom: 0.9 };
  act(() => root.render(<CameraConfiguration cameras={[camera]} onCamerasChange={vi.fn()} />));
  expect(container.textContent).toContain("On · 135° field of view");
  act(() => root.render(<CameraConfiguration cameras={[{ ...camera, lens_correction: false }]} onCamerasChange={vi.fn()} />));
  expect(container.textContent).toContain("Lens correction");
  expect(container.textContent).toMatch(/Lens correction\s*Off/);
});
test("rotation updates the profile while capture dimensions stay native", () => {
  const change = vi.fn();
  act(() => root.render(<CameraConfiguration cameras={[camera]} onCamerasChange={change} />));
  const select = container.querySelector("select")!;
  expect(select.value).toBe("0");
  act(() => { select.value = "90"; select.dispatchEvent(new Event("change", { bubbles: true })); });
  expect(change).toHaveBeenCalledWith([{ ...camera, rotation: 90 }]);
  const saved = JSON.parse(JSON.stringify(change.mock.calls[0][0])).map(normalizeCameraConfig);
  act(() => root.render(<CameraConfiguration cameras={saved} onCamerasChange={change} />));
  expect(container.querySelector("select")!.value).toBe("90");
  expect(container.querySelector("video")!.style.transform).toContain("rotate(90deg)");
  expect(serializeCameras(saved).wrist).toMatchObject({ rotation: 90, width: 640, height: 480 });
});
test("the recording configuration displays rotated output without an editing control", () => {
  act(() => root.render(<CameraConfiguration cameras={[{ ...camera, rotation: 90 }]} readOnly onCamerasChange={vi.fn()} />));
  expect(container.textContent).toContain("480×640 · 90° clockwise");
  expect(container.querySelector("select")).toBeNull();
  expect(container.querySelector("video")!.parentElement!.style.aspectRatio).toBe("480 / 640");
});

test("an invalid stored rotation stays visible until the user selects a valid value", () => {
  const change = vi.fn();
  const invalid = { ...camera, rotation: 45 as never };
  act(() => root.render(<CameraConfiguration cameras={[invalid]} onCamerasChange={change} />));
  expect(container.querySelector("[role=alert]")!.textContent).toContain("Select a valid rotation");
  expect(container.querySelector("video")).toBeNull();
  const select = container.querySelector("select")!;
  expect(select.value).toBe("");
  act(() => { select.value = "180"; select.dispatchEvent(new Event("change", { bubbles: true })); });
  expect(change).toHaveBeenCalledWith([{ ...camera, rotation: 180 }]);
});
test("reference controls appear only for a named robot in read-only mode", () => {
  act(() => root.render(<CameraConfiguration cameras={[camera]} onCamerasChange={vi.fn()} readOnly />));
  expect(container.textContent).not.toContain("Set reference");
  act(() => root.render(<CameraConfiguration cameras={[camera]} onCamerasChange={vi.fn()} readOnly robotName="arm" />));
  expect(container.textContent).toContain("Set reference");
  expect(container.querySelector('input[type="range"]')).toBeNull();
  reference.url = "blob:ref";
  act(() => root.render(<CameraConfiguration cameras={[camera]} onCamerasChange={vi.fn()} readOnly robotName="arm" />));
  expect(container.textContent).toContain("Replace reference");
  expect(container.querySelector("img")!.getAttribute("src")).toBe("blob:ref");
  const slider = container.querySelector<HTMLInputElement>('input[type="range"]')!;
  expect(slider).not.toBeNull();
  reference.url = null;
});
