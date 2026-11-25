// @vitest-environment jsdom
import React, { act } from "react";
import { createRoot, Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import CameraConfiguration from "./CameraConfiguration";
import { CameraConfig, normalizeCameraConfig, serializeCameras } from "@/lib/cameraConfig";
const attached = vi.hoisted(() => ({ value: [] as unknown[] }));
vi.mock("@/hooks/useAvailableCameras", () => ({ useAvailableCameras: () => ({ cameras: attached.value, isLoading: false, refresh: vi.fn() }) }));
vi.mock("@/contexts/ApiContext", () => ({ useOptionalApi: () => ({ baseUrl: "http://h" }) }));
vi.mock("@/hooks/use-toast", () => ({ useToast: () => ({ toast: vi.fn() }) }));
const reference = { url: null as string | null, error: null, save: vi.fn(), clear: vi.fn() };
vi.mock("@/hooks/useCameraReference", () => ({ useCameraReference: () => reference }));
let root: Root;
let container: HTMLDivElement;
const camera: CameraConfig = { id: "wrist", name: "wrist", camera_id: "0x21100000c456367", type: "opencv", width: 640, height: 480 };
const WRIST = {
  camera_id: "0x21100000c456367", index: 3, name: "Innomaker", usb_id: "0c45:6367@2110000",
  label: "Innomaker (USB 0c45:6367@2110000, index 3)", key: "Innomaker [0x21100000c456367]", lens: null, gain: null,
};
const TOP = {
  camera_id: "0x1300000abd8050", index: 1, name: "USB Camera", usb_id: "0abd:8050@130000",
  label: "USB Camera (USB 0abd:8050@130000, index 1)", key: "USB Camera [0x1300000abd8050]", lens: null, gain: null,
};
const rotationSelect = () => container.querySelector<HTMLSelectElement>('select:not([aria-label="Camera"])')!;
beforeEach(() => { Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true }); container = document.createElement("div"); root = createRoot(container); attached.value = [WRIST]; });
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
test("the lens and gain saved for the camera's key are named in the status and an off switch says off", () => {
  attached.value = [{ ...WRIST, lens: { fov: 135, bend: 0, zoom: 1.3 }, gain: { blue: 1.5, green: 1, red: 1.2 } }];
  act(() => root.render(<CameraConfiguration cameras={[camera]} onCamerasChange={vi.fn()} />));
  expect(container.textContent).toContain("On · 135° field of view · zoom 1.30");
  expect(container.textContent).toContain("On · blue ×1.50 · red ×1.20");
  act(() => root.render(<CameraConfiguration cameras={[{ ...camera, lens_correction: false }]} onCamerasChange={vi.fn()} />));
  expect(container.textContent).toContain("Lens correction");
  expect(container.textContent).toMatch(/Lens correction\s*Off/);
});
test("rotation updates the profile while capture dimensions stay native", () => {
  const change = vi.fn();
  act(() => root.render(<CameraConfiguration cameras={[camera]} onCamerasChange={change} />));
  const select = rotationSelect();
  expect(select.value).toBe("0");
  act(() => { select.value = "90"; select.dispatchEvent(new Event("change", { bubbles: true })); });
  expect(change).toHaveBeenCalledWith([{ ...camera, rotation: 90 }]);
  const saved = JSON.parse(JSON.stringify(change.mock.calls[0][0])).map(normalizeCameraConfig);
  act(() => root.render(<CameraConfiguration cameras={saved} onCamerasChange={change} />));
  expect(rotationSelect().value).toBe("90");
  expect(container.querySelector("img")!.style.transform).toContain("rotate(90deg)");
  expect(serializeCameras(saved).wrist).toMatchObject({ rotation: 90, width: 640, height: 480 });
});
test("the recording configuration displays rotated output without an editing control", () => {
  act(() => root.render(<CameraConfiguration cameras={[{ ...camera, rotation: 90 }]} readOnly onCamerasChange={vi.fn()} />));
  expect(container.textContent).toContain("480×640 · 90° clockwise");
  expect(container.querySelector("select")).toBeNull();
  expect(container.querySelector("img")!.parentElement!.style.aspectRatio).toBe("480 / 640");
  expect(container.textContent).toContain("Innomaker (USB 0c45:6367@2110000, index 3)");
});

test("an invalid stored rotation stays visible until the user selects a valid value", () => {
  const change = vi.fn();
  const invalid = { ...camera, rotation: 45 as never };
  act(() => root.render(<CameraConfiguration cameras={[invalid]} onCamerasChange={change} />));
  expect(container.querySelector("[role=alert]")!.textContent).toContain("Select a valid rotation");
  expect(container.querySelector("img")).toBeNull();
  const select = rotationSelect();
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
  expect(container.querySelectorAll("img")[1].getAttribute("src")).toBe("blob:ref");
  const slider = container.querySelector<HTMLInputElement>('input[type="range"]')!;
  expect(slider).not.toBeNull();
  reference.url = null;
});
test("each card shows the server picture of its camera, labelled like the colour and lens tools", () => {
  attached.value = [WRIST, TOP];
  act(() => root.render(<CameraConfiguration cameras={[camera]} onCamerasChange={vi.fn()} />));
  expect(container.querySelector("img")!.src).toContain("http://h/camera-preview/0x21100000c456367?");
  const picker = container.querySelector<HTMLSelectElement>('select[aria-label="Camera"]')!;
  expect(picker.value).toBe("0x21100000c456367");
  expect(Array.from(picker.options).map((o) => o.text)).toEqual([WRIST.label, TOP.label]);
  expect(container.textContent).not.toContain("Switch picture");
});
test("a card's camera can be changed to another attached camera", () => {
  attached.value = [WRIST, TOP];
  const change = vi.fn();
  act(() => root.render(<CameraConfiguration cameras={[camera]} onCamerasChange={change} />));
  const picker = container.querySelector<HTMLSelectElement>('select[aria-label="Camera"]')!;
  act(() => { picker.value = TOP.camera_id; picker.dispatchEvent(new Event("change", { bubbles: true })); });
  expect(change).toHaveBeenCalledWith([{ ...camera, camera_id: TOP.camera_id }]);
});
test("a camera another card uses cannot be picked twice", () => {
  attached.value = [WRIST, TOP];
  const top: CameraConfig = { ...camera, id: "top", name: "top", camera_id: TOP.camera_id };
  act(() => root.render(<CameraConfiguration cameras={[camera, top]} onCamerasChange={vi.fn()} />));
  const picker = container.querySelector<HTMLSelectElement>('select[aria-label="Camera"]')!;
  expect(Array.from(picker.options).find((o) => o.value === TOP.camera_id)!.disabled).toBe(true);
});
test("a camera saved without an id asks to be picked, and an unplugged one says so", () => {
  act(() => root.render(<CameraConfiguration cameras={[{ ...camera, camera_id: "" }]} onCamerasChange={vi.fn()} />));
  expect(container.textContent).toContain("No camera selected");
  expect(container.querySelector<HTMLSelectElement>('select[aria-label="Camera"]')!.value).toBe("__none__");
  attached.value = [TOP];
  act(() => root.render(<CameraConfiguration cameras={[camera]} onCamerasChange={vi.fn()} />));
  expect(container.textContent).toContain("Camera not attached");
});
