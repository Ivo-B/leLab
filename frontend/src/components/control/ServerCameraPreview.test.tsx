// @vitest-environment jsdom
import React, { act } from "react";
import { createRoot, Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import ServerCameraPreview from "./ServerCameraPreview";
import CameraRotationSelect from "./CameraRotationSelect";
vi.mock("@/contexts/ApiContext", () => ({ useOptionalApi: () => ({ baseUrl: "http://h" }) }));
let root: Root;
let container: HTMLDivElement;
beforeEach(() => { Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true }); container = document.createElement("div"); root = createRoot(container); });
afterEach(() => { act(() => root.unmount()); vi.useRealTimers(); });
const cam = { camera_id: "0x1300000abd8050", width: 640, height: 480, rotation: 90 as const };

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
test("the server stream is shown rotated in a box of the rotated shape", () => {
  act(() => root.render(<ServerCameraPreview camera={cam} />));
  const img = container.querySelector("img")!;
  expect(img.src).toBe("http://h/camera-preview/0x1300000abd8050?width=640&height=480&fps=30&lens=true&gain=true");
  expect(img.style.transform).toBe("translate(-50%, -50%) rotate(90deg)");
  expect(parseFloat(img.style.width)).toBeCloseTo(640 / 480 * 100);
  expect((container.firstChild as HTMLElement).style.aspectRatio).toBe("480 / 640");
});
test("the reference ghost sits over the picture with its rotation and opacity", () => {
  act(() => root.render(<ServerCameraPreview camera={cam} ghostSrc="blob:ref" ghostOpacity={0.3} />));
  const [live, ghost] = Array.from(container.querySelectorAll("img"));
  expect(ghost.getAttribute("src")).toBe("blob:ref");
  expect(ghost.style.opacity).toBe("0.3");
  expect(ghost.style.transform).toBe(live.style.transform);
});
test("a paused preview drops the stream", () => {
  act(() => root.render(<ServerCameraPreview camera={cam} paused />));
  expect(container.querySelector("img")).toBeNull();
  expect(container.textContent).toContain("Preview paused");
});
test("a camera without an id asks to pick one", () => {
  act(() => root.render(<ServerCameraPreview camera={{ ...cam, camera_id: "" }} />));
  expect(container.textContent).toContain("No camera selected");
});
test("a stream that fails is asked for again", () => {
  vi.useFakeTimers();
  act(() => root.render(<ServerCameraPreview camera={cam} />));
  act(() => { container.querySelector("img")!.dispatchEvent(new Event("error")); });
  expect(container.querySelector("img")).toBeNull();
  act(() => { vi.advanceTimersByTime(3000); });
  expect(container.querySelector("img")!.src).toContain("&retry=1");
});
