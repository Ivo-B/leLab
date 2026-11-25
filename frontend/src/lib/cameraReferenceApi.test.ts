import { expect, test, vi } from "vitest";

import { captureCameraReference, getCameraReference } from "./cameraReferenceApi";

const BASE = "http://localhost:8000";

test("a missing reference reads as null rather than an error", async () => {
  const fetcher = vi.fn().mockResolvedValue(new Response("{}", { status: 404 }));
  await expect(getCameraReference(BASE, fetcher, "my arm", "front")).resolves.toBeNull();
  expect(fetcher.mock.calls[0][0]).toBe(`${BASE}/robots/my%20arm/camera-reference/front`);
});

test("capturing a reference asks the server to snapshot the profile camera", async () => {
  const fetcher = vi.fn().mockResolvedValue(new Response(new Blob(["png"], { type: "image/png" }), { status: 200 }));
  expect(await captureCameraReference(BASE, fetcher, "my arm", "top")).toBeInstanceOf(Blob);
  expect(fetcher).toHaveBeenCalledWith(`${BASE}/robots/my%20arm/camera-reference/top/capture`, { method: "POST" });
});

test("a refused capture shows the server's reason", async () => {
  const body = JSON.stringify({ status: "error", message: "Pick the camera for 'top' on the Calibration page first." });
  const fetcher = vi.fn().mockResolvedValue(new Response(body, { status: 400 }));
  await expect(captureCameraReference(BASE, fetcher, "arm", "top")).rejects.toThrow("Pick the camera for 'top'");
});
