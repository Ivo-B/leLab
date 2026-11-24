import { expect, test, vi } from "vitest";

import { captureCorrectedFrame, getCameraReference, saveCameraReference } from "./cameraReferenceApi";

const BASE = "http://localhost:8000";

test("a missing reference reads as null rather than an error", async () => {
  const fetcher = vi.fn().mockResolvedValue(new Response("{}", { status: 404 }));
  await expect(getCameraReference(BASE, fetcher, "my arm", "front")).resolves.toBeNull();
  expect(fetcher.mock.calls[0][0]).toBe(`${BASE}/robots/my%20arm/camera-reference/front`);
});

test("saving puts the PNG bytes and surfaces a rejection", async () => {
  const png = new Blob(["png"], { type: "image/png" });
  const fetcher = vi.fn().mockResolvedValueOnce(new Response("{}", { status: 200 }));
  await saveCameraReference(BASE, fetcher, "arm", "front", png);
  expect(fetcher).toHaveBeenCalledWith(`${BASE}/robots/arm/camera-reference/front`, {
    method: "PUT",
    headers: { "Content-Type": "image/png" },
    body: png,
  });

  fetcher.mockResolvedValueOnce(new Response("{}", { status: 400 }));
  await expect(saveCameraReference(BASE, fetcher, "arm", "front", png)).rejects.toThrow("Save reference failed: 400");
});

test("a corrected frame loads only the corrections that are switched on", async () => {
  const video = document.createElement("video");
  const fetcher = vi.fn((url: string) =>
    Promise.resolve(
      new Response(JSON.stringify(url.endsWith("/camera-gains") ? { gains: {} } : { lenses: {} }), { status: 200 }),
    ),
  );
  await captureCorrectedFrame(video, BASE, fetcher);
  expect(fetcher.mock.calls.map((call) => call[0]).sort()).toEqual([`${BASE}/camera-gains`, `${BASE}/camera-lenses`]);

  fetcher.mockClear();
  await captureCorrectedFrame(video, BASE, fetcher, { gain: false });
  expect(fetcher.mock.calls.map((call) => call[0])).toEqual([`${BASE}/camera-lenses`]);

  fetcher.mockClear();
  await captureCorrectedFrame(video, BASE, fetcher, { gain: false, lens: false });
  expect(fetcher).not.toHaveBeenCalled();
});
