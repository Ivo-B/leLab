import { describe, expect, it, vi } from "vitest";
import { fitCameraPose } from "./cameraPoseApi";

const BASE = "http://localhost:8000";
const body = {
  camera: "side" as const,
  image_size: [640, 480] as [number, number],
  board_corners: [[1, 2], [3, 4], [5, 6], [7, 8]] as Array<[number, number]>,
  fovy: null,
  apply: false,
};

describe("fitCameraPose", () => {
  it("POSTs JSON and returns the fit", async () => {
    const fit = {
      camera: "side",
      pos: [0.1, -0.3, 0.3],
      xyaxes: [1, 0, 0, 0, 1, 0],
      fovy: 45,
      rms_px: 0.4,
      fov_fitted: true,
      warnings: [],
      applied: false,
    };
    const fetcher = vi.fn().mockResolvedValue({ ok: true, status: 200, json: () => Promise.resolve(fit) } as Response);
    expect(await fitCameraPose(BASE, fetcher, body)).toEqual(fit);
    expect(fetcher).toHaveBeenCalledWith(
      `${BASE}/chess/camera-pose`,
      expect.objectContaining({ method: "POST", body: JSON.stringify(body) }),
    );
  });

  it("throws the server's reason", async () => {
    const fetcher = vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: () => Promise.resolve({ message: "The reference is 1920x1080" }),
    } as Response);
    await expect(fitCameraPose(BASE, fetcher, body)).rejects.toThrow("The reference is 1920x1080");
  });
});
