import { describe, expect, it, vi } from "vitest";
import { getBoardCalibration, imagePoint, readBoardState, saveBoardCalibration } from "./boardCalibrationApi";

const BASE = "http://localhost:8000";

describe("boardCalibrationApi", () => {
  it("returns null on 404 and the body otherwise", async () => {
    const fetcher = vi.fn().mockResolvedValueOnce({ ok: false, status: 404 } as Response);
    expect(await getBoardCalibration(BASE, fetcher)).toBeNull();
    const body = { robot: "arm", image_size: [4, 4], board_corners: [], bin: [], park_white: [], park_black: [] };
    fetcher.mockResolvedValueOnce({ ok: true, status: 200, json: () => Promise.resolve(body) } as Response);
    expect(await getBoardCalibration(BASE, fetcher)).toEqual(body);
  });

  it("PUTs JSON and POSTs a PNG", async () => {
    const fetcher = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ status: "success" }),
    } as Response);
    const cal = { robot: "arm", image_size: [4, 4] as [number, number], board_corners: [], bin: [], park_white: [], park_black: [] };
    await saveBoardCalibration(BASE, fetcher, cal);
    expect(fetcher).toHaveBeenLastCalledWith(
      `${BASE}/chess/board-calibration`,
      expect.objectContaining({ method: "PUT", body: JSON.stringify(cal) }),
    );
    await readBoardState(BASE, fetcher, new Blob(["x"], { type: "image/png" }));
    const [url, init] = fetcher.mock.lastCall as [string, RequestInit];
    expect(url).toBe(`${BASE}/chess/board-state`);
    expect(init.method).toBe("POST");
    expect((init.headers as Record<string, string>)["Content-Type"]).toBe("image/png");
  });

  it("surfaces the server's message on an error", async () => {
    const fetcher = vi.fn().mockResolvedValue({
      ok: false,
      status: 409,
      json: () => Promise.resolve({ status: "error", message: "Calibrate the board on the Setup page first" }),
    } as Response);
    await expect(readBoardState(BASE, fetcher, new Blob(["x"]))).rejects.toThrow(
      "Calibrate the board on the Setup page first",
    );
  });

  it("maps client coordinates onto natural pixels", () => {
    const img = {
      naturalWidth: 640,
      naturalHeight: 480,
      getBoundingClientRect: () => ({ left: 10, top: 20, width: 320, height: 240 }),
    } as unknown as HTMLImageElement;
    expect(imagePoint(img, 170, 140)).toEqual([320, 240]);
  });
});
