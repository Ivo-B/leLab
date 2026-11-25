import { describe, expect, test, vi } from "vitest";

import { getCameraLenses, lensForCamera, lensSource, undistortPixels } from "./cameraLens";

const BASE = "http://localhost:8000";
const LENS = { fov: 120, bend: 0, zoom: 1 };

// The same lens, size and points, to the same precision, as FIXTURE_* in tests/test_lens.py.
const FIXTURE_LENS = { fov: 110, bend: 0.05, zoom: 0.8 };
const FIXTURE_POINTS: [[number, number], [number, number]][] = [
  [[0, 0], [22.51254984, 17.00080027]],
  [[100, 50], [91.32097101, 42.57550801]],
  [[639, 479], [617.48745016, 462.99919973]],
  [[320, 240], [320.62499912, 240.62499912]],
  [[10, 400], [16.2766783, 397.50433969]],
];

describe("lensSource", () => {
  test.each(FIXTURE_POINTS)("pixel %j samples where the server's model says", ([i, j], [x, y]) => {
    const [sx, sy] = lensSource(FIXTURE_LENS, 640, 480, i, j);
    expect(sx).toBeCloseTo(x, 6);
    expect(sy).toBeCloseTo(y, 6);
  });

  test("the centre keeps its scale and zoom magnifies it", () => {
    const [x1] = lensSource(LENS, 641, 481, 320, 240);
    const [x2] = lensSource(LENS, 641, 481, 321, 240);
    expect(x2 - x1).toBeCloseTo(1, 2);
    const [z1] = lensSource({ ...LENS, zoom: 2 }, 641, 481, 320, 240);
    const [z2] = lensSource({ ...LENS, zoom: 2 }, 641, 481, 321, 240);
    expect(z2 - z1).toBeCloseTo(0.5, 2);
  });

  test("the exact centre of the picture stays put", () => {
    expect(lensSource(LENS, 641, 481, 320, 240)).toEqual([320.5, 240.5]);
  });
});

describe("undistortPixels", () => {
  const width = 24;
  const height = 18;
  const gradient = () => {
    const pixels = new Uint8ClampedArray(width * height * 4);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        pixels.set([x * 10, y * 13, (x + y) * 5, 255], (y * width + x) * 4);
      }
    }
    return pixels;
  };
  const rgbAt = (pixels: Uint8ClampedArray, x: number, y: number) => Array.from(pixels.slice((y * width + x) * 4, (y * width + x) * 4 + 3));

  // Rendered by lelab.lens.apply_lens (cv2.remap) from the same gradient. cv2 interpolates in 1/32 steps, so allow 2 levels.
  test.each([
    [[0, 0], [6, 6, 5]],
    [[12, 9], [121, 119, 106]],
    [[3, 2], [26, 22, 21]],
    [[20, 15], [204, 199, 179]],
    [[23, 17], [224, 215, 195]],
    [[5, 16], [46, 215, 105]],
  ] as [[number, number], number[]][])("pixel %j matches the server's straightened frame", ([x, y], expected) => {
    const out = undistortPixels(gradient(), width, height, FIXTURE_LENS);
    rgbAt(out, x, y).forEach((value, channel) => expect(Math.abs(value - expected[channel])).toBeLessThanOrEqual(2));
  });

  test("what the camera did not see is black and the result is opaque", () => {
    const white = new Uint8ClampedArray(width * height * 4).fill(255);
    const out = undistortPixels(white, width, height, { fov: 120, bend: 0, zoom: 0.4 });
    expect(Array.from(out.slice(0, 4))).toEqual([0, 0, 0, 255]);
    expect(Array.from(out.slice((9 * width + 12) * 4, (9 * width + 12) * 4 + 4))).toEqual([255, 255, 255, 255]);
  });
});

describe("lensForCamera", () => {
  const lenses = { byName: { "USB Camera": LENS, "Innomaker-U20CAM-720P  ": FIXTURE_LENS }, byIndex: {} };

  test("matches the label the browser shows, ignoring stray spaces and Chrome's vendor suffix", () => {
    expect(lensForCamera(lenses, "USB Camera (0abd:8050)")).toEqual(LENS);
    expect(lensForCamera(lenses, "Innomaker-U20CAM-720P")).toEqual(FIXTURE_LENS);
    expect(lensForCamera(lenses, "FaceTime HD Camera")).toBeNull();
  });

  test("two cameras that share a name each get their own lens by OpenCV index", () => {
    const twins = { byName: {}, byIndex: { "1": LENS, "2": FIXTURE_LENS } };
    expect(lensForCamera(twins, "USB Camera", 1)).toEqual(LENS);
    expect(lensForCamera(twins, "USB Camera", 2)).toEqual(FIXTURE_LENS);
    expect(lensForCamera(twins, "USB Camera")).toBeNull();
  });
});

describe("getCameraLenses", () => {
  test("returns the saved lenses", async () => {
    const fetcher = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ status: "success", lenses: { "USB Camera": LENS }, by_index: { "2": LENS } }),
    });
    await expect(getCameraLenses(BASE, fetcher)).resolves.toEqual({
      byName: { "USB Camera": LENS },
      byIndex: { "2": LENS },
    });
    expect(fetcher).toHaveBeenCalledWith(`${BASE}/camera-lenses`, { signal: undefined });
  });

  test("a server error is reported", async () => {
    const fetcher = vi.fn().mockResolvedValue({ ok: false, status: 500 });
    await expect(getCameraLenses(BASE, fetcher)).rejects.toThrow("500");
  });
});
