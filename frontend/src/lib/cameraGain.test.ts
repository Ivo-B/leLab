import { describe, expect, test, vi } from "vitest";

import { applyChannelGains, gainForCamera, gainMatrix, getCameraGains } from "./cameraGain";

const BASE = "http://localhost:8000";
const WHITE_PAPER = { red: 1.4464, green: 1, blue: 1.7868 };

describe("applyChannelGains", () => {
  test("multiplies each channel and leaves alpha alone", () => {
    const pixels = new Uint8ClampedArray([100, 100, 100, 255]);
    applyChannelGains(pixels, WHITE_PAPER);
    expect(Array.from(pixels)).toEqual([145, 100, 179, 255]);
  });

  test("clips at 255 rather than wrapping", () => {
    const pixels = new Uint8ClampedArray([200, 200, 200, 255]);
    applyChannelGains(pixels, WHITE_PAPER);
    expect(pixels[0]).toBe(255);
    expect(pixels[2]).toBe(255);
    expect(pixels[1]).toBe(200);
  });

  test("the measured white paper comes out neutral", () => {
    // Median paper pixel before the gain, in RGB: red 168, green 243, blue 136.
    const pixels = new Uint8ClampedArray([168, 243, 136, 255]);
    applyChannelGains(pixels, WHITE_PAPER);
    expect(pixels[0]).toBe(243);
    expect(pixels[1]).toBe(243);
    expect(pixels[2]).toBe(243);
  });
});

describe("gainMatrix", () => {
  test("puts each gain on its channel's diagonal and passes alpha through", () => {
    expect(gainMatrix(WHITE_PAPER)).toBe("1.4464 0 0 0 0 0 1 0 0 0 0 0 1.7868 0 0 0 0 0 1 0");
  });
});

describe("gainForCamera", () => {
  const gains = { "Innomaker-U20CAM-720P  ": { red: 1, green: 1, blue: 1 }, "USB Camera": WHITE_PAPER };

  test("matches the label the browser shows, ignoring stray spaces and Chrome's vendor suffix", () => {
    expect(gainForCamera(gains, "USB Camera")).toEqual(WHITE_PAPER);
    expect(gainForCamera(gains, "Innomaker-U20CAM-720P")).toEqual({ red: 1, green: 1, blue: 1 });
    expect(gainForCamera(gains, "  USB Camera ")).toEqual(WHITE_PAPER);
    expect(gainForCamera(gains, "USB Camera (0abd:8050)")).toEqual(WHITE_PAPER);
  });

  test("a longer saved name wins when two names are prefixes of the label", () => {
    const both = { USB: { red: 2, green: 1, blue: 2 }, ...gains };
    expect(gainForCamera(both, "USB Camera (0abd:8050)")).toEqual(WHITE_PAPER);
  });

  test("returns null for a camera with no saved gain, or an empty label", () => {
    expect(gainForCamera(gains, "MacBook Pro Camera")).toBeNull();
    expect(gainForCamera(gains, "")).toBeNull();
  });
});

describe("getCameraGains", () => {
  test("reads the saved gains from the server", async () => {
    const fetcher = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ status: "success", gains: { "USB Camera": WHITE_PAPER } }), { status: 200 }),
    );
    await expect(getCameraGains(BASE, fetcher)).resolves.toEqual({ "USB Camera": WHITE_PAPER });
    expect(fetcher.mock.calls[0][0]).toBe(`${BASE}/camera-gains`);
  });

  test("surfaces a server error", async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response("{}", { status: 500 }));
    await expect(getCameraGains(BASE, fetcher)).rejects.toThrow("Load colour gains failed: 500");
  });
});
