import { ApiError, Fetcher } from "./apiClient";
import { SavedByCamera, savedForCamera, savedFromBody } from "./cameraName";

/** Per-channel multipliers the server saves for one camera (`lelab-camera gain`). */
export interface ChannelGains {
  red: number;
  green: number;
  blue: number;
}

/** Saved gains by camera name (names that identify one camera) and by OpenCV index. */
export type CameraGains = SavedByCamera<ChannelGains>;

export async function getCameraGains(baseUrl: string, fetcher: Fetcher, signal?: AbortSignal): Promise<CameraGains> {
  const r = await fetcher(`${baseUrl}/camera-gains`, { signal });
  if (!r.ok) throw new ApiError(`Load colour gains failed: ${r.status}`, r.status, null);
  return savedFromBody<ChannelGains>((await r.json()) as Record<string, unknown>, "gains");
}

/** The gain saved for the camera with this label, or for its OpenCV `index` when that is known; see `savedForCamera`. */
export function gainForCamera(gains: CameraGains, label: string, index?: number): ChannelGains | null {
  return savedForCamera(gains, label, index);
}

/**
 * The SVG colour matrix for the same per-channel multiply, for a `feColorMatrix` filter in sRGB space. Rows
 * are R, G, B, A; the output is clipped to 0..1 by the filter, like `applyChannelGains`.
 */
export function gainMatrix(gains: ChannelGains): string {
  return [`${gains.red} 0 0 0 0`, `0 ${gains.green} 0 0 0`, `0 0 ${gains.blue} 0 0`, "0 0 0 1 0"].join(" ");
}

/**
 * Multiply RGBA pixels in place by the channel gains. Uint8ClampedArray rounds and clips to 0..255, the same
 * as the server's `apply_gains`, so a reference captured here matches the recorded frames.
 */
export function applyChannelGains(pixels: Uint8ClampedArray, gains: ChannelGains): void {
  for (let i = 0; i < pixels.length; i += 4) {
    pixels[i] = pixels[i] * gains.red;
    pixels[i + 1] = pixels[i + 1] * gains.green;
    pixels[i + 2] = pixels[i + 2] * gains.blue;
  }
}
