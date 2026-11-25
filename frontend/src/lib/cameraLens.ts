import { ApiError, Fetcher } from "./apiClient";
import { SavedByCamera, savedForCamera, savedFromBody } from "./cameraName";

/**
 * The fisheye lens model the server saves for one camera (`lelab-camera lens`): the diagonal field of view in
 * degrees, a fine curvature term, and a zoom. The formula is the one in `lelab/lens.py` and in the WebGL
 * shader (`lensRenderer.ts`); the numeric fixtures in `cameraLens.test.ts` and `tests/test_lens.py` pin them.
 */
export interface CameraLens {
  fov: number;
  bend: number;
  zoom: number;
}

/** Saved lenses by camera name (names that identify one camera) and by OpenCV index. */
export type CameraLenses = SavedByCamera<CameraLens>;

export async function getCameraLenses(baseUrl: string, fetcher: Fetcher, signal?: AbortSignal): Promise<CameraLenses> {
  const r = await fetcher(`${baseUrl}/camera-lenses`, { signal });
  if (!r.ok) throw new ApiError(`Load lens corrections failed: ${r.status}`, r.status, null);
  return savedFromBody<CameraLens>((await r.json()) as Record<string, unknown>, "lenses");
}

/** The lens saved for the camera with this label, or for its OpenCV `index` when that is known; see `savedForCamera`. */
export function lensForCamera(lenses: CameraLenses, label: string, index?: number): CameraLens | null {
  return savedForCamera(lenses, label, index);
}

/** Where output pixel `(i, j)` of the straightened picture samples the camera picture, in continuous pixel coordinates. */
export function lensSource(lens: CameraLens, width: number, height: number, i: number, j: number): [number, number] {
  const halfDiagonal = Math.hypot(width, height) / 2;
  const thetaMax = (lens.fov / 2) * (Math.PI / 180);
  const thetaDmax = thetaMax * (1 + lens.bend * thetaMax * thetaMax);
  const ox = (i + 0.5 - width / 2) / halfDiagonal;
  const oy = (j + 0.5 - height / 2) / halfDiagonal;
  const rho = Math.hypot(ox, oy);
  const theta = Math.atan((rho * thetaDmax) / lens.zoom);
  const rSource = (theta * (1 + lens.bend * theta * theta)) / thetaDmax;
  // rSource / rho tends to 1 / zoom at the centre, where both are zero.
  const scale = rho > 1e-12 ? rSource / rho : 1 / lens.zoom;
  return [width / 2 + ox * scale * halfDiagonal, height / 2 + oy * scale * halfDiagonal];
}

/**
 * Straighten RGBA pixels with bilinear sampling, like `cv2.remap` with a zero border in `lens.apply_lens`:
 * what the camera did not see is black, and the result is opaque. Used for the reference frame, so it matches
 * the frames the recorder straightens on the server.
 */
export function undistortPixels(src: Uint8ClampedArray, width: number, height: number, lens: CameraLens): Uint8ClampedArray {
  const out = new Uint8ClampedArray(src.length);
  for (let j = 0; j < height; j++) {
    for (let i = 0; i < width; i++) {
      const [sx, sy] = lensSource(lens, width, height, i, j);
      const fx = sx - 0.5;
      const fy = sy - 0.5;
      const x0 = Math.floor(fx);
      const y0 = Math.floor(fy);
      const wx = fx - x0;
      const wy = fy - y0;
      let red = 0;
      let green = 0;
      let blue = 0;
      for (let dy = 0; dy < 2; dy++) {
        for (let dx = 0; dx < 2; dx++) {
          const x = x0 + dx;
          const y = y0 + dy;
          if (x < 0 || y < 0 || x >= width || y >= height) continue; // a zero tap: black
          const weight = (dx ? wx : 1 - wx) * (dy ? wy : 1 - wy);
          const from = (y * width + x) * 4;
          red += src[from] * weight;
          green += src[from + 1] * weight;
          blue += src[from + 2] * weight;
        }
      }
      const target = (j * width + i) * 4;
      out[target] = red; // a Uint8ClampedArray rounds once, here, like cv2.remap
      out[target + 1] = green;
      out[target + 2] = blue;
      out[target + 3] = 255;
    }
  }
  return out;
}
