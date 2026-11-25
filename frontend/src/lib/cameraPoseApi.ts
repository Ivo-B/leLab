import { ApiError, Fetcher } from "./apiClient";
import type { Point } from "./boardCalibrationApi";

export type CameraName = "top" | "side";

export interface PoseFit {
  camera: CameraName;
  pos: [number, number, number];
  xyaxes: number[];
  fovy: number;
  rms_px: number;
  fov_fitted: boolean;
  warnings: string[];
  applied: boolean;
}

export interface PoseRequest {
  camera: CameraName;
  image_size: [number, number];
  board_corners: Point[];
  fovy: number | null;
  apply: boolean;
}

/** Fit a camera's pose from four board corners clicked on its reference; `apply` also writes the sim cell. */
export async function fitCameraPose(baseUrl: string, fetcher: Fetcher, request: PoseRequest): Promise<PoseFit> {
  const r = await fetcher(`${baseUrl}/chess/camera-pose`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!r.ok) {
    let message = `Fit camera pose failed: ${r.status}`;
    try {
      const err = (await r.json()) as { message?: string };
      if (err?.message) message = err.message;
    } catch {
      /* no JSON body */
    }
    throw new ApiError(message, r.status, null);
  }
  return (await r.json()) as PoseFit;
}
