import type { CSSProperties } from "react";

export const CAMERA_ROTATIONS = [0, 90, 180, 270] as const;
export type CameraRotation = (typeof CAMERA_ROTATIONS)[number];

/** Camera dimensions describe capture before rotation. */
export interface CameraRequest {
  type: string;
  /** The server's id for the physical camera: AVFoundation's uniqueID on macOS, `index:N` elsewhere. */
  camera_id: string;
  width: number;
  height: number;
  fps?: number;
  fourcc?: string;
  backend?: string;
  rotation?: CameraRotation;
  /** When false, the saved per-channel colour gain is not applied. Omitted means on. */
  color_correction?: boolean;
  /** When false, the saved fisheye lens correction is not applied. Omitted means on. */
  lens_correction?: boolean;
}

export interface CameraConfig extends CameraRequest {
  id: string;
  name: string;
}

/** Only an omitted rotation defaults to zero. */
export function cameraRotation(value: unknown = 0): CameraRotation {
  if (!CAMERA_ROTATIONS.some((rotation) => rotation === value)) {
    throw new Error(`Camera rotation must be 0, 90, 180, or 270 degrees clockwise; received ${String(value)}.`);
  }
  return value as CameraRotation;
}

export function cameraRotationError(camera: Pick<CameraRequest, "rotation">): string | null {
  try {
    cameraRotation(camera.rotation);
    return null;
  } catch (error) {
    return error instanceof Error ? error.message : String(error);
  }
}

export function cameraConfigurationError(cameras: CameraConfig[]): string | null {
  for (const camera of cameras) {
    if (!camera.camera_id) return `${camera.name}: pick the camera again`;
    const error = cameraRotationError(camera);
    if (error) return `${camera.name}: ${error}`;
  }
  return null;
}

/** Preserve invalid stored values so the user can correct them explicitly. */
export function loadCameraConfigs(cameras: CameraConfig[]): CameraConfig[] {
  return cameras.map((camera) => cameraRotationError(camera) ? camera : normalizeCameraConfig(camera));
}

export function normalizeCameraConfig(camera: CameraConfig): CameraConfig {
  return { ...camera, rotation: cameraRotation(camera.rotation) };
}

/** Remove profile-only fields before sending a camera to a session. */
export function serializeCamera(camera: CameraRequest): CameraRequest {
  return {
    type: camera.type,
    camera_id: camera.camera_id,
    width: camera.width,
    height: camera.height,
    fps: camera.fps,
    rotation: cameraRotation(camera.rotation),
    ...(camera.color_correction === false ? { color_correction: false } : {}),
    ...(camera.lens_correction === false ? { lens_correction: false } : {}),
    ...(camera.fourcc ? { fourcc: camera.fourcc } : {}),
    ...(camera.backend ? { backend: camera.backend } : {}),
  };
}

export function serializeCameras(cameras: CameraConfig[]): Record<string, CameraRequest> {
  return Object.fromEntries(cameras.map((camera) => [camera.name, serializeCamera(camera)]));
}

/** Quarter turns swap capture and output dimensions in both directions. */
export function rotatedDimensions(width: number, height: number, rotation: CameraRotation) {
  const angle = cameraRotation(rotation);
  return angle === 90 || angle === 270 ? { width: height, height: width } : { width, height };
}

/** Center the complete native rectangle inside the rotated output rectangle. */
export function cameraPreviewGeometry(width: number, height: number, rotation: CameraRotation) {
  const output = rotatedDimensions(width, height, rotation);
  const quarterTurn = rotation === 90 || rotation === 270;
  const videoStyle: CSSProperties = {
    position: "absolute",
    left: "50%",
    top: "50%",
    width: quarterTurn ? `${(width / height) * 100}%` : "100%",
    height: quarterTurn ? `${(height / width) * 100}%` : "100%",
    // Crop, don't letterbox: a camera that delivers another aspect ratio still shows the recorded framing.
    objectFit: "cover",
    maxWidth: "none",
    transform: `translate(-50%, -50%) rotate(${rotation}deg)`,
  };
  return { aspectRatio: `${output.width} / ${output.height}`, videoStyle };
}

/** Rotation follows the physical camera when a policy name is reassigned. */
export function rotationForCamera(
  cameraId: string,
  configured: CameraConfig[],
  overrides: Record<string, CameraRotation>,
): CameraRotation {
  return cameraRotation(overrides[cameraId] ?? configured.find((item) => item.camera_id === cameraId)?.rotation);
}

/** Size each corrected camera frame within a common grid cell. */
export function recordingCameraWindows(
  names: string[],
  cameras: Record<string, CameraRequest>,
  area: { w: number; h: number },
): Record<string, { width: number; height: number }> {
  if (!names.length || area.w <= 0 || area.h <= 0) return {};
  let best: Record<string, { width: number; height: number }> = {};
  let bestArea = -1;
  for (let columns = 1; columns <= names.length; columns++) {
    const rows = Math.ceil(names.length / columns);
    const cellWidth = (area.w - 12 * (columns - 1)) / columns;
    const cellHeight = (area.h - 12 * (rows - 1)) / rows;
    if (cellWidth <= 0 || cellHeight <= 0) continue;
    let total = 0;
    const windows = Object.fromEntries(names.map((name) => {
      const camera = cameras[name];
      const output = camera
        ? rotatedDimensions(camera.width, camera.height, cameraRotation(camera.rotation))
        : { width: 640, height: 480 };
      const aspect = output.width / output.height;
      const width = Math.min(cellWidth, cellHeight * aspect);
      const height = width / aspect;
      total += width * height;
      return [name, { width: Math.floor(width), height: Math.floor(height) }];
    }));
    if (total > bestArea) { best = windows; bestArea = total; }
  }
  return best;
}
