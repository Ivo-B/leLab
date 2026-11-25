import type { CameraRequest } from "./cameraConfig";

export interface CameraLens { fov: number; bend: number; zoom: number }
export interface ChannelGains { blue: number; green: number; red: number }

/**
 * A camera plugged in now, as the server lists it. `camera_id` is what a robot profile stores; `label` and
 * `key` are the ones `lelab-camera-calibrate` and `lelab-camera-lens` show and save the corrections under.
 */
export interface AttachedCamera {
  camera_id: string;
  index: number;
  name: string;
  /** `vid:pid@location`, the id `lelab-camera list` prints; null for a camera without USB controls. */
  usb_id: string | null;
  label: string;
  key: string;
  lens: CameraLens | null;
  gain: ChannelGains | null;
}

export type PreviewCamera = Pick<
  CameraRequest,
  "camera_id" | "width" | "height" | "fps" | "fourcc" | "color_correction" | "lens_correction"
>;

function query(camera: PreviewCamera): string {
  const params = new URLSearchParams({
    width: String(camera.width),
    height: String(camera.height),
    fps: String(camera.fps ?? 30),
  });
  if (camera.fourcc) params.set("fourcc", camera.fourcc);
  params.set("lens", String(camera.lens_correction !== false));
  params.set("gain", String(camera.color_correction !== false));
  return params.toString();
}

/** Live MJPEG of the camera, corrected like the recording and not rotated. */
export const previewUrl = (baseUrl: string, camera: PreviewCamera) =>
  `${baseUrl}/camera-preview/${encodeURIComponent(camera.camera_id)}?${query(camera)}`;

/** One corrected, unrotated PNG frame of the camera. */
export const snapshotUrl = (baseUrl: string, camera: PreviewCamera) =>
  `${baseUrl}/camera-snapshot/${encodeURIComponent(camera.camera_id)}?${query(camera)}`;

export const describeLens = (lens: CameraLens | null | undefined) =>
  lens ? `${Math.round(lens.fov)}° field of view · zoom ${lens.zoom.toFixed(2)}` : null;

export const describeGain = (gain: ChannelGains | null | undefined) =>
  gain ? `blue ×${gain.blue.toFixed(2)} · red ×${gain.red.toFixed(2)}` : null;
