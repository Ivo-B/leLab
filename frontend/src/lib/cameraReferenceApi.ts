import { ApiError, Fetcher, apiRequest } from "./apiClient";
import { ChannelGains, applyChannelGains } from "./cameraGain";
import { CameraLens, undistortPixels } from "./cameraLens";

export function cameraReferenceUrl(baseUrl: string, robot: string, camera: string): string {
  return `${baseUrl}/robots/${encodeURIComponent(robot)}/camera-reference/${encodeURIComponent(camera)}`;
}

/** The saved reference frame as a PNG blob, or null when none has been saved yet. */
export async function getCameraReference(
  baseUrl: string,
  fetcher: Fetcher,
  robot: string,
  camera: string,
  signal?: AbortSignal,
): Promise<Blob | null> {
  const r = await fetcher(cameraReferenceUrl(baseUrl, robot, camera), { signal });
  if (r.status === 404) return null;
  if (!r.ok) throw new ApiError(`Load reference failed: ${r.status}`, r.status, null);
  return r.blob();
}

export async function saveCameraReference(
  baseUrl: string,
  fetcher: Fetcher,
  robot: string,
  camera: string,
  png: Blob,
): Promise<void> {
  const r = await fetcher(cameraReferenceUrl(baseUrl, robot, camera), {
    method: "PUT",
    headers: { "Content-Type": "image/png" },
    body: png,
  });
  if (!r.ok) throw new ApiError(`Save reference failed: ${r.status}`, r.status, null);
}

export async function deleteCameraReference(
  baseUrl: string,
  fetcher: Fetcher,
  robot: string,
  camera: string,
): Promise<void> {
  await apiRequest(baseUrl, fetcher, `/robots/${encodeURIComponent(robot)}/camera-reference/${encodeURIComponent(camera)}`, {
    method: "DELETE",
    action: "Delete reference",
  });
}

/** What the recorder does to a frame before it is stored; the reference frame gets the same, in the same order. */
export interface FrameCorrections {
  gains?: ChannelGains | null;
  lens?: CameraLens | null;
}

/**
 * Snapshot of the frame the preview is showing, unrotated like the stream itself. With `lens` and `gains`, the
 * same fisheye and colour corrections the recorder applies are applied first (lens, then colour), so the
 * reference matches the recorded frames.
 */
export function captureVideoFrame(video: HTMLVideoElement, { gains, lens }: FrameCorrections = {}): Promise<Blob | null> {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx || !canvas.width || !canvas.height) return Promise.resolve(null);
  ctx.drawImage(video, 0, 0);
  if (gains || lens) {
    const frame = ctx.getImageData(0, 0, canvas.width, canvas.height);
    if (lens) frame.data.set(undistortPixels(frame.data, canvas.width, canvas.height, lens));
    if (gains) applyChannelGains(frame.data, gains);
    ctx.putImageData(frame, 0, 0);
  }
  return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
}
