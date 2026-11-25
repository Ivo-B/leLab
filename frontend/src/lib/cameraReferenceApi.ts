import { ApiError, Fetcher, apiRequest } from "./apiClient";

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

/** Snapshot the profile camera on the server, corrected like the recording, and keep it as the reference. */
export async function captureCameraReference(
  baseUrl: string,
  fetcher: Fetcher,
  robot: string,
  camera: string,
): Promise<Blob> {
  const r = await fetcher(`${cameraReferenceUrl(baseUrl, robot, camera)}/capture`, { method: "POST" });
  if (!r.ok) {
    const body = await r.json().catch(() => null);
    throw new ApiError(body?.message ?? `Capture reference failed: ${r.status}`, r.status, body);
  }
  return r.blob();
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
