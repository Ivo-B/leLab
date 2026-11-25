/** A camera from `/available-cameras`, in OpenCV's index order. */
export interface BackendCamera {
  index: number;
  name?: string;
  unique_id?: string;
  available: boolean;
}

/** A video input from the browser, whose label is the same name Chrome shows. */
export interface BrowserCamera {
  deviceId: string;
  label: string;
}

/** One browser device the user can assign when several cameras share a name. */
export interface CameraPicture {
  deviceId: string;
  label: string;
}

/** An OpenCV camera joined to a browser device only when that name identifies one camera. */
export interface MatchedCamera {
  index: number;
  name: string;
  uniqueId: string;
  /** The `vid:pid@location` id `lelab-camera list` prints, or null when the camera is not USB. */
  usbId: string | null;
  deviceId: string;
  available: boolean;
  /** Present when the name is shared, so the picture has to be chosen instead of guessed. */
  pictures?: CameraPicture[];
}

export interface SavedCameraIdentity {
  camera_index?: number;
  device_id?: string;
  unique_id?: string;
}

const norm = (value: string) => value.toLowerCase().replace(/\s+/g, " ").trim();

/** AVFoundation's USB id (`0x1300000abd8050`) in the form `lelab-camera list` prints (`0abd:8050@130000`). */
export function usbCameraId(uniqueId: string | undefined): string | null {
  const match = /^0x([0-9a-fA-F]{9,})$/.exec((uniqueId ?? "").trim());
  if (!match) return null;
  const digits = match[1];
  const location = digits.slice(0, -8).replace(/^0+/, "") || "0";
  return `${digits.slice(-8, -4).toLowerCase()}:${digits.slice(-4).toLowerCase()}@${location.toLowerCase()}`;
}

export function cameraChoiceLabel(camera: Pick<MatchedCamera, "name" | "usbId">): string {
  return camera.usbId ? `${camera.name} · ${camera.usbId}` : camera.name;
}

function labelMatches(browserLabel: string, backendName: string): boolean {
  const wanted = norm(backendName);
  const label = norm(browserLabel);
  if (!wanted || !label) return false;
  return label === wanted || label.startsWith(`${wanted} `) || label.startsWith(`${wanted}(`);
}

/**
 * Pair each OpenCV camera with the browser device of the same name.
 *
 * A name that belongs to one camera is paired. A name shared by several cameras is not: the browser
 * labels are identical (`USB Camera (0abd:8050)`), so assigning them in list order puts one camera's
 * picture on the other's port. Those cameras carry `pictures` and the user picks the stream.
 */
export function matchBrowserCameras(backend: BackendCamera[], browser: BrowserCamera[]): MatchedCamera[] {
  const groups = new Map<string, BackendCamera[]>();
  for (const camera of backend) {
    const key = norm(camera.name || "");
    groups.set(key, [...(groups.get(key) ?? []), camera]);
  }
  const used = new Set<string>();
  return backend.map((camera) => {
    const name = camera.name || `Camera ${camera.index}`;
    const siblings = groups.get(norm(camera.name || "")) ?? [];
    const candidates = browser.filter((device) => device.deviceId && labelMatches(device.label, name));
    const uniqueId = camera.unique_id ?? "";
    const matched: MatchedCamera = {
      index: camera.index,
      name,
      uniqueId,
      usbId: usbCameraId(uniqueId),
      deviceId: "",
      available: camera.available,
    };
    if (siblings.length > 1 || candidates.length !== 1) {
      if (candidates.length > 0 && (siblings.length > 1 || candidates.length > 1)) {
        matched.pictures = candidates.map((device) => ({ deviceId: device.deviceId, label: device.label }));
      }
      return matched;
    }
    const [only] = candidates;
    if (!used.has(only.deviceId)) {
      used.add(only.deviceId);
      matched.deviceId = only.deviceId;
    }
    return matched;
  });
}

/**
 * Keep a saved camera on the USB port it was configured for.
 *
 * The port is the AVFoundation id. A browser device id is the picture, and with two cameras of one name
 * that picture must not be allowed to rewrite the port.
 */
export function syncCameraIdentity<T extends SavedCameraIdentity>(camera: T, available: MatchedCamera[]): T {
  const byUnique = camera.unique_id
    ? available.find((item) => item.uniqueId === camera.unique_id)
    : undefined;
  if (byUnique) {
    return byUnique.index === camera.camera_index ? camera : { ...camera, camera_index: byUnique.index };
  }
  const byIndex = available.find((item) => item.index === camera.camera_index);
  if (byIndex?.uniqueId) return { ...camera, unique_id: byIndex.uniqueId };
  if (!camera.device_id) return camera;
  const byDevice = available.filter((item) => item.deviceId && item.deviceId === camera.device_id);
  if (byDevice.length !== 1) return camera;
  const match = byDevice[0];
  if (match.index === camera.camera_index && (match.uniqueId || "") === (camera.unique_id ?? "")) return camera;
  return { ...camera, camera_index: match.index, unique_id: match.uniqueId || camera.unique_id };
}

/** The next browser device in `pictures`, wrapping around, so a shared name can be checked by eye. */
export function nextPicture(pictures: CameraPicture[], current: string | undefined): string {
  if (pictures.length === 0) return "";
  const at = pictures.findIndex((picture) => picture.deviceId === current);
  return pictures[(at + 1) % pictures.length].deviceId;
}
