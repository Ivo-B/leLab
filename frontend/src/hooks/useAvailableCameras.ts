import { useCallback, useEffect, useState } from "react";
import { useApi } from "@/contexts/ApiContext";
import { matchBrowserCameras, type MatchedCamera } from "@/lib/cameraMatch";

export type AvailableCamera = MatchedCamera;

interface UseAvailableCamerasOptions {
  /** When false, do nothing. Use to gate on modal open. */
  enabled?: boolean;
}

/**
 * Enumerates cv2 camera indices from `/available-cameras`. A name that belongs to one
 * camera is paired with its browser device. A shared name is not: the picture is chosen
 * on the card, and the USB id (`0abd:8050@130000`) tells the cameras apart. Refreshes on
 * USB hotplug.
 */
export function useAvailableCameras({
  enabled = true,
}: UseAvailableCamerasOptions = {}) {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [cameras, setCameras] = useState<AvailableCamera[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const refresh = useCallback(async (): Promise<AvailableCamera[]> => {
    setIsLoading(true);
    try {
      // Need a permission grant before enumerateDevices() returns labels.
      try {
        const probe = await navigator.mediaDevices.getUserMedia({ video: true });
        probe.getTracks().forEach((t) => t.stop());
      } catch {
        // ignore — we'll still try to enumerate, just without labels
      }

      const browserDevices = (await navigator.mediaDevices.enumerateDevices())
        .filter((d) => d.kind === "videoinput")
        .map((d) => ({ deviceId: d.deviceId, label: d.label }));

      const r = await fetchWithHeaders(`${baseUrl}/available-cameras`);
      if (!r.ok) {
        setCameras([]);
        return [];
      }
      const data = await r.json();
      const backendCams: {
        index: number;
        name?: string;
        unique_id?: string;
        available: boolean;
      }[] = data.cameras ?? [];
      // A shared name such as "USB Camera" is left for the user to assign: the browser
      // label does not carry the USB port, so list order would cross the two pictures.
      const merged = matchBrowserCameras(backendCams, browserDevices);
      setCameras(merged);
      return merged;
    } catch {
      setCameras([]);
      return [];
    } finally {
      setIsLoading(false);
    }
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    if (!enabled) return;
    refresh();
    const handler = () => refresh();
    navigator.mediaDevices.addEventListener("devicechange", handler);
    return () =>
      navigator.mediaDevices.removeEventListener("devicechange", handler);
  }, [enabled, refresh]);

  return { cameras, isLoading, refresh };
}
