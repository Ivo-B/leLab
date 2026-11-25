import { useCallback, useEffect, useState } from "react";
import { useApi } from "@/contexts/ApiContext";
import type { AttachedCamera } from "@/lib/cameraApi";

export type AvailableCamera = AttachedCamera;

interface UseAvailableCamerasOptions {
  /** When false, do nothing. Use to gate on modal open. */
  enabled?: boolean;
}

/**
 * The cameras the server sees, listed and labelled as `lelab-camera-calibrate` and `lelab-camera-lens` list
 * them, each with its id, USB id and saved lens and colour gain. Refreshes on USB hotplug.
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
      const r = await fetchWithHeaders(`${baseUrl}/available-cameras`);
      const data = r.ok ? await r.json() : { cameras: [] };
      const listed: AvailableCamera[] = data.cameras ?? [];
      setCameras(listed);
      return listed;
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
    const devices = navigator.mediaDevices;
    const handler = () => refresh();
    devices?.addEventListener("devicechange", handler);
    return () => devices?.removeEventListener("devicechange", handler);
  }, [enabled, refresh]);

  return { cameras, isLoading, refresh };
}
