import { useCallback, useEffect, useState } from "react";
import { useApi } from "@/contexts/ApiContext";
import { captureCameraReference, deleteCameraReference, getCameraReference } from "@/lib/cameraReferenceApi";

/**
 * The saved reference frame for one robot camera, as an object URL ready to use
 * as an `<img src>`. Pass `robot = undefined` to disable it.
 */
export function useCameraReference(robot: string | undefined, camera: string) {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [url, setUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const show = useCallback((blob: Blob | null) => {
    setUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return blob ? URL.createObjectURL(blob) : null;
    });
  }, []);

  useEffect(() => {
    if (!robot) return;
    const controller = new AbortController();
    getCameraReference(baseUrl, fetchWithHeaders, robot, camera, controller.signal)
      .then(show)
      .catch((e) => {
        if (!controller.signal.aborted) setError(String(e.message ?? e));
      });
    return () => {
      controller.abort();
      show(null);
    };
  }, [robot, camera, baseUrl, fetchWithHeaders, show]);

  /** The server snapshots the saved profile camera, so the reference is the camera the recorder opens. */
  const save = useCallback(async () => {
    if (!robot) return;
    setError(null);
    try {
      show(await captureCameraReference(baseUrl, fetchWithHeaders, robot, camera));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [robot, camera, baseUrl, fetchWithHeaders, show]);

  const clear = useCallback(async () => {
    if (!robot) return;
    setError(null);
    try {
      await deleteCameraReference(baseUrl, fetchWithHeaders, robot, camera);
      show(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [robot, camera, baseUrl, fetchWithHeaders, show]);

  return { url, error, save, clear };
}
