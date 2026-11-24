import { useCallback, useEffect, useState } from "react";
import { useApi } from "@/contexts/ApiContext";
import { gainForCamera, getCameraGains } from "@/lib/cameraGain";
import { getCameraLenses, lensForCamera } from "@/lib/cameraLens";
import {
  captureVideoFrame,
  deleteCameraReference,
  getCameraReference,
  saveCameraReference,
} from "@/lib/cameraReferenceApi";

/**
 * The saved reference frame for one robot camera, as an object URL ready to use
 * as an `<img src>`. Pass `robot = undefined` to disable it.
 */
/** Which of the saved per-camera corrections to bake into a reference frame. Both default to on. */
export interface ReferenceCorrections {
  gain?: boolean;
  lens?: boolean;
}

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

  const save = useCallback(
    async (video: HTMLVideoElement, { gain = true, lens = true }: ReferenceCorrections = {}) => {
      if (!robot) return;
      setError(null);
      try {
        // The browser shows the camera before the recorder's lens and colour corrections, so apply the same
        // ones here, keyed by the stream's device label. A correction switched off for the camera is left
        // out, and the frame is stored as the camera sent it.
        const label = (video.srcObject as MediaStream | null)?.getVideoTracks()[0]?.label ?? "";
        const gains = gain ? gainForCamera(await getCameraGains(baseUrl, fetchWithHeaders), label) : null;
        const lensModel = lens ? lensForCamera(await getCameraLenses(baseUrl, fetchWithHeaders), label) : null;
        const png = await captureVideoFrame(video, { gains, lens: lensModel });
        if (!png) throw new Error("No frame to capture yet");
        await saveCameraReference(baseUrl, fetchWithHeaders, robot, camera, png);
        show(png);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    [robot, camera, baseUrl, fetchWithHeaders, show],
  );

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
