import { useEffect, useState } from "react";
import { useOptionalApi } from "@/contexts/ApiContext";
import type { Fetcher } from "@/lib/apiClient";

/**
 * A setting the server saves per camera name (colour gain, lens), for the camera with this browser label.
 * `load` fetches every saved value; `pick` finds this label's. Null when there is no label yet, nothing is saved
 * for it, or the server cannot be reached; the preview then shows the camera as it is.
 */
export function useSavedCameraSetting<All, One>(
  label: string | undefined,
  load: (baseUrl: string, fetcher: Fetcher, signal: AbortSignal) => Promise<All>,
  pick: (all: All, label: string) => One | null,
  what: string,
): One | null {
  const api = useOptionalApi();
  const baseUrl = api?.baseUrl;
  const fetchWithHeaders = api?.fetchWithHeaders;
  const [value, setValue] = useState<One | null>(null);

  useEffect(() => {
    if (!label || !baseUrl || !fetchWithHeaders) {
      setValue(null);
      return;
    }
    const controller = new AbortController();
    load(baseUrl, fetchWithHeaders, controller.signal)
      .then((all) => {
        if (!controller.signal.aborted) setValue(pick(all, label));
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        console.warn(`${what} unavailable; showing the camera as it is`, error);
        setValue(null);
      });
    return () => controller.abort();
    // `load` and `pick` are module-level functions, so they never change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [label, baseUrl, fetchWithHeaders]);

  return value;
}
