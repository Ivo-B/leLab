import { ChannelGains, gainForCamera, getCameraGains } from "@/lib/cameraGain";
import { useSavedCameraSetting } from "./useSavedCameraSetting";

/** The saved colour gain for the camera with this browser label, or null. */
export function useCameraGain(label: string | undefined): ChannelGains | null {
  return useSavedCameraSetting(label, getCameraGains, gainForCamera, "Colour gain");
}
