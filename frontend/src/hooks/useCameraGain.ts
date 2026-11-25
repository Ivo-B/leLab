import { ChannelGains, gainForCamera, getCameraGains } from "@/lib/cameraGain";
import { useSavedCameraSetting } from "./useSavedCameraSetting";

/** The saved colour gain for the camera with this browser label (and OpenCV index, when known), or null. */
export function useCameraGain(label: string | undefined, index?: number): ChannelGains | null {
  return useSavedCameraSetting(label, index, getCameraGains, gainForCamera, "Colour gain");
}
