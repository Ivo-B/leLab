import { CameraLens, getCameraLenses, lensForCamera } from "@/lib/cameraLens";
import { useSavedCameraSetting } from "./useSavedCameraSetting";

/** The saved fisheye lens for the camera with this browser label, or null. */
export function useCameraLens(label: string | undefined): CameraLens | null {
  return useSavedCameraSetting(label, getCameraLenses, lensForCamera, "Lens correction");
}
