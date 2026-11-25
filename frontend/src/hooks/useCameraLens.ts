import { CameraLens, getCameraLenses, lensForCamera } from "@/lib/cameraLens";
import { useSavedCameraSetting } from "./useSavedCameraSetting";

/** The saved fisheye lens for the camera with this browser label (and OpenCV index, when known), or null. */
export function useCameraLens(label: string | undefined, index?: number): CameraLens | null {
  return useSavedCameraSetting(label, index, getCameraLenses, lensForCamera, "Lens correction");
}
