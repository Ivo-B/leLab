import { useEffect, useState } from "react";
import { RobotRecord } from "@/hooks/useRobots";
import { useAvailableCameras } from "@/hooks/useAvailableCameras";
import { ImageFeatures } from "@/lib/checkpointsApi";
import {
  CameraRequest,
  CameraRotation,
  rotatedDimensions,
  rotationForCamera,
  serializeCamera,
} from "@/lib/cameraConfig";

/**
 * Binds each expected camera to a physical one. A robot camera with the same
 * name is bound automatically; the user picks the rest.
 */
export function useCameraBindings(
  imageFeatures: ImageFeatures | null,
  robot: RobotRecord | null,
  enabled: boolean,
) {
  // Per expected camera name → the chosen camera's id (or null).
  const [bindings, setBindings] = useState<Record<string, string | null>>({});
  // User rotation choice per camera id, over the one saved with the robot.
  const [rotations, setRotations] = useState<Record<string, CameraRotation>>({});
  const { cameras: availableCameras } = useAvailableCameras({ enabled });

  // If the selected robot has cameras whose names match an expected camera,
  // auto-bind them when that camera is plugged in.
  useEffect(() => {
    if (!imageFeatures) return;
    const robotCams = robot?.cameras ?? [];
    if (robotCams.length === 0 || availableCameras.length === 0) return;
    setBindings((prev) => {
      let changed = false;
      const next = { ...prev };
      for (const expectedName of Object.keys(imageFeatures)) {
        if (next[expectedName] != null) continue;
        const robotCam = robotCams.find(
          (c) => c.name.toLowerCase() === expectedName.toLowerCase(),
        );
        if (!robotCam) continue;
        if (robotCam.camera_id && availableCameras.some((c) => c.camera_id === robotCam.camera_id)) {
          next[expectedName] = robotCam.camera_id;
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [imageFeatures, robot, availableCameras]);

  const allBound =
    !!imageFeatures && Object.keys(imageFeatures).every((name) => bindings[name] != null);

  const bind = (name: string, cameraId: string) =>
    setBindings((prev) => ({ ...prev, [name]: cameraId }));

  useEffect(() => {
    if (enabled) setRotations({});
  }, [enabled, robot?.name]);

  const rotationFor = (name: string): CameraRotation => {
    const id = bindings[name];
    return id == null ? 0 : rotationForCamera(id, robot?.cameras ?? [], rotations);
  };

  const setRotation = (name: string, rotation: CameraRotation) => {
    const id = bindings[name];
    if (id != null) setRotations((prev) => ({ ...prev, [id]: rotation }));
  };

  /**
   * The backend camera config: every bound camera. The expected resolution is
   * the rotated output, so the API gets the capture dimensions. A camera the robot
   * profile has keeps its colour and lens correction switches.
   */
  const toCameraDict = (fps: number) => {
    const cameraDict: Record<string, CameraRequest> = {};
    for (const [name, dims] of Object.entries(imageFeatures ?? {})) {
      const id = bindings[name];
      if (id == null) continue;
      const rotation = rotationFor(name);
      const capture = rotatedDimensions(dims.width, dims.height, rotation);
      const saved = robot?.cameras?.find((c) => c.camera_id === id);
      cameraDict[name] = serializeCamera({
        type: "opencv",
        camera_id: id,
        color_correction: saved?.color_correction,
        lens_correction: saved?.lens_correction,
        width: capture.width,
        height: capture.height,
        rotation,
        fps,
      });
    }
    return cameraDict;
  };

  return { bindings, bind, allBound, availableCameras, rotationFor, setRotation, toCameraDict };
}
