import React from "react";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { AvailableCamera } from "@/hooks/useAvailableCameras";
import { ImageFeatures } from "@/lib/checkpointsApi";
import ServerCameraPreview from "@/components/control/ServerCameraPreview";
import CameraRotationSelect from "@/components/control/CameraRotationSelect";
import { CameraRotation, rotatedDimensions } from "@/lib/cameraConfig";

interface CameraBindingsProps {
  imageFeatures: ImageFeatures;
  bindings: Record<string, string | null>;
  onBind: (name: string, cameraId: string) => void;
  rotationFor: (name: string) => CameraRotation;
  onRotate: (name: string, rotation: CameraRotation) => void;
  availableCameras: AvailableCamera[];
  /** Drop the previews while the session is starting. */
  paused: boolean;
}

/** One row per expected camera: its resolution, a physical camera picker and a preview. */
const CameraBindings: React.FC<CameraBindingsProps> = ({
  imageFeatures,
  bindings,
  onBind,
  rotationFor,
  onRotate,
  availableCameras,
  paused,
}) => (
  <>
    {Object.entries(imageFeatures).map(([name, dims]) => {
      const value = bindings[name];
      const bound =
        value != null ? availableCameras.find((c) => c.camera_id === value) : undefined;
      const rotation = rotationFor(name);
      const capture = rotatedDimensions(dims.width, dims.height, rotation);
      return (
        <div key={name} className="flex flex-wrap items-center gap-3">
          <div className="flex-1">
            <Label className="text-sm font-medium text-gray-200">{name}</Label>
            <p className="text-xs text-gray-500">
              {dims.width}×{dims.height}
            </p>
          </div>
          <Select
            value={value ?? undefined}
            onValueChange={(v) => onBind(name, v)}
          >
            <SelectTrigger className="bg-gray-800 border-gray-700 text-white w-80">
              <SelectValue placeholder="Select a camera" />
            </SelectTrigger>
            <SelectContent className="bg-gray-900 border-gray-700 text-white">
              {availableCameras.length === 0 ? (
                <div className="px-2 py-1.5 text-xs text-gray-500">
                  No cameras detected
                </div>
              ) : (
                availableCameras.map((cam) => (
                  <SelectItem key={cam.camera_id} value={cam.camera_id}>
                    {cam.label}
                  </SelectItem>
                ))
              )}
            </SelectContent>
          </Select>
          <CameraRotationSelect
            rotation={rotation}
            disabled={!bound || paused}
            onChange={(angle) => onRotate(name, angle)}
          />
          <ServerCameraPreview
            camera={{ camera_id: bound?.camera_id ?? "", width: capture.width, height: capture.height, rotation }}
            paused={paused}
            className="w-32 rounded border border-gray-700"
          />
        </div>
      );
    })}
  </>
);

export default CameraBindings;
