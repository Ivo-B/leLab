import React from "react";
import ServerCameraPreview from "./ServerCameraPreview";
import { CameraConfig } from "@/lib/cameraConfig";

interface CameraFeedProps {
  camera: CameraConfig;
}

const CameraFeed: React.FC<CameraFeedProps> = ({ camera }) => (
  <div className="bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
    <ServerCameraPreview camera={camera} />
    <div className="p-2 text-sm text-gray-300 truncate border-t border-gray-800">{camera.name}</div>
  </div>
);

export default CameraFeed;
