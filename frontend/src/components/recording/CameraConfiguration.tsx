import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Camera, Plus, X, RefreshCw, ChevronRight } from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { useToast } from "@/hooks/use-toast";
import { AvailableCamera, useAvailableCameras } from "@/hooks/useAvailableCameras";
import { useCameraReference } from "@/hooks/useCameraReference";
import ServerCameraPreview from "@/components/control/ServerCameraPreview";
import CameraRotationSelect from "@/components/control/CameraRotationSelect";
import { Switch } from "@/components/ui/switch";
import CameraReferenceControls from "@/components/recording/CameraReferenceControls";
import {
  CameraConfig,
  cameraRotation,
  cameraRotationError,
  rotatedDimensions,
} from "@/lib/cameraConfig";
import { describeGain, describeLens } from "@/lib/cameraApi";
export type { CameraConfig } from "@/lib/cameraConfig";

// Sentinels distinguish "leave unset" (auto-detect / platform default) from an
// explicit choice. Radix Select disallows an empty-string value, so we map these
// to `undefined` on the CameraConfig.
const FOURCC_AUTO = "__auto__";
const BACKEND_DEFAULT = "__default__";
const FOURCC_OPTIONS = ["MJPG", "YUYV", "I420", "NV12", "H264", "MP4V"];
// Mirrors lerobot's Cv2Backends enum names.
const BACKEND_OPTIONS = [
  "ANY",
  "V4L2",
  "DSHOW",
  "PVAPI",
  "ANDROID",
  "AVFOUNDATION",
  "MSMF",
];

interface CameraConfigurationProps {
  cameras: CameraConfig[];
  onCamerasChange: (cameras: CameraConfig[]) => void;
  // When true, hide the add/remove/edit controls and show the configured
  // cameras as read-only live previews. Used in the recording dialog, where
  // cameras are managed on the Calibration page rather than added ad hoc.
  readOnly?: boolean;
  // Robot the cameras belong to. In read-only mode this enables the reference
  // frame controls, which are stored per robot and camera.
  robotName?: string;
}

const CameraConfiguration: React.FC<CameraConfigurationProps> = ({
  cameras,
  onCamerasChange,
  readOnly = false,
  robotName,
}) => {
  const { toast } = useToast();

  const {
    cameras: availableCameras,
    isLoading: isLoadingCameras,
    refresh: refreshCameras,
  } = useAvailableCameras();
  const [selectedCameraId, setSelectedCameraId] = useState<string>("");
  const [cameraName, setCameraName] = useState("");

  const addCamera = () => {
    if (!selectedCameraId || !cameraName.trim()) {
      toast({
        title: "Missing Information",
        description: "Please select a camera and provide a name.",
        variant: "destructive",
      });
      return;
    }

    const selectedCamera = availableCameras.find((cam) => cam.camera_id === selectedCameraId);
    if (!selectedCamera) {
      toast({
        title: "Invalid Camera",
        description: "Selected camera is not available.",
        variant: "destructive",
      });
      return;
    }

    if (cameras.some((cam) => cam.camera_id === selectedCamera.camera_id)) {
      toast({
        title: "Camera Already Added",
        description: "This camera is already in the configuration.",
        variant: "destructive",
      });
      return;
    }

    const newCamera: CameraConfig = {
      id: `camera_${Date.now()}`,
      name: cameraName.trim(),
      type: "opencv",
      camera_id: selectedCamera.camera_id,
      width: 640,
      height: 480,
      fps: 30,
      rotation: 0,
    };

    onCamerasChange([...cameras, newCamera]);

    setSelectedCameraId("");
    setCameraName("");

    toast({
      title: "Camera Added",
      description: `${newCamera.name} has been added to the configuration.`,
    });
  };

  const removeCamera = (cameraId: string) => {
    onCamerasChange(cameras.filter((cam) => cam.id !== cameraId));
    toast({
      title: "Camera Removed",
      description: "Camera has been removed from the configuration.",
    });
  };

  const updateCamera = (cameraId: string, updates: Partial<CameraConfig>) => {
    onCamerasChange(
      cameras.map((cam) =>
        cam.id === cameraId ? { ...cam, ...updates } : cam
      )
    );
  };

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-white border-b border-gray-700 pb-2">
        {readOnly ? "Cameras" : "Camera Configuration"}
      </h3>

      {/* Add Camera Section — hidden in read-only mode (cameras are managed on
          the Calibration page, not added here). */}
      {!readOnly && (
      <div className="bg-gray-800/50 rounded-lg p-4 space-y-4">
        <h4 className="text-md font-medium text-gray-300">Add Camera</h4>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-sm font-medium text-gray-300">
                Available Cameras
              </Label>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={() => refreshCameras()}
                disabled={isLoadingCameras}
                className="h-6 w-6 text-gray-400 hover:text-white"
                title="Rescan for cameras (e.g. after plugging in a new USB camera)"
                aria-label="Rescan for cameras"
              >
                <RefreshCw
                  className={`w-3.5 h-3.5 ${isLoadingCameras ? "animate-spin" : ""}`}
                />
              </Button>
            </div>
            <Select
              value={selectedCameraId}
              onValueChange={setSelectedCameraId}
              disabled={isLoadingCameras}
            >
              <SelectTrigger className="bg-gray-800 border-gray-700 text-white">
                <SelectValue
                  placeholder={
                    isLoadingCameras ? "Loading cameras..." : "Select camera"
                  }
                />
              </SelectTrigger>
              <SelectContent className="bg-gray-800 border-gray-700">
                {availableCameras.map((camera) => {
                  const alreadyAdded = cameras.some((cam) => cam.camera_id === camera.camera_id);
                  return (
                    <SelectItem
                      key={camera.camera_id}
                      value={camera.camera_id}
                      className="text-white hover:bg-gray-700"
                      disabled={alreadyAdded}
                    >
                      <span className="font-medium">{camera.label}</span>
                      {alreadyAdded && <span className="text-xs text-gray-400"> · already added</span>}
                    </SelectItem>
                  );
                })}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label className="text-sm font-medium text-gray-300">
              Camera Name
            </Label>
            <Input
              value={cameraName}
              onChange={(e) => setCameraName(e.target.value)}
              placeholder="e.g., workspace_cam"
              className="bg-gray-800 border-gray-700 text-white"
            />
          </div>

          <div className="space-y-2 flex flex-col justify-end">
            <Button
              onClick={addCamera}
              className="bg-blue-500 hover:bg-blue-600 text-white"
              disabled={!selectedCameraId || !cameraName.trim()}
            >
              <Plus className="w-4 h-4 mr-2" />
              Add Camera
            </Button>
          </div>
        </div>
      </div>
      )}

      {/* Configured Cameras */}
      {cameras.length > 0 && (
        <div className="space-y-4">
          {!readOnly && (
            <h4 className="text-md font-medium text-gray-300">
              Configured Cameras ({cameras.length})
            </h4>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 gap-4">
            {cameras.map((camera) => (
              <CameraCard
                key={camera.id}
                camera={camera}
                attachedCameras={availableCameras}
                camerasLoaded={!isLoadingCameras}
                takenIds={cameras.filter((c) => c.id !== camera.id).map((c) => c.camera_id)}
                readOnly={readOnly}
                robotName={robotName}
                onRemove={() => removeCamera(camera.id)}
                onUpdate={(updates) => updateCamera(camera.id, updates)}
              />
            ))}
          </div>
        </div>
      )}

      {cameras.length === 0 && (
        <div className="text-center py-8 text-gray-500">
          <Camera className="w-12 h-12 mx-auto mb-4 text-gray-600" />
          <p>
            {readOnly
              ? "No cameras are configured for this robot."
              : "No cameras configured. Add a camera to get started."}
          </p>
        </div>
      )}
    </div>
  );
};

interface CameraCardProps {
  camera: CameraConfig;
  /** The cameras plugged in now, labelled as the colour and lens tools label them. */
  attachedCameras: AvailableCamera[];
  camerasLoaded: boolean;
  /** Ids the other cards already use. */
  takenIds: string[];
  readOnly: boolean;
  robotName?: string;
  onRemove: () => void;
  onUpdate: (updates: Partial<CameraConfig>) => void;
}

const NO_CAMERA = "__none__";

const CameraCard: React.FC<CameraCardProps> = ({
  camera,
  attachedCameras,
  camerasLoaded,
  takenIds,
  readOnly,
  robotName,
  onRemove,
  onUpdate,
}) => {
  const [ghostOpacity, setGhostOpacity] = useState(0.5);
  const reference = useCameraReference(readOnly ? robotName : undefined, camera.name);
  const attached = attachedCameras.find((item) => item.camera_id === camera.camera_id);
  const lens = describeLens(attached?.lens);
  const gain = describeGain(attached?.gain);
  const rotationError = cameraRotationError(camera);
  const rotation = rotationError ? null : cameraRotation(camera.rotation);
  const output = rotation === null ? null : rotatedDimensions(camera.width, camera.height, rotation);
  const unplugged = !!camera.camera_id && camerasLoaded && !attached;
  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
      <ServerCameraPreview
        camera={unplugged ? { ...camera, camera_id: "" } : camera}
        ghostSrc={reference.url ?? undefined}
        ghostOpacity={ghostOpacity}
        fallback={unplugged ? "Camera not attached" : undefined}
      />

      {/* Camera Info */}
      <div className="p-3 space-y-2">
        <div className="flex items-center justify-between">
          <div className="min-w-0">
            <h5 className="font-medium text-white truncate">{camera.name}</h5>
            {readOnly && <p className="text-xs text-gray-400 truncate">{attached?.label ?? "Camera not attached"}</p>}
          </div>
          {!readOnly && (
            <Button
              onClick={onRemove}
              size="sm"
              variant="ghost"
              className="text-red-400 hover:text-red-300 hover:bg-red-900/20 p-1"
            >
              <X className="w-4 h-4" />
            </Button>
          )}
        </div>
        {!readOnly && (
          <select
            aria-label="Camera"
            value={camera.camera_id || NO_CAMERA}
            onChange={(e) => e.target.value !== NO_CAMERA && onUpdate({ camera_id: e.target.value })}
            className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-white"
          >
            {!camera.camera_id && <option value={NO_CAMERA}>Pick the camera</option>}
            {camera.camera_id && !attached && <option value={camera.camera_id}>Not attached</option>}
            {attachedCameras.map((item) => (
              <option key={item.camera_id} value={item.camera_id} disabled={takenIds.includes(item.camera_id)}>
                {item.label}
                {takenIds.includes(item.camera_id) ? " · used by another camera" : ""}
              </option>
            ))}
          </select>
        )}
        <div className="flex items-center justify-between gap-3">
          <div>
            <Label htmlFor={`colour-${camera.id}`} className="text-xs text-gray-300">
              Colour correction
            </Label>
            <p className="text-xs text-gray-400">
              {camera.color_correction === false ? "Off" : `On · ${gain ?? "no saved gain for this camera"}`}
            </p>
          </div>
          {!readOnly && (
            <Switch
              id={`colour-${camera.id}`}
              checked={camera.color_correction !== false}
              onCheckedChange={(checked) => onUpdate({ color_correction: checked })}
              aria-label="Colour correction"
            />
          )}
        </div>
        <div className="flex items-center justify-between gap-3">
          <div>
            <Label htmlFor={`lens-${camera.id}`} className="text-xs text-gray-300">
              Lens correction
            </Label>
            <p className="text-xs text-gray-400">
              {camera.lens_correction === false ? "Off" : `On · ${lens ?? "no saved lens for this camera"}`}
            </p>
          </div>
          {!readOnly && (
            <Switch
              id={`lens-${camera.id}`}
              checked={camera.lens_correction !== false}
              onCheckedChange={(checked) => onUpdate({ lens_correction: checked })}
              aria-label="Lens correction"
            />
          )}
        </div>

        {readOnly && robotName && (
          <CameraReferenceControls
            hasReference={!!reference.url}
            opacity={ghostOpacity}
            onOpacityChange={setGhostOpacity}
            onSave={() => void reference.save()}
            onClear={() => void reference.clear()}
            disabled={!attached || !!rotationError}
            error={reference.error}
          />
        )}
        {rotationError && <p role="alert" className="text-xs text-red-400">{rotationError} Select a valid rotation.</p>}
        {readOnly ? (
          <p className="text-xs text-gray-400">
            {output ? `${output.width}×${output.height} · ${rotation}° clockwise` : "Invalid rotation"}
            {camera.fps ? ` · ${camera.fps} fps` : ""}
          </p>
        ) : (
        <>
        <CameraRotationSelect rotation={rotation} onChange={(value) => onUpdate({ rotation: value })} />
        <Collapsible>
          <CollapsibleTrigger className="group flex items-center gap-1.5 text-xs font-medium text-gray-300 hover:text-white transition-colors">
            <ChevronRight className="w-3.5 h-3.5 transition-transform group-data-[state=open]:rotate-90" />
            Configuration
          </CollapsibleTrigger>
          <CollapsibleContent className="pt-2 space-y-2">
            <div className="grid grid-cols-1 gap-2 text-xs text-gray-400">
              <div className="flex items-center gap-2">
                <span className="w-16">Capture:</span>
                <div className="flex items-center gap-1">
                  <NumberInput
                    value={camera.width}
                    onChange={(v) => {
                      if (v !== undefined) onUpdate({ width: v });
                    }}
                    className="bg-gray-800 border-gray-700 text-white text-xs h-6 px-2 w-16"
                    min="320"
                    max="1920"
                  />
                  <span className="flex items-center">×</span>
                  <NumberInput
                    value={camera.height}
                    onChange={(v) => {
                      if (v !== undefined) onUpdate({ height: v });
                    }}
                    className="bg-gray-800 border-gray-700 text-white text-xs h-6 px-2 w-16"
                    min="240"
                    max="1080"
                  />
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-16">FPS:</span>
                <NumberInput
                  value={camera.fps ?? 30}
                  onChange={(v) => {
                    if (v !== undefined) onUpdate({ fps: v });
                  }}
                  className="bg-gray-800 border-gray-700 text-white text-xs h-6 px-2 w-16"
                  min="10"
                  max="60"
                />
              </div>
              <div className="flex items-center gap-2">
                <span className="w-16">FOURCC:</span>
                <Select
                  value={camera.fourcc ?? FOURCC_AUTO}
                  onValueChange={(v) =>
                    onUpdate({ fourcc: v === FOURCC_AUTO ? undefined : v })
                  }
                >
                  <SelectTrigger className="bg-gray-800 border-gray-700 text-white text-xs h-6 px-2 w-28">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-gray-800 border-gray-700">
                    <SelectItem
                      value={FOURCC_AUTO}
                      className="text-white hover:bg-gray-700 text-xs"
                    >
                      Auto
                    </SelectItem>
                    {FOURCC_OPTIONS.map((code) => (
                      <SelectItem
                        key={code}
                        value={code}
                        className="text-white hover:bg-gray-700 text-xs"
                      >
                        {code}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-16">Backend:</span>
                <Select
                  value={camera.backend ?? BACKEND_DEFAULT}
                  onValueChange={(v) =>
                    onUpdate({ backend: v === BACKEND_DEFAULT ? undefined : v })
                  }
                >
                  <SelectTrigger className="bg-gray-800 border-gray-700 text-white text-xs h-6 px-2 w-28">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-gray-800 border-gray-700">
                    <SelectItem
                      value={BACKEND_DEFAULT}
                      className="text-white hover:bg-gray-700 text-xs"
                    >
                      Default
                    </SelectItem>
                    {BACKEND_OPTIONS.map((name) => (
                      <SelectItem
                        key={name}
                        value={name}
                        className="text-white hover:bg-gray-700 text-xs"
                      >
                        {name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <p className="text-[10px] text-gray-500 leading-tight">
                Overriding the backend can reorder camera indices on macOS.
              </p>
            </div>
            <div className="text-xs text-gray-500">
              Type: {camera.type} | Camera id: {camera.camera_id || "none"}
            </div>
          </CollapsibleContent>
        </Collapsible>
        </>
        )}
      </div>
    </div>
  );
};

export default CameraConfiguration;
