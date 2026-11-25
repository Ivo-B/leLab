import React, { useEffect, useState } from "react";
import { VideoOff } from "lucide-react";
import { useOptionalApi } from "@/contexts/ApiContext";
import { CameraRotation, cameraPreviewGeometry, cameraRotationError } from "@/lib/cameraConfig";
import { PreviewCamera, previewUrl } from "@/lib/cameraApi";

const RETRY_MS = 3000;

interface Props {
  camera: PreviewCamera & { rotation?: CameraRotation };
  paused?: boolean;
  className?: string;
  fallback?: string;
  /** Saved reference frame, drawn over the live picture as a translucent ghost. */
  ghostSrc?: string;
  ghostOpacity?: number;
}

/**
 * The server's picture of this camera: the OpenCV device the recorder opens, with the saved lens and colour
 * gain already applied. The stream is unrotated, like the saved reference, so both are rotated here.
 */
export default function ServerCameraPreview({
  camera,
  paused = false,
  className = "",
  fallback,
  ghostSrc,
  ghostOpacity = 0.5,
}: Props) {
  const baseUrl = useOptionalApi()?.baseUrl ?? "http://localhost:8000";
  const [failures, setFailures] = useState(0);
  const [failed, setFailed] = useState(false);
  const url = camera.camera_id ? previewUrl(baseUrl, camera) : "";
  useEffect(() => {
    setFailed(false);
    setFailures(0);
  }, [url]);
  useEffect(() => {
    if (!failed) return;
    // The camera may be busy recording or unplugged for a moment; ask again.
    const timer = window.setTimeout(() => {
      setFailures((n) => n + 1);
      setFailed(false);
    }, RETRY_MS);
    return () => window.clearTimeout(timer);
  }, [failed]);
  const src = url && failures ? `${url}&retry=${failures}` : url;
  const rotation = camera.rotation ?? 0;
  const rotationError = cameraRotationError({ rotation });
  const geometry = rotationError
    ? { aspectRatio: `${camera.width} / ${camera.height}`, videoStyle: {} }
    : cameraPreviewGeometry(camera.width, camera.height, rotation);
  const show = !rotationError && !paused && !!src && !failed;
  return (
    <div className={`relative overflow-hidden bg-gray-800 ${className}`} style={{ aspectRatio: geometry.aspectRatio }}>
      {show ? (
        <>
          <img src={src} alt="" style={geometry.videoStyle} onError={() => setFailed(true)} />
          {ghostSrc && (
            <img
              src={ghostSrc}
              alt=""
              aria-hidden
              className="pointer-events-none"
              style={{ ...geometry.videoStyle, opacity: ghostOpacity }}
            />
          )}
        </>
      ) : (
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <VideoOff className="w-5 h-5 text-gray-500 mb-1" />
          <span className="text-xs text-gray-500">
            {rotationError ?? fallback ?? (paused ? "Preview paused" : !src ? "No camera selected" : "No picture, retrying…")}
          </span>
        </div>
      )}
    </div>
  );
}
