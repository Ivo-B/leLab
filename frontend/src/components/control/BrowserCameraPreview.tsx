import React, { useId, useState } from "react";
import { VideoOff } from "lucide-react";
import { useCameraStream } from "@/hooks/useCameraStream";
import { useCameraGain } from "@/hooks/useCameraGain";
import { useCameraLens } from "@/hooks/useCameraLens";
import LensVideoCanvas from "@/components/control/LensVideoCanvas";
import { CameraRotation, cameraPreviewGeometry, cameraRotationError } from "@/lib/cameraConfig";
import { gainMatrix } from "@/lib/cameraGain";

interface Props {
  deviceId: string;
  paused?: boolean;
  rotation?: CameraRotation;
  width?: number;
  height?: number;
  className?: string;
  fallback?: string;
  /** Saved reference frame, drawn over the live video as a translucent ghost. */
  ghostSrc?: string;
  ghostOpacity?: number;
  /** When false, the preview shows the camera's own colours. */
  colorCorrection?: boolean;
  /** When false, the preview shows the lens as it is, without the saved fisheye correction. */
  lensCorrection?: boolean;
}

/**
 * Browser streams contain raw frames, so only browser previews use CSS rotation. The saved colour gain of
 * this camera is applied as an SVG filter, and its saved fisheye lens by drawing the stream through a WebGL
 * shader, so the preview shows what the recorder records. The stream's own `<video>` stays in the page, hidden
 * while the lens canvas is shown, so streaming and reference capture work the same with or without a lens.
 * The box keeps the recorded capture size's shape even when the browser stream
 * has another size, so the preview frames the scene the way recording will.
 */
export default function BrowserCameraPreview({
  deviceId,
  paused = false,
  rotation = 0,
  width = 640,
  height = 480,
  className = "",
  fallback,
  ghostSrc,
  ghostOpacity = 0.5,
  colorCorrection = true,
  lensCorrection = true,
}: Props) {
  const rotationError = cameraRotationError({ rotation });
  const { videoRef, hasError, label } = useCameraStream(deviceId, paused || !!rotationError, { width, height });
  const savedGain = useCameraGain(label);
  const gains = colorCorrection ? savedGain : null;
  const savedLens = useCameraLens(label);
  const lens = lensCorrection ? savedLens : null;
  const [lensUnsupported, setLensUnsupported] = useState(false);
  const lensShown = !!lens && !lensUnsupported;
  const filterId = `colour-gain-${useId().replace(/[^a-zA-Z0-9]/g, "")}`;
  const geometry = rotationError
    ? { aspectRatio: `${width} / ${height}`, videoStyle: {} }
    : cameraPreviewGeometry(width, height, rotation);
  const showVideo = !rotationError && !paused && deviceId && !hasError;
  const colourStyle = gains ? { filter: `url(#${filterId})` } : {};
  return (
    <div
      className={`relative overflow-hidden bg-gray-800 ${className}`}
      style={{ aspectRatio: geometry.aspectRatio }}
    >
      {showVideo ? (
        <>
        {gains && (
          <svg aria-hidden="true" focusable="false" width="0" height="0" className="absolute">
            <filter id={filterId} colorInterpolationFilters="sRGB">
              <feColorMatrix type="matrix" values={gainMatrix(gains)} />
            </filter>
          </svg>
        )}
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          style={
            lensShown
              ? { ...geometry.videoStyle, opacity: 0, pointerEvents: "none" }
              : { ...geometry.videoStyle, ...colourStyle }
          }
        />
        {lens && lensShown && (
          <LensVideoCanvas
            videoRef={videoRef}
            lens={lens}
            style={{ ...geometry.videoStyle, ...colourStyle }}
            onUnsupported={() => setLensUnsupported(true)}
          />
        )}
        {lens && lensUnsupported && (
          <span className="absolute bottom-1 left-1 rounded bg-black/60 px-1 text-[10px] text-amber-300">
            Lens preview needs WebGL; the recording is still corrected
          </span>
        )}
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
            {rotationError ?? fallback ?? (paused ? "Preview paused" : deviceId ? "Preview failed" : "No camera selected")}
          </span>
        </div>
      )}
    </div>
  );
}
