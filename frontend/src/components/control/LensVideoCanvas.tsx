import React, { CSSProperties, RefObject, useEffect, useRef } from "react";
import { CameraLens } from "@/lib/cameraLens";
import { createLensRenderer } from "@/lib/lensRenderer";

interface Props {
  /** The (hidden) video element that receives the camera stream. */
  videoRef: RefObject<HTMLVideoElement | null>;
  lens: CameraLens;
  style?: CSSProperties;
  /** Called once when the browser cannot run the lens shader, so the parent can show the plain video. */
  onUnsupported: () => void;
}

/** The camera stream drawn through the saved fisheye lens, so the preview shows the straightened picture the recorder records. */
export default function LensVideoCanvas({ videoRef, lens, style, onUnsupported }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lensRef = useRef(lens);
  lensRef.current = lens;
  const unsupportedRef = useRef(onUnsupported);
  unsupportedRef.current = onUnsupported;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const renderer = createLensRenderer(canvas);
    if (!renderer) {
      unsupportedRef.current();
      return;
    }
    let frame = 0;
    const tick = () => {
      const video = videoRef.current;
      if (video && video.readyState >= 2) renderer.draw(video, lensRef.current);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(frame);
      renderer.dispose();
    };
  }, [videoRef]);

  return <canvas ref={canvasRef} data-testid="lens-canvas" aria-hidden="true" style={style} />;
}
