import { useCallback, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { useApi } from "@/contexts/ApiContext";
import { useCameraReference } from "@/hooks/useCameraReference";
import { Point, getBoardCalibration, imagePoint } from "@/lib/boardCalibrationApi";
import { CameraName, PoseFit, fitCameraPose } from "@/lib/cameraPoseApi";

const HINTS = ["a1", "h1", "h8", "a8"];
/** A fit further than this from the clicks is not written: the clicks are out of order or off the corners. */
const MAX_RMS_PX = 3;

function errorText(e: unknown): string {
  if (e && typeof e === "object") {
    const { message, name } = e as { message?: unknown; name?: unknown };
    if (typeof message === "string" && message) return message;
    if (typeof name === "string" && name) return name;
  }
  return String(e);
}

interface CameraPoseCardProps {
  robot: string | undefined;
  camera: CameraName;
}

/**
 * Four clicks on the board corners of one camera's saved reference give that camera's pose; Write puts it
 * into the sim cell, so simulated pictures from this camera match the real ones.
 */
export default function CameraPoseCard({ robot, camera }: CameraPoseCardProps) {
  const { baseUrl, fetchWithHeaders } = useApi();
  const reference = useCameraReference(robot, camera);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [points, setPoints] = useState<Point[]>([]);
  const [fit, setFit] = useState<PoseFit | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const complete = points.length === 4;

  const changePoints = useCallback((next: Point[]) => {
    setPoints(next);
    setFit(null);
    setMessage(null);
  }, []);

  const onImageClick = (e: React.MouseEvent<HTMLImageElement>) => {
    if (complete || !imgRef.current) return;
    changePoints([...points, imagePoint(imgRef.current, e.clientX, e.clientY)]);
  };

  const imageSize = (): [number, number] | null => {
    const img = imgRef.current;
    return img && img.naturalWidth ? [img.naturalWidth, img.naturalHeight] : null;
  };

  const useSavedCorners = async () => {
    const size = imageSize();
    if (!size) return;
    setBusy(true);
    try {
      const calibration = await getBoardCalibration(baseUrl, fetchWithHeaders);
      if (!calibration) {
        setMessage("No board calibration is saved yet; calibrate the board above first.");
        return;
      }
      const [calWidth, calHeight] = calibration.image_size;
      changePoints(
        calibration.board_corners.map(([x, y]): Point => [(x * size[0]) / calWidth, (y * size[1]) / calHeight]),
      );
    } catch (e) {
      setMessage(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  const run = async (apply: boolean) => {
    const size = imageSize();
    if (!size || !complete) return;
    setBusy(true);
    try {
      const result = await fitCameraPose(baseUrl, fetchWithHeaders, {
        camera,
        image_size: size,
        board_corners: points,
        fovy: null,
        apply,
      });
      setFit(result);
      setMessage(apply ? `Written to the sim cell as the ${camera} camera.` : null);
    } catch (e) {
      setMessage(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  if (!robot) {
    return <p className="text-sm text-muted-foreground">Select a robot on Home first.</p>;
  }
  if (reference.error) {
    return <p className="text-sm text-destructive">{reference.error}</p>;
  }
  if (!reference.url) {
    return (
      <p className="text-sm text-muted-foreground">
        Capture the {camera} camera reference on the Calibration page, then come back here.
      </p>
    );
  }

  const img = imgRef.current;
  const sx = img && img.naturalWidth ? img.clientWidth / img.naturalWidth : 1;
  const sy = img && img.naturalHeight ? img.clientHeight / img.naturalHeight : 1;
  const prompt = complete ? "All four corners set." : `Click ${HINTS[points.length]}`;
  const writable = fit !== null && !fit.applied && fit.rms_px <= MAX_RMS_PX;

  return (
    <div className="flex flex-col gap-3">
      <p className="text-sm font-medium">{camera} camera</p>
      <div className="relative inline-block self-start">
        <img
          ref={imgRef}
          src={reference.url}
          alt={`${camera} camera reference`}
          onClick={onImageClick}
          className="max-w-full cursor-crosshair rounded-md border border-border"
        />
        {points.map((point, i) => (
          <span
            key={i}
            className="pointer-events-none absolute h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border border-white"
            style={{ left: point[0] * sx, top: point[1] * sy, background: "#f59e0b" }}
          />
        ))}
      </div>
      <p className="text-xs text-muted-foreground">
        Click the board corners on the straightened reference, in the order a1, h1, h8, a8.
      </p>
      <p className="text-sm font-medium">{prompt}</p>
      <div className="flex flex-wrap gap-2">
        <Button onClick={() => void run(false)} disabled={!complete || busy}>
          Fit
        </Button>
        <Button variant="outline" onClick={() => void run(true)} disabled={!writable || busy}>
          Write to sim cell
        </Button>
        <Button variant="outline" onClick={() => changePoints([])} disabled={points.length === 0 || busy}>
          Reset points
        </Button>
        {camera === "top" ? (
          <Button variant="outline" onClick={() => void useSavedCorners()} disabled={busy}>
            Use saved board corners
          </Button>
        ) : null}
      </div>
      {fit ? (
        <div className="text-sm">
          <p>
            Position {fit.pos.map((v) => Math.round(v * 1000)).join(" ")} mm · field of view {fit.fovy.toFixed(1)}°{" "}
            ({fit.fov_fitted ? "fitted" : "kept"}) · error {fit.rms_px.toFixed(1)} px
          </p>
          {fit.warnings.map((warning) => (
            <p key={warning} className="text-amber-500">
              {warning}
            </p>
          ))}
        </div>
      ) : null}
      {message ? <p className="text-sm text-muted-foreground">{message}</p> : null}
    </div>
  );
}
