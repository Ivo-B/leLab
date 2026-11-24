import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { useApi } from "@/contexts/ApiContext";
import { useAvailableCameras } from "@/hooks/useAvailableCameras";
import { useCameraReference } from "@/hooks/useCameraReference";
import { sizeConstraints } from "@/hooks/useCameraStream";
import {
  BoardCalibration,
  BoardState,
  Point,
  getBoardCalibration,
  imagePoint,
  readBoardState,
  saveBoardCalibration,
} from "@/lib/boardCalibrationApi";
import { CameraConfig, previewDeviceId } from "@/lib/cameraConfig";
import { captureCorrectedFrame } from "@/lib/cameraReferenceApi";

type Group = "board_corners" | "bin" | "park_white" | "park_black";

const STEPS: Array<{ group: Group; label: string; hints: string[] }> = [
  { group: "board_corners", label: "Board", hints: ["a1", "h1", "h8", "a8"] },
  { group: "bin", label: "Bin", hints: ["corner 1", "corner 2", "corner 3", "corner 4"] },
  { group: "park_white", label: "White parking", hints: ["corner 1", "corner 2", "corner 3", "corner 4"] },
  { group: "park_black", label: "Black parking", hints: ["corner 1", "corner 2", "corner 3", "corner 4"] },
];
const TOTAL = 16;
const COLOURS: Record<Group, string> = {
  board_corners: "#f59e0b",
  bin: "#ef4444",
  park_white: "#e5e7eb",
  park_black: "#1f2937",
};
const FILES = "abcdefgh";

function flatten(groups: Record<Group, Point[]>): Array<{ group: Group; point: Point }> {
  return STEPS.flatMap(({ group }) => groups[group].map((point) => ({ group, point })));
}

function emptyGroups(): Record<Group, Point[]> {
  return { board_corners: [], bin: [], park_white: [], park_black: [] };
}

/** A DOMException from getUserMedia can have an empty message; its name (e.g. OverconstrainedError) still says what failed. */
function errorText(e: unknown): string {
  if (e && typeof e === "object") {
    const { message, name } = e as { message?: unknown; name?: unknown };
    if (typeof message === "string" && message) return message;
    if (typeof name === "string" && name) return name;
  }
  return String(e);
}

interface BoardCalibrationCardProps {
  robot: string | undefined;
  /** The robot's camera named `top`, as saved on the Calibration page. */
  topCamera: CameraConfig | null;
}

export default function BoardCalibrationCard({ robot, topCamera }: BoardCalibrationCardProps) {
  const { baseUrl, fetchWithHeaders } = useApi();
  const reference = useCameraReference(robot, "top");
  // Listed only on Check position, so opening Setup does not probe the cameras.
  const { refresh: listCameras } = useAvailableCameras({ enabled: false });
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [groups, setGroups] = useState<Record<Group, Point[]>>(emptyGroups);
  const [existing, setExisting] = useState<BoardCalibration | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [state, setState] = useState<BoardState | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getBoardCalibration(baseUrl, fetchWithHeaders)
      .then((cal) => {
        if (!cancelled) setExisting(cal);
      })
      .catch((e) => {
        if (!cancelled) setMessage(errorText(e));
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders]);

  const ownCalibration = existing !== null && existing.robot === robot;
  const placed = flatten(groups);
  const current = STEPS.find(({ group }) => groups[group].length < 4);
  const complete = placed.length === TOTAL;

  const onImageClick = useCallback(
    (e: React.MouseEvent<HTMLImageElement>) => {
      if (!current || !imgRef.current) return;
      const point = imagePoint(imgRef.current, e.clientX, e.clientY);
      setGroups((prev) => ({ ...prev, [current.group]: [...prev[current.group], point] }));
      setMessage(null);
    },
    [current],
  );

  const save = async () => {
    const img = imgRef.current;
    if (!robot || !img || !complete) return;
    setBusy(true);
    try {
      const cal: BoardCalibration = {
        robot,
        image_size: [img.naturalWidth, img.naturalHeight],
        ...groups,
        ...(existing
          ? {
              pixel_diff_threshold: existing.pixel_diff_threshold,
              occupied_fraction: existing.occupied_fraction,
              colour_threshold: existing.colour_threshold,
            }
          : {}),
      };
      await saveBoardCalibration(baseUrl, fetchWithHeaders, cal);
      setExisting(cal);
      setMessage("Saved. Place the pieces and check the position below.");
    } catch (e) {
      setMessage(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  const checkPosition = async () => {
    const bindMessage = "Bind the top camera on the Calibration page to check the position live.";
    if (!topCamera) {
      setMessage(bindMessage);
      return;
    }
    setBusy(true);
    let stream: MediaStream | null = null;
    try {
      const deviceId = previewDeviceId(topCamera, await listCameras());
      if (!deviceId) {
        setMessage(bindMessage);
        return;
      }
      stream = await navigator.mediaDevices.getUserMedia({
        video: { deviceId: { exact: deviceId }, ...sizeConstraints(topCamera) },
      });
      const video = document.createElement("video");
      video.srcObject = stream;
      video.muted = true;
      await video.play();
      await new Promise((resolve) => setTimeout(resolve, 300));
      // The board-state diff is per pixel against the reference, so a frame at another size cannot be compared.
      const img = imgRef.current;
      const refSize = img ? `${img.naturalWidth}×${img.naturalHeight}` : "";
      const frameSize = `${video.videoWidth}×${video.videoHeight}`;
      if (img && frameSize !== refSize) {
        setMessage(
          `The top camera delivered ${frameSize} but the reference is ${refSize}; recapture the reference or fix the camera size.`,
        );
        return;
      }
      // The reference is stored lens- and colour-corrected, so the frame it is diffed against must be too.
      const png = await captureCorrectedFrame(video, baseUrl, fetchWithHeaders, {
        gain: topCamera.color_correction !== false,
        lens: topCamera.lens_correction !== false,
      });
      if (!png) throw new Error("Could not read a frame from the top camera");
      setState(await readBoardState(baseUrl, fetchWithHeaders, png));
      setMessage(null);
    } catch (e) {
      setMessage(errorText(e));
    } finally {
      stream?.getTracks().forEach((track) => track.stop());
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
        Capture the top camera reference on the Calibration page with the board empty, then come back here.
      </p>
    );
  }

  const prompt = current
    ? `Click ${current.label === "Board" ? "" : `${current.label.toLowerCase()} `}${current.hints[groups[current.group].length]}`
    : `All ${TOTAL} points set.`;

  return (
    <div className="flex flex-col gap-3">
      <p className="text-sm text-muted-foreground">
        {ownCalibration ? "A calibration is saved. Click again to replace it. " : ""}
        Order: board corners a1, h1, h8, a8; then the bin; then the white and black parking squares.
      </p>
      {existing && !ownCalibration ? (
        <p className="text-sm text-amber-500">
          The saved board calibration belongs to {existing.robot}; calibrate this one.
        </p>
      ) : null}
      <div className="relative inline-block self-start">
        <img
          ref={imgRef}
          src={reference.url}
          alt="top camera reference"
          onClick={onImageClick}
          className="max-w-full cursor-crosshair rounded-md border border-border"
        />
        {placed.map(({ group, point }, i) => {
          const img = imgRef.current;
          const sx = img && img.naturalWidth ? img.clientWidth / img.naturalWidth : 1;
          const sy = img && img.naturalHeight ? img.clientHeight / img.naturalHeight : 1;
          return (
            <span
              key={i}
              className="pointer-events-none absolute h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border border-white"
              style={{ left: point[0] * sx, top: point[1] * sy, background: COLOURS[group] }}
            />
          );
        })}
      </div>
      <p className="text-xs text-muted-foreground">
        Click on the straightened reference; recapture it after changing the lens.
      </p>
      <p className="text-sm font-medium">{prompt}</p>
      <div className="flex flex-wrap gap-2">
        <Button onClick={save} disabled={!complete || busy}>
          Save calibration
        </Button>
        <Button variant="outline" onClick={() => setGroups(emptyGroups())} disabled={placed.length === 0 || busy}>
          Reset points
        </Button>
        <Button variant="outline" onClick={checkPosition} disabled={!ownCalibration || busy}>
          Check position
        </Button>
      </div>
      {message ? <p className="text-sm text-muted-foreground">{message}</p> : null}
      {state ? (
        <div className="grid w-64 grid-cols-8 gap-px rounded-md border border-border bg-border p-px">
          {[8, 7, 6, 5, 4, 3, 2, 1].flatMap((rank) =>
            [...FILES].map((file) => {
              const name = `${file}${rank}`;
              const value = state.squares[name];
              const bg = value === "white" ? "bg-gray-100" : value === "black" ? "bg-gray-800" : "bg-background";
              return <div key={name} title={`${name}: ${value}`} className={`aspect-square ${bg}`} />;
            }),
          )}
        </div>
      ) : null}
    </div>
  );
}
