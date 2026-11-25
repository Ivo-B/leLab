import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";
import { useHfAuth } from "@/contexts/HfAuthContext";
import { RobotRecord } from "@/hooks/useRobots";
import { CameraConfig, cameraConfigurationError, serializeCameras } from "@/lib/cameraConfig";
import { useCameraBindings } from "@/hooks/useCameraBindings";
import { ImageFeatures } from "@/lib/checkpointsApi";
import { CHESS_TASK } from "@/lib/boardCalibrationApi";

/** An existing local dataset that new episodes get appended to. */
export interface AppendTarget {
  repoId: string;
  numEpisodes: number;
  fps: number;
  imageFeatures: ImageFeatures;
}

/**
 * Owns the recording modal's state and the "start recording" orchestration,
 * shared by the Landing page (new dataset) and the dataset browser (append
 * episodes to the dataset being browsed).
 */
export const useRecording = (robot: RobotRecord | null) => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const { auth } = useHfAuth();

  const [open, setOpen] = useState(false);
  const [appendTo, setAppendTo] = useState<AppendTarget | null>(null);
  const [datasetName, setDatasetName] = useState("");
  const [singleTask, setSingleTask] = useState("");
  const [chessTransfers, setChessTransfers] = useState(false);
  const [numEpisodes, setNumEpisodes] = useState(5);
  const [episodeTimeS, setEpisodeTimeS] = useState(60);
  const [resetTimeS, setResetTimeS] = useState(15);
  const [streamingEncoding, setStreamingEncoding] = useState(true);
  const [cameras, setCameras] = useState<CameraConfig[]>([]);

  // Appending must feed every camera the dataset has, at its resolution, so
  // the robot's cameras are bound to those slots the way inference binds them.
  const cameraBindings = useCameraBindings(
    appendTo?.imageFeatures ?? null,
    robot,
    open && appendTo !== null,
  );

  /** Open the modal to record a brand-new dataset with the robot's cameras. */
  const openForNew = (name: string) => {
    setDatasetName(name);
    setChessTransfers(false);
    setCameras(robot ? [...(robot.cameras ?? [])] : []);
    setOpen(true);
  };

  /** Open the modal to append episodes, prefilled with the dataset's last task; a chess dataset stays in chess transfer mode. */
  const openToAppend = (target: AppendTarget, lastTask: string) => {
    setAppendTo(target);
    setSingleTask(lastTask);
    setChessTransfers(lastTask === CHESS_TASK);
    setOpen(true);
  };

  // A bound camera keeps the capture settings the robot gives that camera, as
  // in a new recording.
  const captureSettings = (cameraId: string) => {
    const cam = robot?.cameras?.find((c) => c.camera_id === cameraId);
    return {
      ...(cam?.fourcc ? { fourcc: cam.fourcc } : {}),
      ...(cam?.backend ? { backend: cam.backend } : {}),
    };
  };

  const onStart = async () => {
    // Start stays disabled until a calibrated robot is selected.
    if (!robot) return;
    const cameraError = cameraConfigurationError(robot.cameras ?? []);
    if (cameraError) {
      toast({
        title: "Camera settings need fixing",
        description: `${cameraError} Correct the camera settings in Calibration.`,
        variant: "destructive",
      });
      return;
    }
    const task = chessTransfers ? CHESS_TASK : singleTask;
    if ((!appendTo && !datasetName) || !task) {
      toast({
        title: "Missing dataset details",
        description: appendTo
          ? "Please enter a task description."
          : "Please enter a dataset name and task description.",
        variant: "destructive",
      });
      return;
    }

    // An appended dataset keeps its exact on-disk id: the backend only stamps
    // a timestamp onto new recordings.
    const datasetRepoId = appendTo
      ? appendTo.repoId
      : auth.status === "authenticated"
        ? `${auth.username}/${datasetName}`
        : datasetName;

    const cameraDict = appendTo
      ? Object.fromEntries(
          Object.entries(cameraBindings.toCameraDict(appendTo.fps)).map(([name, cam]) => [
            name,
            { ...cam, ...captureSettings(cam.camera_id) },
          ]),
        )
      : serializeCameras(cameras);

    const recordingConfig = {
      leader_port: robot.leader_port,
      follower_port: robot.follower_port,
      leader_config: robot.leader_config,
      follower_config: robot.follower_config,
      dataset_repo_id: datasetRepoId,
      single_task: task,
      chess_transfers: chessTransfers,
      num_episodes: numEpisodes,
      episode_time_s: episodeTimeS,
      reset_time_s: resetTimeS,
      // LeRobot refuses to append frames at a different rate than the dataset's.
      fps: appendTo ? appendTo.fps : 30,
      video: true,
      push_to_hub: false,
      resume: appendTo !== null,
      streaming_encoding: streamingEncoding,
      cameras: cameraDict,
    };

    setOpen(false);
    navigate("/recording", { state: { recordingConfig } });
  };

  return {
    openForNew,
    openToAppend,
    modalProps: {
      open,
      onOpenChange: setOpen,
      robot,
      appendTo,
      cameraBindings,
      datasetName,
      setDatasetName,
      singleTask,
      setSingleTask,
      chessTransfers,
      setChessTransfers,
      numEpisodes,
      setNumEpisodes,
      episodeTimeS,
      setEpisodeTimeS,
      resetTimeS,
      setResetTimeS,
      streamingEncoding,
      setStreamingEncoding,
      cameras,
      setCameras,
      onStart,
    },
  };
};
