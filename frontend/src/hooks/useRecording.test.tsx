import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CHESS_TASK } from "@/lib/boardCalibrationApi";
import type { RobotRecord } from "./useRobots";
import { useRecording } from "./useRecording";

const navigate = vi.hoisted(() => vi.fn());
vi.mock("react-router-dom", () => ({ useNavigate: () => navigate }));
vi.mock("@/hooks/use-toast", () => ({ useToast: () => ({ toast: vi.fn() }) }));
vi.mock("@/contexts/HfAuthContext", () => ({ useHfAuth: () => ({ auth: { status: "unauthenticated" } }) }));
vi.mock("@/hooks/useCameraBindings", () => ({
  useCameraBindings: () => ({ availableCameras: [], toCameraDict: () => ({}), allBound: true }),
}));

const ROBOT: RobotRecord = {
  name: "arm",
  leader_port: "/dev/leader",
  follower_port: "/dev/follower",
  leader_config: "leader",
  follower_config: "follower",
  cameras: [],
  is_clean: true,
};
const APPEND_TARGET = { repoId: "local/chess-real", numEpisodes: 3, fps: 30, imageFeatures: {} };

beforeEach(() => navigate.mockReset());

describe("useRecording chess transfer mode", () => {
  it("starts with an empty task field and posts the chess task", async () => {
    const { result } = renderHook(() => useRecording(ROBOT));
    act(() => result.current.openForNew("chess-real"));
    act(() => result.current.modalProps.setChessTransfers(true));
    await act(() => result.current.modalProps.onStart());

    expect(navigate).toHaveBeenCalledTimes(1);
    const { recordingConfig } = navigate.mock.calls[0][1].state;
    expect(recordingConfig.single_task).toBe(CHESS_TASK);
    expect(recordingConfig.chess_transfers).toBe(true);
  });

  it("turns the mode off for a new dataset", () => {
    const { result } = renderHook(() => useRecording(ROBOT));
    act(() => result.current.modalProps.setChessTransfers(true));
    act(() => result.current.openForNew("plain"));
    expect(result.current.modalProps.chessTransfers).toBe(false);
  });

  it("turns the mode on when appending to a dataset recorded with the chess task", () => {
    const { result } = renderHook(() => useRecording(ROBOT));
    act(() => result.current.openToAppend(APPEND_TARGET, CHESS_TASK));
    expect(result.current.modalProps.chessTransfers).toBe(true);
    act(() => result.current.openToAppend(APPEND_TARGET, "pick up the cube"));
    expect(result.current.modalProps.chessTransfers).toBe(false);
  });
});
