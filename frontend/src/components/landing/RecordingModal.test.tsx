import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { CHESS_TASK } from "@/lib/boardCalibrationApi";
import RecordingModal from "./RecordingModal";

vi.mock("@/contexts/HfAuthContext", () => ({ useHfAuth: () => ({ auth: { status: "unknown" } }) }));
vi.mock("@/components/recording/CameraConfiguration", () => ({ default: () => null }));

function renderModal(chessTransfers: boolean, setChessTransfers = vi.fn()) {
  render(
    <RecordingModal
      open
      onOpenChange={vi.fn()}
      robot={null}
      appendTo={null}
      cameraBindings={{} as never}
      datasetName="chess-real"
      setDatasetName={vi.fn()}
      singleTask="my own task"
      setSingleTask={vi.fn()}
      chessTransfers={chessTransfers}
      setChessTransfers={setChessTransfers}
      numEpisodes={3}
      setNumEpisodes={vi.fn()}
      episodeTimeS={30}
      setEpisodeTimeS={vi.fn()}
      resetTimeS={10}
      setResetTimeS={vi.fn()}
      streamingEncoding
      setStreamingEncoding={vi.fn()}
      cameras={[]}
      setCameras={vi.fn()}
      onStart={vi.fn()}
    />,
  );
  return setChessTransfers;
}

describe("RecordingModal chess transfer mode", () => {
  it("keeps the typed task while the mode is off and toggles it on", () => {
    const setChessTransfers = renderModal(false);
    expect(screen.getByLabelText(/Task Description/)).toHaveValue("my own task");
    expect(screen.getByLabelText(/Task Description/)).toBeEnabled();
    fireEvent.click(screen.getByRole("switch", { name: "Chess transfer mode" }));
    expect(setChessTransfers).toHaveBeenCalledWith(true);
  });

  it("fixes the task to the chess task string when on", () => {
    renderModal(true);
    expect(screen.getByLabelText(/Task Description/)).toHaveValue(CHESS_TASK);
    expect(screen.getByLabelText(/Task Description/)).toBeDisabled();
  });
});
