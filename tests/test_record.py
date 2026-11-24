# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for lelab.record — request schemas and handler entry points."""

from __future__ import annotations

import shutil
from unittest.mock import MagicMock

import pytest


def test_recording_request_rejects_missing_required_fields() -> None:
    from pydantic import ValidationError

    from lelab.record import RecordingRequest

    with pytest.raises(ValidationError):
        RecordingRequest()


def test_normalize_dataset_repo_id_adds_local_namespace_for_bare_name() -> None:
    from lelab.record import _normalize_dataset_repo_id

    assert _normalize_dataset_repo_id("test recording") == "local/test_recording"


def test_normalize_dataset_repo_id_preserves_hub_namespace() -> None:
    from lelab.record import _normalize_dataset_repo_id

    assert _normalize_dataset_repo_id("samuel-hills/test recording") == "samuel-hills/test_recording"


def test_recording_status_handler_exposes_state_fields() -> None:
    from lelab.record import handle_recording_status

    result = handle_recording_status()
    assert isinstance(result, dict)
    # Pinning the exact keys so a rename in handle_recording_status surfaces here.
    assert "recording_active" in result
    assert "current_phase" in result
    assert "session_ended" in result
    assert "available_controls" in result


def test_handle_stop_recording_when_idle_returns_dict(tmp_lerobot_home) -> None:
    from lelab.record import handle_stop_recording

    result = handle_stop_recording()
    assert isinstance(result, dict)


def test_stop_recording_and_wait_when_idle_returns() -> None:
    from lelab.record import stop_recording_and_wait

    stop_recording_and_wait()


def test_stop_recording_and_wait_lets_the_session_finalize(monkeypatch: pytest.MonkeyPatch) -> None:
    """Server shutdown must end the session through its teardown, not kill it."""
    import threading
    import time
    from types import SimpleNamespace

    from lelab import record

    events = {"stop_recording": False, "exit_early": False}
    finalized = threading.Event()

    def session() -> None:
        while not events["stop_recording"]:
            time.sleep(0.01)
        finalized.set()

    thread = threading.Thread(target=session, daemon=True)
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "recording_events", events)
    monkeypatch.setattr(record, "recording_thread", thread)
    monkeypatch.setattr(record, "recording_config", SimpleNamespace(dataset_repo_id="local/x"))
    monkeypatch.setattr(record, "current_phase", "recording")
    monkeypatch.setattr(record, "phase_start_time", None)
    thread.start()

    record.stop_recording_and_wait()

    assert finalized.is_set()


def test_resolve_dataset_dir_rejects_traversal(tmp_lerobot_home) -> None:
    from lelab.record import _resolve_dataset_dir

    assert _resolve_dataset_dir("../../etc") is None
    assert _resolve_dataset_dir(".") is None


def test_resolve_dataset_dir_accepts_nested_repo_id(tmp_lerobot_home) -> None:
    from lelab.record import _resolve_dataset_dir

    target = _resolve_dataset_dir("alice/pusht")
    assert target == tmp_lerobot_home / "alice" / "pusht"


def test_handle_delete_dataset_rejects_traversal(tmp_lerobot_home) -> None:
    from lelab.record import DatasetInfoRequest, handle_delete_dataset

    result = handle_delete_dataset(DatasetInfoRequest(dataset_repo_id="../../etc"))
    assert result["success"] is False


def test_handle_delete_dataset_reports_missing(tmp_lerobot_home) -> None:
    from lelab.record import DatasetInfoRequest, handle_delete_dataset

    result = handle_delete_dataset(DatasetInfoRequest(dataset_repo_id="nope/nope"))
    assert result["success"] is False


def test_handle_delete_dataset_removes_directory(tmp_lerobot_home) -> None:
    from lelab.record import DatasetInfoRequest, handle_delete_dataset

    dataset_dir = tmp_lerobot_home / "alice" / "pusht"
    (dataset_dir / "meta").mkdir(parents=True)
    (dataset_dir / "meta" / "info.json").write_text("{}")

    result = handle_delete_dataset(DatasetInfoRequest(dataset_repo_id="alice/pusht"))
    assert result["success"] is True
    assert not dataset_dir.exists()


def test_cleanup_failed_recording_removes_empty_new_dataset(tmp_lerobot_home) -> None:
    from lelab.record import _cleanup_failed_recording

    dataset_dir = tmp_lerobot_home / "alice" / "pusht"
    (dataset_dir / "meta").mkdir(parents=True)
    (dataset_dir / "meta" / "info.json").write_text("{}")

    _cleanup_failed_recording("alice/pusht", resume=False, saved_episodes=0, dir_preexisted=False)
    assert not dataset_dir.exists()


def test_cleanup_failed_recording_keeps_dataset_with_saved_episodes(tmp_lerobot_home) -> None:
    from lelab.record import _cleanup_failed_recording

    dataset_dir = tmp_lerobot_home / "alice" / "pusht"
    (dataset_dir / "meta").mkdir(parents=True)
    (dataset_dir / "meta" / "info.json").write_text("{}")

    _cleanup_failed_recording("alice/pusht", resume=False, saved_episodes=2, dir_preexisted=False)
    assert dataset_dir.exists()


def test_cleanup_failed_recording_keeps_resumed_dataset(tmp_lerobot_home) -> None:
    from lelab.record import _cleanup_failed_recording

    dataset_dir = tmp_lerobot_home / "alice" / "pusht"
    (dataset_dir / "meta").mkdir(parents=True)
    (dataset_dir / "meta" / "info.json").write_text("{}")

    _cleanup_failed_recording("alice/pusht", resume=True, saved_episodes=0, dir_preexisted=False)
    assert dataset_dir.exists()


def test_cleanup_failed_recording_keeps_preexisting_dir(tmp_lerobot_home) -> None:
    from lelab.record import _cleanup_failed_recording

    # Second-resolution name collision: create() raised FileExistsError on a dir
    # this attempt doesn't own, so its contents must survive.
    dataset_dir = tmp_lerobot_home / "alice" / "pusht"
    (dataset_dir / "meta").mkdir(parents=True)
    (dataset_dir / "meta" / "info.json").write_text("{}")

    _cleanup_failed_recording("alice/pusht", resume=False, saved_episodes=0, dir_preexisted=True)
    assert (dataset_dir / "meta" / "info.json").exists()


def test_resolve_dataset_dir_follows_hf_home_when_lerobot_home_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from lelab.record import _resolve_dataset_dir

    monkeypatch.delenv("HF_LEROBOT_HOME", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    assert _resolve_dataset_dir("alice/pusht") == (tmp_path / "lerobot" / "alice" / "pusht").resolve()


def test_cleanup_failed_recording_uses_hf_home_cache(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from lelab.record import _cleanup_failed_recording

    monkeypatch.delenv("HF_LEROBOT_HOME", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    dataset_dir = tmp_path / "lerobot" / "alice" / "pusht"
    (dataset_dir / "meta").mkdir(parents=True)

    _cleanup_failed_recording("alice/pusht", resume=False, saved_episodes=0, dir_preexisted=False)
    assert not dataset_dir.exists()


def test_dataset_dir_preexisted_defaults_true_when_probe_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home
) -> None:
    from lelab import record

    def boom(repo_id):
        raise OSError("Too many levels of symbolic links")

    monkeypatch.setattr(record, "_resolve_dataset_dir", boom)
    assert record._dataset_dir_preexisted("alice/pusht") is True


def test_worker_releases_active_state_when_path_probe_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home
) -> None:
    from unittest.mock import MagicMock

    from lelab import record

    def boom(repo_id):
        raise OSError("Too many levels of symbolic links")

    def fail_recording(cfg, events):
        raise RuntimeError("recording failed")

    monkeypatch.setattr(record, "_resolve_dataset_dir", boom)
    monkeypatch.setattr(record, "create_record_config", lambda request: MagicMock())
    monkeypatch.setattr(record, "record_with_web_events", fail_recording)

    request = record.RecordingRequest(
        leader_port="/dev/null",
        follower_port="/dev/null",
        leader_config="l",
        follower_config="f",
        dataset_repo_id="alice/pusht",
        single_task="t",
    )
    result = record.handle_start_recording(request)
    assert result["success"] is True
    record.recording_thread.join(timeout=10)

    assert not record.recording_thread.is_alive()
    assert record.recording_active is False
    # The next start isn't rejected as "already active".
    assert record.handle_start_recording(request)["success"] is True
    record.recording_thread.join(timeout=10)
    assert record.recording_active is False


def test_cleanup_failed_recording_tolerates_missing_dir(tmp_lerobot_home) -> None:
    from lelab.record import _cleanup_failed_recording

    # No dataset dir was ever created (e.g. failure before LeRobotDataset.create()) — no-op, no crash.
    _cleanup_failed_recording("alice/never-created", resume=False, saved_episodes=0, dir_preexisted=False)


def test_create_record_config_pins_dshow_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Windows, recording must use the DSHOW backend so a camera_index opens
    the same device /available-cameras enumerated (via pygrabber, DSHOW order).
    """
    import lelab.record as record
    from lerobot.cameras.configs import Cv2Backends

    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr(record, "setup_calibration_files", lambda leader, follower: ("leader", "follower"))

    request = record.RecordingRequest(
        leader_port="COM_LEADER",
        follower_port="COM_FOLLOWER",
        leader_config="leader",
        follower_config="follower",
        dataset_repo_id="user/dataset",
        single_task="pick up the cube",
        cameras={"wrist": {"type": "opencv", "camera_index": 0, "width": 640, "height": 480, "fps": 30}},
    )

    config = record.create_record_config(request)
    assert config.robot.cameras["wrist"].backend == Cv2Backends.DSHOW


def test_build_camera_configs_uses_default_backend_when_unset() -> None:
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    cameras = {"cam": {"type": "opencv", "camera_index": 0, "width": 640, "height": 480, "fps": 30}}
    configs = _build_camera_configs(cameras, Cv2Backends.AVFOUNDATION)

    assert configs["cam"].backend == Cv2Backends.AVFOUNDATION
    assert configs["cam"].fourcc is None
    assert configs["cam"].index_or_path == 0


def test_build_camera_configs_passes_fourcc_through() -> None:
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    cameras = {"cam": {"type": "opencv", "camera_index": 0, "fourcc": "MJPG"}}
    configs = _build_camera_configs(cameras, Cv2Backends.ANY)

    assert configs["cam"].fourcc == "MJPG"


def test_build_camera_configs_explicit_backend_overrides_default() -> None:
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    cameras = {"cam": {"type": "opencv", "camera_index": 0, "backend": "V4L2"}}
    configs = _build_camera_configs(cameras, Cv2Backends.AVFOUNDATION)

    assert configs["cam"].backend == Cv2Backends.V4L2


def test_build_camera_configs_invalid_backend_raises() -> None:
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    cameras = {"cam": {"type": "opencv", "camera_index": 0, "backend": "NOPE"}}
    with pytest.raises(KeyError):
        _build_camera_configs(cameras, Cv2Backends.ANY)


def test_build_camera_configs_skips_non_opencv_type() -> None:
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    cameras = {"cam": {"type": "realsense", "camera_index": 0}}
    configs = _build_camera_configs(cameras, Cv2Backends.ANY)

    assert configs == {}


EPISODES = 2


def _record(repo_id: str):
    """Write a small finished LeRobot dataset and return its root."""
    import numpy as np

    from lerobot.datasets import LeRobotDataset

    features = {
        "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
    }
    dataset = LeRobotDataset.create(repo_id, fps=10, features=features, use_videos=False)
    for _ in range(EPISODES):
        for frame in range(5):
            values = np.full(2, frame, dtype=np.float32)
            dataset.add_frame({"action": values, "observation.state": values, "task": "pick"})
        dataset.save_episode()
    dataset.finalize()
    return dataset.root


def _resume_request(repo_id: str):
    from lelab.record import RecordingRequest

    return RecordingRequest(
        leader_port="/dev/leader",
        follower_port="/dev/follower",
        leader_config="leader",
        follower_config="follower",
        dataset_repo_id=repo_id,
        single_task="pick",
        resume=True,
    )


def test_resume_reopens_the_dataset_in_the_local_cache(
    tmp_lerobot_home, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lelab.record as record
    from lerobot.datasets import LeRobotDataset

    _record("local/ds")
    monkeypatch.setattr(record, "setup_calibration_files", lambda leader, follower: ("leader", "follower"))

    config = record.create_record_config(_resume_request("local/ds"))
    dataset = LeRobotDataset.resume(config.dataset.repo_id, root=config.dataset.root)
    assert dataset.meta.total_episodes == EPISODES


def test_resume_refuses_an_unreadable_dataset_before_the_robot(
    tmp_lerobot_home, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lelab.record as record

    root = _record("local/ds")
    shutil.rmtree(root / "meta" / "episodes")
    for path in (root / "data").rglob("*.parquet"):
        path.write_bytes(b"PAR1 truncated")
    monkeypatch.setattr(
        record, "create_record_config", lambda request: pytest.fail("reached the robot setup")
    )

    result = record.handle_start_recording(_resume_request("local/ds"))
    assert result["success"] is False
    assert "record it again" in result["message"]
    assert record.recording_active is False


def test_handle_delete_dataset_calls_delete_repo_when_requested(
    tmp_lerobot_home, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lelab.record import DatasetInfoRequest, handle_delete_dataset

    monkeypatch.setattr("lerobot.utils.constants.HF_LEROBOT_HOME", tmp_lerobot_home)
    dataset_dir = tmp_lerobot_home / "user" / "dataset"
    dataset_dir.mkdir(parents=True)

    spy = MagicMock()
    monkeypatch.setattr("huggingface_hub.delete_repo", spy)

    result = handle_delete_dataset(DatasetInfoRequest(dataset_repo_id="user/dataset", delete_from_hub=True))

    assert result["success"] is True
    spy.assert_called_once_with("user/dataset", repo_type="dataset")
    assert not dataset_dir.exists()


def test_handle_delete_dataset_skips_hub_call_by_default(
    tmp_lerobot_home, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lelab.record import DatasetInfoRequest, handle_delete_dataset

    monkeypatch.setattr("lerobot.utils.constants.HF_LEROBOT_HOME", tmp_lerobot_home)
    dataset_dir = tmp_lerobot_home / "user" / "dataset"
    dataset_dir.mkdir(parents=True)

    spy = MagicMock()
    monkeypatch.setattr("huggingface_hub.delete_repo", spy)

    result = handle_delete_dataset(DatasetInfoRequest(dataset_repo_id="user/dataset"))

    assert result["success"] is True
    spy.assert_not_called()
    assert not dataset_dir.exists()


def test_handle_delete_dataset_hub_failure_keeps_local_copy(
    tmp_lerobot_home, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lelab.record import DatasetInfoRequest, handle_delete_dataset

    monkeypatch.setattr("lerobot.utils.constants.HF_LEROBOT_HOME", tmp_lerobot_home)
    dataset_dir = tmp_lerobot_home / "user" / "dataset"
    dataset_dir.mkdir(parents=True)

    def raise_error(*args, **kwargs):
        raise RuntimeError("network error")

    monkeypatch.setattr("huggingface_hub.delete_repo", raise_error)

    result = handle_delete_dataset(DatasetInfoRequest(dataset_repo_id="user/dataset", delete_from_hub=True))

    assert result["success"] is False
    assert dataset_dir.exists()


def _chess_request(**overrides):
    from lelab.record import RecordingRequest

    fields = {
        "leader_port": "/dev/l",
        "follower_port": "/dev/f",
        "leader_config": "l",
        "follower_config": "f",
        "dataset_repo_id": "u/d",
        "single_task": "x",
        "chess_transfers": True,
        "cameras": {"top": {"type": "opencv", "camera_index": 0, "rotation": 0}},
    }
    fields.update(overrides)
    return RecordingRequest(**fields)


def test_recording_request_chess_transfers_defaults_off() -> None:
    from lelab.record import RecordingRequest

    request = RecordingRequest(
        leader_port="/dev/l",
        follower_port="/dev/f",
        leader_config="l",
        follower_config="f",
        dataset_repo_id="u/d",
        single_task="x",
    )
    assert request.chess_transfers is False


def test_start_recording_in_chess_mode_refuses_without_top_camera(tmp_lerobot_home) -> None:
    from lelab import record
    from lelab.record import RecordingRequest, handle_start_recording

    request = RecordingRequest(
        leader_port="/dev/l",
        follower_port="/dev/f",
        leader_config="l",
        follower_config="f",
        dataset_repo_id="u/d",
        single_task="whatever",
        chess_transfers=True,
        cameras={"wrist": {"type": "opencv", "camera_index": 0, "rotation": 0}},
        test_mode=True,
    )
    result = handle_start_recording(request)
    assert result["success"] is False
    assert "top" in result["message"]
    assert record.recording_active is False
    assert record.recording_thread is None or not record.recording_thread.is_alive()


def test_recording_status_reports_the_current_transfer(monkeypatch: pytest.MonkeyPatch) -> None:
    from lelab import record
    from lelab.record import RecordingRequest, handle_recording_status

    request = RecordingRequest(
        leader_port="/dev/l",
        follower_port="/dev/f",
        leader_config="l",
        follower_config="f",
        dataset_repo_id="u/d",
        single_task="x",
        chess_transfers=True,
        cameras={"top": {"type": "opencv", "camera_index": 0, "rotation": 0}},
    )
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "recording_config", request)
    monkeypatch.setattr(record, "current_transfer", ("e2", "e4"))
    status = handle_recording_status()
    assert status["chess_transfers"] is True
    assert status["transfer"] == {"source": "e2", "target": "e4"}
    monkeypatch.setattr(record, "current_transfer", None)
    assert handle_recording_status()["transfer"] is None


def test_chess_mode_installs_gain_and_lens_before_the_record_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home
) -> None:
    from lelab import record
    from lelab.chess.calibration import save_calibration
    from lelab.chess.transfers import CHESS_TASK
    from tests.chess_render import calibration

    save_calibration(calibration())
    calls: list[str] = []
    tasks: list[str] = []

    def fake_record(cfg, events):
        calls.append("record")
        tasks.append(record.recording_config.single_task)
        raise RuntimeError("stop here")

    monkeypatch.setattr(record, "create_record_config", lambda request: MagicMock())
    monkeypatch.setattr(record.time, "sleep", lambda s: None)
    monkeypatch.setattr(record.color_gain, "gains_for_session", lambda cameras: {})
    monkeypatch.setattr(record.lens, "lenses_for_session", lambda cameras: {})
    monkeypatch.setattr(record.color_gain, "install", lambda index_map: calls.append("gain"))
    monkeypatch.setattr(record.lens, "install", lambda index_map: calls.append("lens"))
    monkeypatch.setattr(record, "record_with_web_events", fake_record)

    result = record.handle_start_recording(_chess_request(single_task="pick the bolt"))
    assert result["success"] is True
    record.recording_thread.join(timeout=10)
    assert not record.recording_thread.is_alive()
    assert calls == ["gain", "lens", "record"]
    assert tasks == [CHESS_TASK]


def _overlay_session(monkeypatch: pytest.MonkeyPatch):
    import random

    from lelab import record
    from lelab.chess.overlay import GoalOverlayStep
    from lelab.chess.transfers import TransferSampler
    from tests.chess_render import calibration

    step = GoalOverlayStep(calibration=calibration())
    monkeypatch.setattr(record, "_overlay_step", step)
    monkeypatch.setattr(record, "_transfer_sampler", TransferSampler(random.Random(3)))
    monkeypatch.setattr(record, "current_transfer", None)
    return step


def test_a_transfer_stays_until_its_episode_is_saved(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from types import SimpleNamespace

    from lelab import record
    from lelab.chess.transfers import read_transfer_sidecar

    step = _overlay_session(monkeypatch)
    record._ensure_transfer()
    first = record.current_transfer
    assert first is not None and step.goal == first
    # A re-recorded take goes through reset and recording again with the same goal.
    record._ensure_transfer()
    record._ensure_transfer()
    assert record.current_transfer == first

    record._record_saved_transfer(SimpleNamespace(root=tmp_path, num_episodes=1))
    assert record.current_transfer is None and step.goal is None
    assert read_transfer_sidecar(tmp_path) == [{"episode_index": 0, "source": first[0], "target": first[1]}]


def test_saved_transfers_make_a_chess_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from types import SimpleNamespace

    from lelab import dataset_check, record
    from tests.test_dataset_check import make_dataset

    root = make_dataset(tmp_path / "d", episodes=2, sidecar=False)
    assert any("garage_transfers.json" in p for p in dataset_check.chess_problems(root))

    _overlay_session(monkeypatch)
    for saved in (1, 2):
        record._ensure_transfer()
        record._record_saved_transfer(SimpleNamespace(root=str(root), num_episodes=saved))
    assert dataset_check.chess_problems(root) == []


def test_saving_outside_chess_mode_writes_no_sidecar(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from types import SimpleNamespace

    from lelab import record

    monkeypatch.setattr(record, "_overlay_step", None)
    monkeypatch.setattr(record, "current_transfer", None)
    record._record_saved_transfer(SimpleNamespace(root=tmp_path, num_episodes=1))
    assert not (tmp_path / "meta").exists()


def test_top_camera_feed_shows_the_goal_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    import cv2
    import numpy as np

    from lelab import record
    from lelab.chess.overlay import SOURCE_RGB, TARGET_RGB
    from tests.chess_render import calibration

    step = _overlay_session(monkeypatch)
    step.set_goal("e2", "e4")
    frame = np.full((400, 400, 3), 120, dtype=np.uint8)
    camera = SimpleNamespace(read_latest=lambda: frame)
    robot = SimpleNamespace(cameras={"top": camera, "wrist": camera})
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "current_robot", robot)

    def decoded(cam_key: str) -> np.ndarray:
        frames = record.camera_feed_frames(cam_key, fps=1000.0)
        chunk = next(frames)
        frames.close()
        jpeg = chunk.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0]
        return cv2.cvtColor(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    cal = calibration()
    top = decoded("top")
    for square, rgb in (("e2", SOURCE_RGB), ("e4", TARGET_RGB)):
        cx, cy = (int(round(v)) for v in cal.square_center(square))
        assert np.abs(top[cy, cx].astype(int) - rgb).max() < 40
    wrist = decoded("wrist")
    assert np.abs(wrist.astype(int) - 120).max() < 10
    # The camera's own frame buffer is never painted on.
    assert (frame == 120).all()
