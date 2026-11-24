# UVC Camera Controls (macOS) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let LeLab set the white balance (and the other UVC image controls) of a USB camera on macOS, where OpenCV cannot, so the Waveshare IMX335 top camera stops producing a blue-less yellow image.

**Architecture:** A small transport in `lelab/utils/uvc.py` sends USB Video Class control requests to the camera's VideoControl interface through libusb (`ctypes` over the `libusb-package` wheel). It never claims the interface, so macOS keeps streaming video to OpenCV and the browser. `lelab/camera_controls.py` persists the chosen values per USB id and re-applies them on server start and whenever the camera list is read. A `lelab-camera` command and three HTTP endpoints expose it.

**Tech Stack:** Python 3.12, `ctypes`, `libusb-package` (macOS only), FastAPI, pytest.

## Findings that shape the design (measured 2026-11-23 on the demo Mac)

- The top camera is USB `0abd:8050` ("USB Camera", Waveshare IMX335 (B), 175°). Its processing unit advertises white-balance temperature (2800–6500 K, step 10), auto white balance, hue, saturation, gamma, brightness, contrast, sharpness, backlight compensation and power-line frequency. It does **not** advertise red/blue gain.
- Out of the box: auto white balance on, mean RGB 124/102/8, median blue 5, B/R 0.06.
- Auto white balance off + 2800 K, over the same scene: mean RGB 88/108/26, B/R 0.29. Whites go from deep yellow to a light green-yellow. 2800 K is the floor of the control.
- Hue only rotates the remaining cast. Measured white points on the cabinet and paper trace a circle of roughly constant radius around neutral (R/G, B/G: −110 → 1.54/0.82, −60 → 1.04/0.46, −10 → 0.68/0.48, 40 → 0.48/0.70, 90 → 0.53/1.12), so no hue value cancels it. Saturation 80 makes it worse. Keep hue at the camera default (−10).
- Settings persist while the camera is powered and across `VideoCapture` open/close, so they only need re-applying after a replug.
- Setting them needs no root and works while another app streams the camera. pyusb is not usable: it claims the interface before every interface-recipient request and macOS refuses (`Access denied`). libusb's `libusb_control_transfer` does not need the claim.
- `libusb_close`/`libusb_exit` segfault if `argtypes` is not declared (64-bit handle truncated). Every libusb function the module calls must declare `argtypes` and `restype`.

## Global Constraints

- macOS only for the transport. On other platforms every public function raises `UvcUnavailableError`; Linux users set white balance through V4L2 (`v4l2-ctl`).
- No new dependency except `libusb-package>=1.0.26 ; sys_platform == 'darwin'`. Run `uv lock` after editing `pyproject.toml`.
- Only open USB devices whose device class is `0xEF` (miscellaneous / interface association), so listing never touches the arms' serial adapters or hubs.
- Never claim an interface, never reset a device, never detach a kernel driver.
- State lives in `~/.cache/huggingface/lerobot/camera_controls.json`; import the path from `lelab/utils/config.py`.
- Out of scope: a frontend UI for the controls, Linux and Windows transports, any per-frame software recolouring.

## File Structure

- Create `lelab/utils/uvc.py`: control table, descriptor parser, `UvcCamera`, libusb transport, `list_cameras`, `open_camera`.
- Create `lelab/camera_controls.py`: persistence and the apply/set/reset operations used by the server and the CLI.
- Create `lelab/scripts/camera_controls.py`: `lelab-camera` command.
- Modify `lelab/utils/config.py`: `CAMERA_CONTROLS_FILE`.
- Modify `lelab/server.py`: three endpoints, apply on startup, apply when cameras are listed on macOS.
- Modify `pyproject.toml`, `uv.lock`: dependency and script entry.
- Create `tests/uvc_fixtures.py` (real descriptors, already captured), `tests/test_uvc.py`, `tests/test_camera_controls.py`, `tests/test_uvc_live.py`.
- Modify `README.md`, `CHESS_DEMO_GUIDE.md`: short notes.

---

### Task 1: UVC control layer

**Files:**
- Create: `lelab/utils/uvc.py`
- Test: `tests/test_uvc.py`, fixtures in `tests/uvc_fixtures.py`

**Interfaces:**
- Produces:
  - `class UvcUnavailableError(RuntimeError)`, `class UnsupportedControlError(ValueError)`.
  - `CONTROLS: dict[str, Control]`; `GATES: dict[str, int]` mapping an auto control to the value that selects manual mode (`white_balance_temperature_auto: 0`, `exposure_auto: 1`).
  - `parse_video_control(config: bytes) -> VideoControlInfo | None`.
  - `class UvcCamera(transport, info, vid, pid, name)` with `supported() -> list[str]`, `describe(name) -> dict`, `read(name) -> int`, `write(name, value) -> int`, `apply(controls) -> dict[str, int]`, `reset() -> dict[str, int]`.
  - `list_cameras() -> list[UvcCameraInfo]`, `open_camera(vid, pid) -> UvcCamera` (context manager).

- [x] **Step 1: Write the failing tests** in `tests/test_uvc.py`: descriptor parsing against both real fixtures (IMX335: VC interface 0, processing unit 2, camera terminal 1, processing-unit bitmap `7f15`; Innomaker: bitmap `7f17`); `supported()` lists white balance for the IMX335 and nothing outside the bitmap; encode/decode of a signed 16-bit hue (−10); range clamping to the camera's min/max; `apply({"white_balance_temperature": 2800, "white_balance_temperature_auto": 0})` writes the auto control **before** the temperature; unsupported name raises `UnsupportedControlError`; `reset()` drops gates to manual, restores defaults, then restores the gates' defaults.
- [x] **Step 2: Run to see them fail** — `uv run pytest tests/test_uvc.py -q` → `ModuleNotFoundError: lelab.utils.uvc`.
- [x] **Step 3: Implement** the control table (UVC 1.5, tables 4-7/4-8 camera terminal and 4-13/4-14 processing unit), the descriptor walk, `UvcCamera` over an injected transport, then the libusb transport with `argtypes` on every function.
- [x] **Step 4: Run to see them pass** — `uv run pytest tests/test_uvc.py -q`.

### Task 2: Persistence and apply

**Files:**
- Create: `lelab/camera_controls.py`
- Modify: `lelab/utils/config.py` (add `CAMERA_CONTROLS_FILE = os.path.join(os.path.expanduser("~/.cache/huggingface/lerobot"), "camera_controls.json")`)
- Test: `tests/test_camera_controls.py`

**Interfaces:**
- Consumes: Task 1's `list_cameras`, `open_camera`, `UvcUnavailableError`, `UnsupportedControlError`, `CONTROLS`.
- Produces: `device_id(vid, pid) -> str` (`"0abd:8050"`), `parse_device_id(text) -> tuple[int, int]`, `load_saved() -> dict[str, dict[str, int]]`, `list_cameras() -> list[dict]`, `set_controls(device, controls, save) -> dict[str, int]`, `apply_saved() -> dict[str, dict | str]`, `reset_controls(device, forget) -> dict[str, int]`, `class CameraControlsError(ValueError)`.

- [x] **Step 1: Failing tests**: saved file round-trips and merges per device; invalid id, unknown control name and non-integer value raise `CameraControlsError`; `apply_saved()` applies each saved device and reports a missing camera as a message instead of raising; with no saved file it returns `{}` without touching USB; a failure on one camera does not stop the next.
- [x] **Step 2: Run to see them fail.**
- [x] **Step 3: Implement** with an atomic `tmp` + `os.replace` write, the path read from `lelab.utils.config` at call time so tests can redirect it.
- [x] **Step 4: Run to see them pass** — `uv run pytest tests/test_camera_controls.py -q`.

### Task 3: Server endpoints and hooks

**Files:**
- Modify: `lelab/server.py` (next to the camera-reference routes; startup event; `get_available_cameras` macOS branch)
- Test: `tests/test_camera_controls.py` (endpoint tests, same file)

**Interfaces:**
- Produces: `GET /camera-controls` → `{"status": "success", "available": bool, "cameras": [...], "saved": {...}}`; `PUT /camera-controls/{device}` with `{"controls": {name: int}, "save": bool}` → applied values; `POST /camera-controls/apply-saved`. 400 for `CameraControlsError`, 503 for `UvcUnavailableError`.

- [x] **Step 1: Failing tests** with `TestClient` and a monkeypatched `lelab.camera_controls` backend: list, set without and with `save`, bad payload → 400, unavailable → 503 on PUT and `available: false` on GET.
- [x] **Step 2: Implement** the routes; call `apply_saved()` best-effort (never raise) from the startup event and from the macOS branch of `/available-cameras`.
- [x] **Step 3: Run** `uv run pytest tests/test_camera_controls.py tests/test_server.py -q`.

### Task 4: `lelab-camera` command

**Files:**
- Create: `lelab/scripts/camera_controls.py`
- Modify: `pyproject.toml` (`[project.scripts]` add `lelab-camera = "lelab.scripts.camera_controls:main"`; dependency `libusb-package>=1.0.26 ; sys_platform == 'darwin'`), `uv.lock`
- Test: `tests/test_camera_controls.py` (CLI section)

**Interfaces:**
- Produces: `main(argv: list[str] | None = None) -> int` with `list`, `set <id> name=value ... [--save]`, `apply`, `reset <id> [--keep-saved]`.

- [x] **Step 1: Failing tests** for argument parsing (`set 0abd:8050 white_balance_temperature=2800 --save`) and exit codes (0 ok, 1 partial, 2 invalid input, 3 unavailable, 4 camera not attached, 5 camera rejected the request).
- [x] **Step 2: Implement**, then `uv lock` and `uv sync`.
- [x] **Step 3: Run** `uv run pytest tests/test_camera_controls.py -q` and `uv run lelab-camera list`.

### Task 5: Documentation

**Files:**
- Modify: `README.md` (new "Camera colour on macOS" section after "Camera rotation"), `CHESS_DEMO_GUIDE.md` (setup step 2 note and one troubleshooting row)

- [x] **Step 1: Write** what the cast is, the command that fixes it, that it persists across restarts and needs re-applying after a replug only by LeLab itself, the 2800 K floor and the remaining green tint.
- [x] **Step 2: Check** the docs mention only commands that exist (`uv run lelab-camera ...`).

### Task 6: Live test on the hardware

**Files:**
- Create: `tests/test_uvc_live.py` (skipped unless `LELAB_UVC_LIVE=0abd:8050` is set)

- [x] **Step 1: Write** a test that opens the camera, records the current white-balance values, sets auto off and 3000 K, reads them back, restores the originals.
- [x] **Step 2: Run on the demo Mac** — `LELAB_UVC_LIVE=0abd:8050 uv run pytest tests/test_uvc_live.py -q`.
- [x] **Step 3: End-to-end**: `uv run lelab-camera set 0abd:8050 white_balance_temperature_auto=0 white_balance_temperature=2800 --save`, grab a 640×480 frame through OpenCV (`AVFoundation`), confirm B/R rises from about 0.06 to about 0.29 and the controls read back unchanged after the stream closes.
- [x] **Step 4: Full suite** — `uv run pytest -q` and `uv run ruff check lelab tests`.

## Result (2026-11-23, demo Mac)

All six tasks done. Top camera, 640×480 through OpenCV: defaults give B/R 0.06 (paper B/G 0.04, yellow); manual 2800 K gives B/R 0.30 (paper R/G 0.71, B/G 0.51, pale green). The values survive closing the stream, and after the camera is reset the server re-applies the saved file when the camera list is read. Full suite: 806 passed, 3 skipped. Not done: the lens undistort.

### Follow-up: fixed per-channel colour gain (same day)

Problem: 2800 K still leaves a green cast (paper R/G 0.70, B/G 0.54). Decision: a saved gain per camera name, applied to every frame, seeded with the measured values.

- `lelab/color_gain.py`: gains keyed by the OS camera name (`camera_gains.json`), mapped to the current OpenCV index through the AVFoundation listing (`lelab/utils/camera_listing.py`, moved from `server.py`). `install()` wraps `OpenCVCamera._postprocess_image`, which LeRobot runs on every frame, so recording, the live view and inference are all corrected.
- Inference runs `lelab.scripts.rollout_gain` (LeRobot's rollout after installing the map from `LELAB_CAMERA_GAINS`).
- The browser's chess reference capture applies the same gain, found by the stream's device label (`frontend/src/lib/cameraGain.ts`), so the reference matches the recorded frames.
- `lelab-camera gain "USB Camera" --blue 1.7868 --red 1.4464`; `GET /camera-gains`.

Measured through LeRobot's `OpenCVCamera` on the demo Mac: paper R/G 0.70 → 1.00, B/G 0.54 → 0.97. Browser previews (`BrowserCameraPreview`, used by calibration, bindings and the control feed) show the same correction as an SVG `feColorMatrix` filter on the `<video>`; the captured reference reads the raw pixels and applies the gain once in JavaScript. Verified in headless Chrome: raw paper R/G 0.72, B/G 0.54; filtered R/G 1.04, B/G 0.97; a canvas read of the filtered element returns the unfiltered pixels (no double correction).

## Self-Review

- Spec coverage: set white balance on macOS (Tasks 1, 4), persist and re-apply (2, 3), docs (5), test on hardware (6).
- Placeholder scan: none; each step names the exact tests, files and commands.
- Type consistency: `device_id` strings are `"vid:pid"` lowercase hex everywhere; control names are the keys of `CONTROLS`; values are `int`.
- Known limit, not hidden: at the camera's 2800 K floor a green-yellow tint remains. The camera has no red/blue gain control, so removing it needs a fixed per-channel gain in software, measured from a white reference. That is a separate decision after this lands.
