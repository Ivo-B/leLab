# Camera identity and exposure in the colour calibration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two USB cameras with the same name (and the same `vid:pid`) are calibrated, saved and applied independently, and the calibration tool fixes an overexposed white sheet itself instead of refusing it.

**Decisions (2026-11-25):** cameras are told apart by USB port (the AVFoundation `unique_id`); the calibration locks a manual exposure it finds automatically and saves it with the white balance.

## Findings

- **Name-only identity.** `color_gain` and `lens` stores key by camera name; `index_by_name` skips a name shared by two cameras (so neither is corrected); `find_camera` returns the first name match; the Tk dropdown shows the index but `Calibrator.save` stores under `camera.name`, so the second camera overwrites the first. In the browser, `valueForCamera` matches by label and `useAvailableCameras` pairs browser devices to OpenCV indices by label in order, which is a guess for identical labels.
- **vid:pid identity for USB controls.** `uvc.open_camera(vid, pid)` opens the first match, so the white-balance sweep can drive camera A while frames come from camera B. `camera_controls.json` has one slot per `vid:pid`.
- **No exposure handling.** `Calibrator` only sweeps white balance. `check_exposure` refuses more than 10% clipped pixels and says "lower the exposure", which the tool cannot do. The IMX335 advertises `exposure_auto` and `exposure_time_absolute` (`tests/test_uvc.py`), and `uvc.py` already writes them; no gain control.
- AVFoundation `unique_id` of a USB camera is `0x<location 7 hex><vid 4><pid 4>` (`usb_id_from_unique_id`), so two identical cameras on different ports differ in the location digits.

## Global constraints

- macOS only for the identity and USB parts, like the rest of the colour code.
- Old name-only and `vid:pid`-only saved files keep working; they are rewritten in the new form on the next save.
- A saved entry that cannot be matched to exactly one attached camera is skipped with a log line, never guessed.

## File structure

- Modify `lelab/utils/camera_listing.py`: `camera_key(name, unique_id)`, key parsing, resolution (`unique_id` match, then unique legacy name, else skip).
- Modify `lelab/utils/camera_store.py`, `lelab/color_gain.py`, `lelab/lens.py`: keys carry the id; `index_gains` and the lens equivalent use the new resolution.
- Modify `lelab/utils/uvc.py`: `location` on `UvcCameraInfo`; `open_camera(vid, pid, location=None)`.
- Modify `lelab/camera_controls.py`, `lelab/scripts/camera_controls.py`: `vid:pid@location` device ids; plain `vid:pid` still accepted.
- Modify `lelab/color_calibration.py`: `CalibrationCamera.camera_id`, exposure step in `Calibrator`.
- Modify `lelab/scripts/color_calibration_gui.py`: dropdown label shows the port, saves under the key, shows the chosen exposure.
- Modify `lelab/server.py`, `frontend/src/hooks/useCameraGain.ts`, `useCameraLens.ts`, `lib/cameraGain.ts`: gains per OpenCV index; optional server thumbnail to tell identical cameras apart.
- Tests: `tests/test_color_gain.py`, `test_color_calibration.py`, `test_camera_controls.py`, `test_uvc.py`, `test_lens.py`, frontend `cameraGain.test.ts`.
- Docs: `README.md` (camera colour section), `CHESS_DEMO_GUIDE.md`.

---

### Task 0: Spike on the demo Mac (no commit)

- [ ] **Step 1:** With both cameras plugged in, run `list_cameras_in_cv2_order()`; confirm the two `unique_id`s differ.
- [ ] **Step 2:** Declare `libusb_get_bus_number` / `libusb_get_port_numbers` (with `argtypes`) and print them per device; check they reproduce the location digits of the `unique_id` (`bus << 24 | port nibbles`, high port first).
- [ ] **Step 3:** If they do not match, stop and pick another link (`ioreg` registry id) before Task 2.

### Task 1: Camera identity in the stores

**Produces:** `camera_key(name, unique_id) -> str` (`"USB Camera [0x11300000abd8050]"`), `resolve(cameras, saved, noun) -> dict[int, V]`.

- [ ] **Step 1: Failing tests:** two cameras named "USB Camera" with different ids get different saved gains; a legacy key `"USB Camera"` applies when exactly one attached camera has that name; with two, it is skipped and logged; a saved id that is not attached is skipped; `save` for camera B leaves camera A's entry alone.
- [ ] **Step 2:** Implement in `camera_listing.py`; switch `color_gain.index_gains` and the lens module to it; `CameraStore.normalize` keeps trimming names inside the key.
- [ ] **Step 3:** `uv run pytest tests/test_color_gain.py tests/test_lens.py tests/test_camera_store.py -q`.

### Task 2: USB device selected by location

- [ ] **Step 1: Failing tests:** `open_camera(vid, pid, location)` opens the device at that location with a fake `_usb_devices`; without `location` and with two matches it raises `CameraNotFoundError("2 cameras ... give a location")`; `device_id` / `parse_device_id` round-trip `0abd:8050@1130000`; saved controls for two devices do not merge.
- [ ] **Step 2:** Implement `location` in `UvcCameraInfo`, `list_cameras`, `open_camera`; extend `camera_controls` ids; `lelab-camera list` prints the full id.
- [ ] **Step 3:** `CalibrationCamera.device` returns the id with location (from the `unique_id`), and `UvcWhiteBalance` uses it.

### Task 3: Exposure in the calibration

**Order in `Calibrator.run`:** lock manual exposure, fit exposure on the box, sweep white balance, re-check exposure, measure the gain, save.

- [ ] **Step 1: Failing tests** with a simulated camera whose frame brightness scales with `exposure_time_absolute`:
  - a clipped sheet converges to a green median in 170-215 with under 1% clipped pixels in at most 8 steps;
  - a dark sheet is raised the same way;
  - exposure stays inside the camera's min/max;
  - the exposure is re-checked after the sweep;
  - a camera without exposure control falls back to `brightness`, then to the old error text;
  - `save` stores `exposure_auto=1` and `exposure_time_absolute` together with the temperature.
- [ ] **Step 2:** Add an `Exposure` protocol next to `WhiteBalance` (`range`, `set`, `save`) and `UvcExposure`, both through `camera_controls`. `check_exposure` raises only after the fit has failed, with the final exposure in the message.
- [ ] **Step 3:** Result panel and CLI output show the chosen exposure; `reset_controls` restores auto exposure.

### Task 4: Window and CLI

- [ ] **Step 1: Failing tests** (`tests/test_color_calibration_gui.py`): dropdown labels differ for two identical cameras; Save passes the camera key; the result text includes exposure.
- [ ] **Step 2:** Implement; `find_camera` accepts an index, a key, or a name that is unique, and errors with the list when a name matches several.

### Task 5: Browser

- [ ] **Step 1: Failing tests** (`cameraGain.test.ts`): gains are looked up by OpenCV index; a label shared by two cameras does not pick the wrong gain.
- [ ] **Step 2:** `/camera-gains` also returns `by_index`; `useCameraGain` uses the camera's index; the preview card shows a server thumbnail of the OpenCV index so identical cameras can be told apart.
- [ ] **Step 3:** `npm test` in `frontend/`.

### Task 6: Docs and verification

- [ ] **Step 1:** Update README and CHESS_DEMO_GUIDE: identity by port (a camera moved to another port is calibrated again), the exposure lock and when to redo it.
- [ ] **Step 2:** On the demo Mac with both cameras: calibrate each, check `camera_gains.json` and `camera_controls.json` hold two entries, read the controls back per camera, record a few frames from each and compare colour.
- [ ] **Step 3:** `uv run pytest -q`, `uv run ruff check lelab tests`, `graphify update .`.

## Known limits

- Moving a camera to another USB port changes its key; it has to be calibrated again.
- A locked exposure assumes steady light; recalibrate when the lighting changes.
- The browser can only tell identical cameras apart through the thumbnail check, because Chrome hides the AVFoundation id.
