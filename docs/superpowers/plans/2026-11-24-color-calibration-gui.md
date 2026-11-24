# Automatic Colour Calibration Tool (GUI) — Implementation and Refactoring Plan

**Goal:** One standalone tool that calibrates a USB camera's colour automatically: it finds the white-balance temperature that comes closest to neutral, measures a white sheet, and saves the per-channel gain that LeLab already applies to previews, recordings and inference. A GUI for people (`lelab-camera-calibrate`) and the same engine headless for scripts (`lelab-camera calibrate`).

**Why:** Calibration today is three manual steps: set white balance with `lelab-camera set`, measure a white patch by hand, and type the gains into `lelab-camera gain`. The numbers are per room and per light, so they have to be redone whenever the lighting changes.

## Current state (what exists and is reused)

| Piece | Where | Reused as |
|---|---|---|
| Camera hardware controls over USB (white balance, ranges) | `lelab/utils/uvc.py`, `lelab/camera_controls.py` | Set manual WB, sweep temperatures, save the chosen one |
| Saved gains per camera name, applied to every frame | `lelab/color_gain.py` | `save_gains`, `apply_gains`, `validate_gains`, `MAX_GAIN` |
| Camera list in OpenCV index order, with names and AVFoundation unique ids | `lelab/utils/camera_listing.py` | Pick the camera; map it to its USB id |
| Calibration-page switch and previews | frontend | Unchanged; they read the saved gain |

The measuring itself only exists in throwaway scripts under `/tmp`. It becomes the tested core of this tool.

## Design

### Engine: `lelab/color_calibration.py` (no GUI code, fully testable)

- `CalibrationCamera(index, name, usb_id)`: a camera as listed, with `usb_id = (vid, pid)` when it is a USB camera. The AVFoundation unique id of a USB camera ends in `vvvvpppp` (`0x11300000abd8050` is `0abd:8050`), which links the OS camera to its USB controls without guessing by name.
- `default_roi(width, height)`: a centred box, 30% of each side, where the user holds the white sheet.
- `measure_patch(frames, roi) -> PatchStats`: median blue, green and red over the box across several frames, plus the share of clipped pixels and the spread of green. Medians ignore specks and glare.
- `neutrality(stats)`: `|log(R/G)| + |log(B/G)|`, which is 0 for a perfectly neutral white.
- `gains_from_patch(stats) -> (gains, warnings)`: green stays 1.0; blue = G/B, red = G/R. It refuses a patch that is too dark (green median below 50), mostly clipped (over 10% of pixels at 255), or that needs a gain above `MAX_GAIN`. It warns when the box is not uniform (the sheet does not fill it) or partly clipped.
- `Calibrator(capture, controls)`: drives one camera through two small interfaces, so tests use fakes:
  - `FrameSource.read() -> BGR frame` (OpenCV `VideoCapture` in real use),
  - `WhiteBalance` (`range()`, `set(kelvin)`, `save(kelvin)`) backed by `camera_controls`; `None` when the camera has no USB controls (built-in cameras), in which case the sweep is skipped and only the gain is measured.
- `Calibrator.run(roi, progress)`: switch auto WB off; try temperatures from the minimum to the maximum in 7 steps, letting each settle and measuring the patch; keep the most neutral and leave the camera on it; measure again and compute the gains; return a `CalibrationResult` with the temperature, gains, before and after stats and warnings. `progress(fraction, message)` feeds a progress bar, and an optional frame observer lets the GUI keep showing the picture while the sweep owns the camera.
- `Calibrator.save(result, name)`: saves both halves, the temperature in the camera-controls file and the gain with `color_gain.save_gains`. Nothing is saved before this, so a discarded calibration leaves the camera on an unsaved temperature that a replug or `lelab-camera apply` undoes.

### GUI: `lelab/scripts/color_calibration_gui.py` → `lelab-camera-calibrate`

Tk (bundled with Python, no new dependency). Frames go to Tk as PPM, which Tk reads natively.

- Camera menu (OS names, USB cameras first) and Refresh.
- Live preview at 640×480 with the measuring box drawn on it. Drag on the picture to move or redraw the box. A "Show corrected" switch previews the gain.
- Live readout under the picture: red/green and blue/green of the box, and the gain that would result now.
- **Auto calibrate** runs the sweep in a worker thread with a progress bar. The live preview pauses meanwhile, because the worker owns the camera.
- **Save** stores the result and shows where the values went. **Reset camera** restores the camera's defaults and forgets the saved gain. Problems (too dark, clipped, no USB controls) appear in the status line in plain words.

### Headless: `lelab-camera calibrate [CAMERA] [--roi x,y,w,h] [--no-sweep] [--dry-run]`

Same engine, prints the chosen temperature, gains and before/after neutrality. `CAMERA` is a name or an OpenCV index, defaulting to the first USB camera.

## Refactoring (before the feature)

1. **Make the camera listing public.** `_avfoundation_cameras_in_cv2_order` is imported by `server.py` and `color_gain.py` despite the leading underscore. Rename it `list_cameras_in_cv2_order`, and add `usb_id_from_unique_id`. Update both callers and the tests that monkeypatch it.
2. **Keep `color_gain` about storing and applying gains.** Measuring and choosing gains goes into the new `color_calibration` engine, not into `color_gain`, so neither module grows two jobs.
3. **One place for the white-balance controls the sweep uses.** The engine talks to a small `WhiteBalance` adapter over `camera_controls` (`set_controls`, `save`) instead of calling `uvc` directly, so the GUI, the CLI and the existing `lelab-camera set` share the same validation and saved file.
4. **The CLI delegates.** `lelab-camera calibrate` is a thin wrapper over the engine; the GUI is a thin view over the same engine. Neither contains measuring code.

## Tasks

1. Refactor the listing (step 1); run the existing tests. 
2. Engine pure functions with tests first: `usb_id_from_unique_id`, `default_roi`, `measure_patch`, `neutrality`, `gains_from_patch` and its refusals and warnings.
3. `Calibrator` with fakes: a fake camera whose image colour depends on the set temperature; test that the sweep picks the most neutral temperature, saves it, and returns near-neutral after-stats; test the no-USB-controls path.
4. Headless CLI subcommand with tests (exit codes as in `lelab-camera`: 2 invalid input, 3 unavailable, 4 camera not found).
5. GUI: view-model helpers tested without a display (box drag maths, frame-to-PPM conversion); the Tk window itself is checked by hand.
6. Hardware run on the demo Mac: headless calibration of the top camera, then the GUI, with a screenshot.
7. Docs: README "Camera colour on macOS" and the chess guide's setup step point to the tool first; the manual `gain` command stays for scripting.

## Out of scope

- Finding the white sheet automatically anywhere in the picture: unreliable on a wooden desk with screens in view, so the user places the box.
- Linux and Windows: the camera listing and the USB controls are macOS-only today.
- A calibration page inside the LeLab web app; the tool is standalone and the web app picks up the saved gain.
