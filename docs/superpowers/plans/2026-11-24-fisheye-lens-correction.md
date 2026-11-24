# Fisheye Lens Correction — Implementation and Refactoring Plan

**Goal:** Straighten the wide-angle distortion of a camera (the top camera of the SO-101 kit bends the table edges and the chess board) with a saved per-camera lens model, so the straight-line geometry the chess board homography, the empty-board reference and the policy rely on is actually straight. The correction is calibrated by hand in a small window (one live picture, a grid, three sliders), saved per camera *name* like the colour gain, switchable per camera on the calibration card, and applied in every place a frame is produced.

**Decisions made with the user:** manual calibration only (a slider with a straight-line grid overlay, no printed checkerboard); any camera, not just the top one; the same on/off switch as the colour correction.

## Why it has to be everywhere

The chess transfer mode clicks board corners on the saved reference frame; the empty-board reference is compared with live frames; the policy trains on recorded frames. A homography is exact only on a straight (rectilinear) picture, and the reference, the live view, the recording and inference must show identical pixels. So the same transform runs in four places, with one model:

| Where | How |
|---|---|
| Recording and inference | Python, `cv2.remap` in `OpenCVCamera._postprocess_image` (the colour-gain hook) |
| Inference subprocess | the same maps, passed through an environment variable like the gains |
| Browser live preview (calibration card, recording dialog, ghost alignment) | a WebGL fragment shader on a canvas fed by the `<video>` |
| Browser reference capture | a CPU remap of the captured frame (one frame, exact same formula) |

## The lens model

Equidistant fisheye with one correction term, in coordinates that do not depend on the resolution: a pixel's position is measured from the picture centre in units of the half-diagonal `R = hypot(w, h) / 2`, so a model calibrated at 1920×1080 also fits 960×540 of the same aspect ratio. y points down.

Parameters, saved as `{"fov": degrees, "bend": number, "zoom": number}`:

- `fov`: the diagonal field of view of the lens in degrees (10–200). `θmax = fov/2` is the ray angle at the picture corner.
- `bend`: a fine-tune of the lens's projection curve, `θd = θ·(1 + bend·θ²)` (the first term of OpenCV's fisheye model); 0 is a pure equidistant lens.
- `zoom`: 1 keeps the centre's scale; above 1 magnifies (crops the edges), below 1 shows more and leaves black corners.

For every output pixel at offset `o` (in `R` units, `ρ = |o|`):

```
θdmax = θmax·(1 + bend·θmax²)           # the corner, by definition source radius 1
θ     = atan(ρ·θdmax / zoom)            # the ray angle the straightened picture shows at ρ
rs    = θ·(1 + bend·θ²) / θdmax         # where that ray sits in the camera picture
source = centre + o/ρ · rs · R           # sample here (bilinear), black if outside
```

A tiny `fov` is the identity. The lens is rejected when the projection curve folds back (`1 + 3·bend·θmax² ≤ 0`). `fit_zoom(lens, width, height)` is the smallest zoom (the widest view) with no black borders. *Implementation finding:* the black appears first at the middle of the shorter edges, not at the corners, because the correction stretches the picture outward; so the answer depends on the aspect ratio and is found by bisection rather than by a corner formula.

The formula exists three times (Python, GLSL, TypeScript) and is pinned by shared numeric fixtures, so they cannot drift apart silently.

## Current state (reused)

| Piece | Where | Reused as |
|---|---|---|
| Frame hook on `OpenCVCamera._postprocess_image` | `lelab/color_gain.py` `install` | moves into a shared module that runs lens, then colour |
| Name → OpenCV index map, session on/off flag, env-var hand-off | `color_gain.index_gains`, `gains_for_session`, `encode_for_env` | `index_gains` is generic (name → dict of numbers); the lens code calls it |
| Rollout wrapper | `lelab/scripts/rollout_gain.py` | also installs lenses |
| Frontend name matching, `useCameraGain`, switch, reference capture | `frontend/src/lib/cameraGain.ts` etc. | generalised, lens added next to the gain |
| Tk helpers (`display_scale`, `to_ppm`) | `lelab/scripts/color_calibration_gui.py` | move to `lelab/utils/tk_preview.py`, used by both windows |

## Refactoring (before the feature)

1. **`lelab/camera_frames.py`: one hook, ordered stages.** Today `color_gain.install` wraps `_postprocess_image` and owns the "already wrapped" flag. A second correction cannot also wrap it without two modules fighting over that flag. The new module holds per-stage `{camera index → function}` maps and one wrapper that applies `lens`, then `colour`, then LeRobot's own postprocessing. `color_gain.install(index_map)` keeps its signature and tests and registers the `colour` stage.
2. **Share the Tk preview helpers.** `display_scale` and `to_ppm` move to `lelab/utils/tk_preview.py`; the colour window and its tests import them from there.
3. **Frontend: one name matcher and one hook.** `gainForCamera`'s matching (normalise, exact, longest prefix before a space or "(") becomes `valueForCamera` in `lib/cameraName.ts`; `gainForCamera` stays as a thin typed wrapper. `useCameraGain` and the new `useCameraLens` share one `useSavedCameraSetting` hook.
4. **Reference capture takes options.** `reference.save(video, applyGain)` becomes `save(video, { gain, lens })` so the card passes both switches.

## Design

### Backend

- `lelab/lens.py` (no GUI, no hardware): `LensError`, `Lens = dict[str, float]`, `validate_lens`; `load_saved` / `save_lens` / `forget_lens` on `~/.cache/huggingface/lerobot/camera_lens.json` (atomic write like the gains); `fit_zoom(lens, width, height)`; `parse_size`; `source_coordinates(lens, width, height)` returning the sampling positions; cached fixed-point `cv2.remap` maps per size and lens; `apply_lens(image, lens)`; `lenses_for_session(cameras)` honouring `lens_correction: false`; `install` / `encode_for_env` / `decode_from_env` with `LELAB_CAMERA_LENSES`.
- `record.py` installs both maps before the cameras open; `rollout.py` puts both in the subprocess environment; `rollout_gain` installs both.
- `camera_config.lerobot_camera_settings` drops `lens_correction` like `color_correction`.
- `GET /camera-lenses` → `{"status": "success", "lenses": {...}}` for the browser.
- CLI: `lelab-camera lens [NAME] [--fov F] [--bend B] [--zoom Z | --fit WIDTHxHEIGHT | --forget]` (no name lists the saved lenses; a lens needs at least `--fov`).

### Window: `lelab-camera-lens` (`lelab/scripts/camera_lens_gui.py`)

- Camera menu (any camera), `--size WxH` to calibrate at the resolution you record at.
- Live corrected picture with a straight-line grid over it and a **Show original** switch. The instruction is one sentence: make the table edges and board lines run along the grid lines.
- Sliders: Field of view, Fine curvature, Zoom; **Fit** (no black corners); **Save** and **Reset** (forget). A folded lens shows an error and keeps the last valid one.
- The maps are built at the display size, so dragging a slider stays smooth.

### Browser

- `lib/cameraLens.ts`: `CameraLens` type, `getCameraLenses`, `lensForCamera`, `lensSource` (the formula), `undistortPixels` (bilinear CPU remap for the reference frame).
- `lib/lensRenderer.ts`: the WebGL renderer (shader built from the same formula) with `createLensRenderer(canvas)`, which returns `null` when WebGL is missing.
- `components/control/LensVideoCanvas.tsx`: draws the `<video>` through the renderer every frame; the video stays in the DOM (hidden) so streaming, labels and reference capture are unchanged. Without WebGL the preview shows the raw video and says why.
- `BrowserCameraPreview` gets `lensCorrection` (default on) next to `colorCorrection`; the saved gain filter keeps working on the canvas.
- Calibration card: a **Lens correction** switch under the colour switch, with a status line (`On · 120° field of view`, `On · no saved lens for this camera`, `Off`). The reference frame is captured with the same lens when the switch is on.

## Tasks

1. Refactor 1 (`camera_frames`) and 2 (Tk helpers); the existing tests stay green.
2. `lens.py` with tests first: validation and folding, persistence, numeric fixtures, black borders and `fit_zoom`, and the synthetic-grid test (a plane pattern rendered through an equidistant fisheye comes out as the plane pattern).
3. Wiring: recording, rollout and wrapper, `lens_correction` flag, `GET /camera-lenses`, `lelab-camera lens`; tests for each.
4. Window with tested helpers (slider values to a lens, grid lines); hand check with the real camera.
5. Frontend refactors 3 and 4, then `cameraLens.ts` with the shared numeric fixtures, the renderer, the canvas, the preview prop, the card switch and the reference capture; component tests with the renderer mocked.
6. Verify the shader against Python in Chrome (the same image through both, compared numerically), then the whole path on the real camera: calibrate, record a frame, compare with the browser reference.
7. Docs: README section "Fisheye lens", the chess guide setup step.

## Result and verification

- Backend: 66 lens/gain/CLI tests, including the synthetic plane through an equidistant fisheye, and the shared fixtures.
- Frontend: 187 tests; the TypeScript `undistortPixels` reproduces the server's `cv2.remap` on a test image within 2 levels.
- The WebGL shader was run in Chrome on a test image with hard checker edges and compared with `lelab.lens.apply_lens`: on every pixel that samples fully inside the picture the largest difference is 4 levels (mean 0.32, GPU 8-bit weights at the edges). At the very rim, within half a pixel of the picture's edge, OpenCV blends towards black and the GPU clamps to the edge pixel; the rim is not used for anything.
- Real camera: the top camera's table edges straighten at about 135° (4:3 and 16:9 need different fit zooms).

## Known limits and out of scope

- The model assumes the picture centre is the lens centre and that the capture keeps the full sensor aspect ratio. A resolution with a different aspect ratio (a cropped 4:3 mode of a 16:9 sensor) needs its own lens; calibrate at the size you record at (`--size`).
- Manual calibration is as good as the eye: the straight-line grid is the only feedback. An automatic checkerboard calibration (OpenCV's fisheye calibration) was left out on purpose and can later write the same three numbers.
- macOS only for applying the saved lens in recording and inference, like the colour gain (it needs the camera name → index listing).
- No change to training data already recorded: those frames keep the geometry they were recorded with.
