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

"""``lelab-camera``: read and set UVC camera controls (white balance, hue, ...) on macOS.

Examples::

    lelab-camera list
    lelab-camera set 0abd:8050 white_balance_temperature_auto=0 white_balance_temperature=2800 --save
    lelab-camera apply
    lelab-camera reset 0abd:8050
    lelab-camera gain "USB Camera" --blue 1.7868 --red 1.4464   # per-channel colour gain, see README
    lelab-camera gain                                             # show saved gains
    lelab-camera gain "USB Camera" --forget
    lelab-camera calibrate "USB Camera"      # white sheet in the middle of the picture; see lelab-camera-calibrate
    lelab-camera lens "USB Camera" --fov 120 --fit 1920x1080   # fisheye correction, see lelab-camera-lens
    lelab-camera lens                                           # show saved lenses

Exit codes: 0 done, 1 some saved camera could not be updated, 2 invalid request, 3 not available on this
system, 4 camera not attached, 5 the camera rejected a request, 6 the white sheet could not be measured.
"""

from __future__ import annotations

import argparse
import sys

from .. import camera_controls, color_calibration, color_gain, lens
from ..utils import uvc
from ..utils.camera_listing import CameraKeyError, list_cameras_in_cv2_order, resolve_camera_key, split_key

EXIT_OK, EXIT_PARTIAL, EXIT_INVALID, EXIT_UNAVAILABLE, EXIT_NOT_FOUND, EXIT_TRANSFER = 0, 1, 2, 3, 4, 5
EXIT_MEASUREMENT = 6


def _parse_pairs(pairs: list[str]) -> dict[str, int]:
    controls: dict[str, int] = {}
    for pair in pairs:
        name, sep, raw = pair.partition("=")
        if not sep:
            raise camera_controls.CameraControlsError(f"'{pair}' must look like name=value")
        try:
            controls[name] = int(raw)
        except ValueError:
            raise camera_controls.CameraControlsError(
                f"'{name}' must be a whole number, got '{raw}'"
            ) from None
    return controls


def _print_values(values: dict[str, int]) -> None:
    for name, value in values.items():
        print(f"  {name} = {value}")


def _list() -> int:
    cameras = camera_controls.list_cameras()
    if not cameras:
        print("No UVC cameras found.")
    for camera in cameras:
        print(f"{camera['id']}  {camera['name']}")
        for control in camera["controls"]:
            extra = f"default {control['default']}"
            if control["kind"] == "range":
                extra += f", {control['min']}..{control['max']}"
            print(f"  {control['name']} = {control['value']}  ({extra})")
        if camera["saved"]:
            print(f"  saved: {camera['saved']}")
        if "error" in camera:
            print(f"  error: {camera['error']}")
    return EXIT_OK


def _apply() -> int:
    results = camera_controls.apply_saved()
    if not results:
        print("Nothing saved.")
    partial = False
    for ident, result in results.items():
        if isinstance(result, str):
            partial = True
            print(f"{ident}: not updated: {result}")
        else:
            print(ident)
            _print_values(result)
    return EXIT_PARTIAL if partial else EXIT_OK


def _key(wanted: str, cameras: list[dict] | None = None) -> str:
    """The key a setting is saved under for a camera given by name, index or key (see `resolve_camera_key`)."""
    return resolve_camera_key(wanted, list_cameras_in_cv2_order() if cameras is None else cameras)


def _forget(forget, key: str, cameras: list[dict]) -> bool:
    """Forget the setting under ``key``.

    A setting saved under the bare name (by an older version) goes too, but only when exactly one attached camera
    has that name: with two, the old entry is not clearly this camera's and is left alone.
    """
    if forget(key):
        return True
    name, unique_id = split_key(key)
    sharing = [c for c in cameras if str(c.get("name", "")).strip() == name]
    return unique_id is not None and len(sharing) == 1 and forget(name)


def _gain(args: argparse.Namespace) -> int:
    if args.name is None:
        if args.blue is not None or args.red is not None or args.forget:
            raise color_gain.ColorGainError(
                "Name the camera to change its gain, e.g. lelab-camera gain 'USB Camera' --blue 1.79 --red 1.45"
            )
        saved = color_gain.load_saved()
        if not saved:
            print("No colour gains saved.")
        for name, gains in saved.items():
            print(f"{name}: blue={gains['blue']:.4f} green={gains['green']:.4f} red={gains['red']:.4f}")
        return EXIT_OK
    cameras = list_cameras_in_cv2_order()
    key = _key(args.name, cameras)
    if args.forget:
        if _forget(color_gain.forget_gains, key, cameras):
            print(f"Forgot the colour gain for '{key}'.")
        else:
            print(f"No colour gain saved for '{key}'.")
        return EXIT_OK
    if args.blue is None or args.red is None:
        raise color_gain.ColorGainError("Give --blue and --red (green defaults to 1.0)")
    gains = color_gain.save_gains(key, {"blue": args.blue, "green": args.green, "red": args.red})
    print(f"'{key}': blue={gains['blue']:.4f} green={gains['green']:.4f} red={gains['red']:.4f}")
    return EXIT_OK


def _format_lens(values: dict[str, float]) -> str:
    return f"fov={values['fov']:g}° bend={values['bend']:g} zoom={values['zoom']:g}"


def _lens(args: argparse.Namespace) -> int:
    changing = args.fov is not None or args.bend != 0.0 or args.zoom is not None or args.fit or args.forget
    if args.name is None:
        if changing:
            raise lens.LensError(
                "Name the camera to change its lens, e.g. lelab-camera lens 'USB Camera' --fov 120"
            )
        saved = lens.load_saved()
        if not saved:
            print("No lens corrections saved.")
        for name, values in saved.items():
            print(f"{name}: {_format_lens(values)}")
        return EXIT_OK
    cameras = list_cameras_in_cv2_order()
    name = lens.normalize_name(_key(args.name, cameras))
    if args.forget:
        forgot = _forget(lens.forget_lens, name, cameras)
        print(f"Forgot the lens for '{name}'." if forgot else f"No lens saved for '{name}'.")
        return EXIT_OK
    if args.fov is None:
        raise lens.LensError(
            "Give --fov, the lens's diagonal field of view in degrees (see lelab-camera-lens)"
        )
    if args.zoom is not None and args.fit:
        raise lens.LensError("Give either --zoom or --fit, not both")
    values = {"fov": args.fov, "bend": args.bend, "zoom": 1.0 if args.zoom is None else args.zoom}
    if args.fit:
        width, height = lens.parse_size(args.fit)
        values["zoom"] = lens.fit_zoom(lens.validate_lens(values), width, height)
    print(f"'{name}': {_format_lens(lens.save_lens(name, values))}")
    return EXIT_OK


def _parse_roi(text: str) -> color_calibration.Roi:
    parts = text.split(",")
    try:
        x, y, w, h = (int(part) for part in parts)
    except ValueError:
        raise camera_controls.CameraControlsError(
            f"--roi must look like x,y,width,height, got '{text}'"
        ) from None
    if w <= 0 or h <= 0:
        raise camera_controls.CameraControlsError("--roi width and height must be positive")
    return x, y, w, h


def _calibrate(args: argparse.Namespace) -> int:
    roi = _parse_roi(args.roi) if args.roi else None
    camera = color_calibration.find_camera(color_calibration.list_calibration_cameras(), args.camera)
    print(camera.label)
    white_balance = None if args.no_sweep else color_calibration.white_balance_for(camera)
    source = color_calibration.OpenCVFrameSource(camera.index)
    try:
        height, width = source.read().shape[:2]
        box = (
            color_calibration.clamp_roi(roi, width, height)
            if roi
            else color_calibration.default_roi(width, height)
        )
        print(f"  measuring the box x={box[0]} y={box[1]} {box[2]}x{box[3]} of a {width}x{height} picture")
        brightness = None if args.no_brightness else color_calibration.brightness_for(camera)
        calibrator = color_calibration.Calibrator(source, white_balance, brightness=brightness)
        result = calibrator.run(box, progress=lambda fraction, message: print(f"  {message}"))
    finally:
        source.close()
    for line in color_calibration.describe_result(result):
        print(f"  {line}")
    if args.dry_run:
        calibrator.restore()
        print("Not saved (--dry-run); the camera is back as it was.")
        return EXIT_OK
    calibrator.save(result, camera.key)
    print(f"Saved for '{camera.key}'. LeLab uses it from the next preview, recording or inference.")
    return EXIT_OK


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lelab-camera", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="Show UVC cameras, their controls and saved values.")
    set_parser = sub.add_parser("set", help="Set controls on one camera.")
    set_parser.add_argument(
        "device",
        help="USB id, e.g. 0abd:8050, or 0abd:8050@1130000 to name one of two identical cameras (see `list`).",
    )
    set_parser.add_argument("controls", nargs="+", metavar="NAME=VALUE")
    set_parser.add_argument("--save", action="store_true", help="Keep the values and re-apply them on start.")
    sub.add_parser("apply", help="Write the saved values to the cameras.")
    reset_parser = sub.add_parser("reset", help="Restore the camera's own defaults.")
    reset_parser.add_argument("device")
    reset_parser.add_argument("--keep-saved", action="store_true", help="Do not forget the saved values.")
    gain_parser = sub.add_parser(
        "gain", help="Show or set the per-channel colour gain of a camera, by its name (macOS)."
    )
    gain_parser.add_argument(
        "name",
        nargs="?",
        help='Camera name as the OS lists it ("USB Camera"), or its OpenCV index for identical names.',
    )
    gain_parser.add_argument("--blue", type=float, help="Multiplier for the blue channel.")
    gain_parser.add_argument("--green", type=float, default=1.0, help="Multiplier for green (default 1.0).")
    gain_parser.add_argument("--red", type=float, help="Multiplier for the red channel.")
    gain_parser.add_argument("--forget", action="store_true", help="Remove the gain for this camera.")
    calibrate_parser = sub.add_parser(
        "calibrate", help="Measure a white sheet and save white balance and colour gain (macOS)."
    )
    calibrate_parser.add_argument(
        "camera", nargs="?", help="Camera name or OpenCV index (default: the first USB camera)."
    )
    calibrate_parser.add_argument("--roi", help="Box to measure as x,y,width,height (default: the centre).")
    calibrate_parser.add_argument(
        "--no-sweep", action="store_true", help="Keep the camera's white balance; measure only the gain."
    )
    calibrate_parser.add_argument(
        "--no-brightness", action="store_true", help="Do not adjust the camera's brightness to the sheet."
    )
    calibrate_parser.add_argument("--dry-run", action="store_true", help="Measure and print, save nothing.")
    lens_parser = sub.add_parser(
        "lens", help="Show or set the fisheye lens correction of a camera, by its name (macOS)."
    )
    lens_parser.add_argument(
        "name",
        nargs="?",
        help='Camera name as the OS lists it ("USB Camera"), or its OpenCV index for identical names.',
    )
    lens_parser.add_argument(
        "--fov", type=float, help="Diagonal field of view of the lens in degrees (10 to 200)."
    )
    lens_parser.add_argument(
        "--bend", type=float, default=0.0, help="Fine curvature of the lens (default 0)."
    )
    lens_parser.add_argument(
        "--zoom", type=float, help="1 keeps the centre's scale, above 1 crops, below 1 adds black."
    )
    lens_parser.add_argument(
        "--fit", metavar="WIDTHxHEIGHT", help="Choose the zoom with no black borders for this picture size."
    )
    lens_parser.add_argument("--forget", action="store_true", help="Remove the lens for this camera.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "list":
            return _list()
        if args.command == "apply":
            return _apply()
        if args.command == "gain":
            return _gain(args)
        if args.command == "calibrate":
            return _calibrate(args)
        if args.command == "lens":
            return _lens(args)
        if args.command == "set":
            values = camera_controls.set_controls(args.device, _parse_pairs(args.controls), args.save)
        else:
            values = camera_controls.reset_controls(args.device, forget=not args.keep_saved)
        print(args.device)
        _print_values(values)
        return EXIT_OK
    except (
        camera_controls.CameraControlsError,
        uvc.UnsupportedControlError,
        color_gain.ColorGainError,
        lens.LensError,
        CameraKeyError,
    ) as exc:
        print(f"lelab-camera: {exc}", file=sys.stderr)
        return EXIT_INVALID
    except uvc.UvcUnavailableError as exc:
        print(f"lelab-camera: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except uvc.CameraNotFoundError as exc:
        print(f"lelab-camera: {exc}", file=sys.stderr)
        return EXIT_NOT_FOUND
    except uvc.UvcTransferError as exc:
        print(f"lelab-camera: the camera rejected the request: {exc}", file=sys.stderr)
        return EXIT_TRANSFER
    except color_calibration.CameraNotAttachedError as exc:
        print(f"lelab-camera: {exc}", file=sys.stderr)
        return EXIT_NOT_FOUND
    except color_calibration.CalibrationError as exc:
        print(f"lelab-camera: {exc}", file=sys.stderr)
        return EXIT_MEASUREMENT


if __name__ == "__main__":
    raise SystemExit(main())
