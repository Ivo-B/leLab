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

"""``lelab-camera-lens``: a window that straightens a wide-angle camera by eye.

Pick the camera and look at something straight in the scene, such as the edge of the table or the chess
board. Drag a line along that edge on the picture and Field of view is set so the edge comes out straight;
or raise Field of view by hand until the edge lies along the grid. Nudge Curvature if the ends still bow, and
press Fit to get rid of black borders. Save keeps the lens under the camera's name and USB id, the same
key as the colour gain, so two cameras that share a name each keep their own; LeLab then applies it to
previews, recordings and inference. Calibrate at the size you record at (``--size``): a lens fits pictures of
the same aspect ratio only.
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from collections.abc import Sequence
from tkinter import ttk

import cv2
import numpy as np

from .. import color_calibration, lens
from ..color_calibration import CalibrationCamera
from ..utils.camera_listing import index_by_key
from ..utils.tk_preview import display_scale, to_ppm

TICK_MS = 40
GRID_DIVISIONS = 8
# Slider ranges. Bend is allowed less room than ``lens`` accepts: past these values it is hard to steer by hand.
SLIDER_RANGES = {"fov": lens.FOV_RANGE, "bend": (-0.1, 0.3), "zoom": (0.3, 2.0)}
DEFAULTS: lens.Lens = {"fov": lens.FOV_RANGE[0], "bend": 0.0, "zoom": 1.0}


# -- pure helpers, tested without a display -----------------------------------------------------------


def grid_lines(
    width: int, height: int, divisions: int = GRID_DIVISIONS
) -> list[tuple[float, float, float, float]]:
    """Straight reference lines as ``(x0, y0, x1, y1)``: ``divisions`` columns and rows, edges left out."""
    vertical = [
        (width * i / divisions, 0.0, width * i / divisions, float(height)) for i in range(1, divisions)
    ]
    horizontal = [
        (0.0, height * i / divisions, float(width), height * i / divisions) for i in range(1, divisions)
    ]
    return vertical + horizontal


def lens_or_message(values: dict[str, float]) -> tuple[lens.Lens | None, str | None]:
    """The checked lens for the slider values, or a message when they are not a usable lens."""
    try:
        return lens.validate_lens(values), None
    except lens.LensError as exc:
        return None, str(exc)


def describe(values: lens.Lens) -> str:
    return f"{values['fov']:.0f}° field of view · bend {values['bend']:+.3f} · zoom {values['zoom']:.2f}"


def _listing(cameras: Sequence[CalibrationCamera]) -> list[dict[str, object]]:
    return [{"index": camera.index, "name": camera.name, "unique_id": camera.unique_id} for camera in cameras]


def saved_lens_for(camera: CalibrationCamera, cameras: Sequence[CalibrationCamera]) -> lens.Lens | None:
    """The lens saved for this camera: under its key, or under its name when no other camera has it."""
    return index_by_key(_listing(cameras), lens.load_saved(), "Lens").get(camera.index)


def save_lens_for(camera: CalibrationCamera, values: lens.Lens) -> lens.Lens:
    """Keep this lens under the camera's key, so a shared name does not overwrite the other camera."""
    return lens.save_lens(camera.key, values)


def forget_lens_for(camera: CalibrationCamera, cameras: Sequence[CalibrationCamera]) -> bool:
    """Drop the lens saved for this camera, including a name-only entry that applied only to it."""
    forgotten = lens.forget_lens(camera.key)
    if sum(other.name == camera.name for other in cameras) == 1:
        forgotten = lens.forget_lens(camera.name) or forgotten
    return forgotten


def straighten_for_display(frame: np.ndarray, size: tuple[int, int], values: lens.Lens | None) -> np.ndarray:
    """The frame scaled to the window and straightened there. The model does not depend on the resolution."""
    small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return lens.apply_lens(small, values) if values is not None else small


# -- the window ---------------------------------------------------------------------------------------


class LensApp:
    def __init__(
        self,
        root: tk.Tk,
        cameras: list[CalibrationCamera],
        initial: CalibrationCamera,
        size: tuple[int, int] | None = None,
    ) -> None:
        self.root = root
        self.cameras = cameras
        self.size = size
        self.camera: CalibrationCamera | None = None
        self.source: color_calibration.OpenCVFrameSource | None = None
        self.frame: np.ndarray | None = None
        self.display = (640, 360)
        self.photo: tk.PhotoImage | None = None
        self.last_good: lens.Lens = dict(DEFAULTS)
        self.variables = {name: tk.DoubleVar(value=DEFAULTS[name]) for name in lens.PARAMS}
        self.value_labels: dict[str, ttk.Label] = {}
        self.stroke: list[tuple[float, float]] | None = None  # the line being dragged, in display pixels

        root.title("LeLab camera lens correction")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._build()
        self._select(initial)
        self.root.after(TICK_MS, self._tick)

    def _build(self) -> None:
        pad = {"padx": 8, "pady": 4}
        top = ttk.Frame(self.root)
        top.pack(fill="x", **pad)
        ttk.Label(top, text="Camera").pack(side="left")
        self.camera_menu = ttk.Combobox(top, state="readonly", width=48)
        self.camera_menu.pack(side="left", padx=6)
        self.camera_menu.bind("<<ComboboxSelected>>", self._on_camera_chosen)
        ttk.Button(top, text="Refresh", command=self._refresh).pack(side="left")
        self._fill_camera_menu()

        ttk.Label(
            self.root,
            text="Look at something that is really straight, like the table edge or the board. Drag along it on the "
            "picture, or raise Field of view until it lies along the grid.",
        ).pack(anchor="w", **pad)

        self.canvas = tk.Canvas(self.root, width=640, height=360, bg="black", highlightthickness=0)
        self.canvas.pack(**pad)
        self.image_item = self.canvas.create_image(0, 0, anchor="nw")
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        sliders = ttk.Frame(self.root)
        sliders.pack(fill="x", **pad)
        titles = {"fov": "Field of view (°)", "bend": "Curvature", "zoom": "Zoom"}
        for row, name in enumerate(lens.PARAMS):
            ttk.Label(sliders, text=titles[name], width=18).grid(row=row, column=0, sticky="w")
            low, high = SLIDER_RANGES[name]
            ttk.Scale(sliders, from_=low, to=high, variable=self.variables[name], length=420).grid(
                row=row, column=1, padx=6
            )
            self.value_labels[name] = ttk.Label(sliders, text="", width=8)
            self.value_labels[name].grid(row=row, column=2)

        options = ttk.Frame(self.root)
        options.pack(fill="x", **pad)
        self.show_grid = tk.BooleanVar(value=True)
        self.show_original = tk.BooleanVar(value=False)
        ttk.Checkbutton(options, text="Show grid", variable=self.show_grid, command=self._draw_grid).pack(
            side="left"
        )
        ttk.Checkbutton(options, text="Show original", variable=self.show_original).pack(side="left", padx=12)

        buttons = ttk.Frame(self.root)
        buttons.pack(fill="x", **pad)
        ttk.Button(buttons, text="Fit (no black borders)", command=self._fit).pack(side="left")
        self.save_button = ttk.Button(buttons, text="Save", command=self._save)
        self.save_button.pack(side="left", padx=6)
        ttk.Button(buttons, text="Forget saved lens", command=self._forget).pack(side="left")

        self.status_label = ttk.Label(self.root, text="", wraplength=780)
        self.status_label.pack(anchor="w", padx=8, pady=(0, 8))

    # -- cameras --

    def _fill_camera_menu(self) -> None:
        self.camera_menu["values"] = [camera.label for camera in self.cameras]

    def _on_camera_chosen(self, _event: object = None) -> None:
        index = self.camera_menu.current()
        if 0 <= index < len(self.cameras) and self.cameras[index] != self.camera:
            self._select(self.cameras[index])

    def _refresh(self) -> None:
        self.cameras = color_calibration.list_calibration_cameras()
        self._fill_camera_menu()
        if self.camera in self.cameras:
            self.camera_menu.current(self.cameras.index(self.camera))
        elif self.cameras:
            self._select(self.cameras[0])
        self._status(f"Found {len(self.cameras)} camera(s).")

    def _select(self, camera: CalibrationCamera) -> None:
        if self.source is not None:
            self.source.close()
        self.camera, self.source, self.frame = camera, None, None
        self.camera_menu.current(self.cameras.index(camera))
        saved = saved_lens_for(camera, self.cameras)
        self._set_values(saved or DEFAULTS)
        try:
            self.source = color_calibration.OpenCVFrameSource(camera.index, self.size)
        except color_calibration.CalibrationError as exc:
            self._status(str(exc), error=True)
            return
        self._status(f"Showing {camera.label} ({'with its saved lens' if saved else 'no saved lens yet'}).")

    # -- measuring from a drawn line --

    def _on_press(self, event: tk.Event) -> None:
        if self.frame is not None:
            self.stroke = [(float(event.x), float(event.y))]

    def _on_drag(self, event: tk.Event) -> None:
        if self.stroke is None:
            return
        self.stroke.append((float(event.x), float(event.y)))
        self.canvas.delete("stroke")
        if len(self.stroke) >= 2:
            flat = [coordinate for point in self.stroke for coordinate in point]
            self.canvas.create_line(*flat, fill="#ffd400", width=2, tags="stroke")

    def _on_release(self, _event: tk.Event) -> None:
        stroke, self.stroke = self.stroke, None
        self.canvas.delete("stroke")
        if stroke is None:
            return
        try:
            fov = lens.fit_fov_to_line(stroke, *self.display, bend=self.variables["bend"].get())
        except lens.LensError as exc:
            self._status(str(exc), error=True)
            return
        self.variables["fov"].set(fov)
        self._fit()
        self._status(
            f"Field of view set to {fov:.1f}° from the line you drew. Check the grid, then Save. "
            "Nudge Curvature if the ends still bow."
        )

    # -- values --

    def _set_values(self, values: lens.Lens) -> None:
        for name in lens.PARAMS:
            self.variables[name].set(values[name])
        self.last_good = dict(values)

    def _current_values(self) -> dict[str, float]:
        return {name: round(self.variables[name].get(), 4) for name in lens.PARAMS}

    def _fit(self) -> None:
        if self.frame is None:
            return
        height, width = self.frame.shape[:2]
        values, message = lens_or_message(self._current_values())
        if values is None:
            self._status(message or "", error=True)
            return
        self.variables["zoom"].set(lens.fit_zoom(values, width, height))
        self._status("Zoom set to the widest view with no black borders.")

    def _save(self) -> None:
        values, message = lens_or_message(self._current_values())
        if values is None or self.camera is None:
            self._status(message or "No camera", error=True)
            return
        try:
            save_lens_for(self.camera, values)
        except OSError as exc:
            self._status(f"Could not save: {exc}", error=True)
            return
        self._status(
            f"Saved for '{self.camera.key}': {describe(values)}. LeLab uses it from the next preview, recording "
            "or inference (reload an open browser page). Record the chess reference again afterwards."
        )

    def _forget(self) -> None:
        if self.camera is None:
            return
        forgotten = forget_lens_for(self.camera, self.cameras)
        self._set_values(DEFAULTS)
        self._status(f"Forgot the lens for '{self.camera.key}'." if forgotten else "No lens was saved.")

    # -- frames --

    def _tick(self) -> None:
        if self.source is not None:
            try:
                self.frame = self.source.read()
            except color_calibration.CalibrationError as exc:
                self._status(str(exc), error=True)
        if self.frame is not None:
            self._show(self.frame)
        self.root.after(TICK_MS, self._tick)

    def _show(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        scale = display_scale(width, height)
        size = (round(width * scale), round(height * scale))
        if size != self.display:
            self.display = size
            self.canvas.configure(width=size[0], height=size[1])
            self._draw_grid()
        values, message = lens_or_message(self._current_values())
        if values is not None:
            self.last_good = values
        elif message:
            self._status(message, error=True)
        original = self.show_original.get() or self.stroke is not None  # show the bow you are measuring
        shown = straighten_for_display(frame, size, None if original else self.last_good)
        self.photo = tk.PhotoImage(data=to_ppm(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)), format="PPM")
        self.canvas.itemconfigure(self.image_item, image=self.photo)
        for name in lens.PARAMS:
            self.value_labels[name].configure(text=f"{self.variables[name].get():.2f}")

    def _draw_grid(self) -> None:
        self.canvas.delete("grid")
        if self.show_grid.get():
            for x0, y0, x1, y1 in grid_lines(*self.display):
                self.canvas.create_line(x0, y0, x1, y1, fill="#00e5ff", width=1, tags="grid")

    def _status(self, text: str, error: bool = False) -> None:
        self.status_label.configure(text=text, foreground="#b42318" if error else "")

    def close(self) -> None:
        if self.source is not None:
            self.source.close()
        self.root.destroy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lelab-camera-lens", description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--camera", help="Camera name or OpenCV index to start with (default: first USB camera)."
    )
    parser.add_argument(
        "--size",
        type=lens.parse_size,
        metavar="WIDTHxHEIGHT",
        help="Capture size, the one you record at (default: the camera's own).",
    )
    args = parser.parse_args(argv)
    cameras = color_calibration.list_calibration_cameras()
    try:
        initial = color_calibration.find_camera(cameras, args.camera)
    except color_calibration.CameraNotAttachedError as exc:
        hint = "" if sys.platform == "darwin" else " The camera listing needs macOS."
        print(f"lelab-camera-lens: {exc}.{hint}", file=sys.stderr)
        return 4
    root = tk.Tk()
    LensApp(root, cameras, initial, args.size)
    root.lift()
    root.attributes("-topmost", True)
    root.after(500, lambda: root.attributes("-topmost", False))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
