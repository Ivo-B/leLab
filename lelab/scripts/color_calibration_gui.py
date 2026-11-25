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

"""``lelab-camera-calibrate``: a window that calibrates a camera's colour against a white sheet.

Pick the camera, hold a white sheet of paper under it, drag the box over the sheet (or keep the centre box)
and press Auto calibrate. The camera's white balance is swept to the most neutral temperature and the rest is
measured as a per-channel gain; Save keeps both, and LeLab applies them from the next preview, recording or
inference. ``lelab-camera calibrate`` does the same without a window.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

import cv2
import numpy as np

from .. import camera_controls, color_calibration, color_gain
from ..color_calibration import CalibrationCamera, CalibrationResult, Roi
from ..utils.camera_listing import CameraKeyError
from ..utils.tk_preview import display_scale, to_ppm

MIN_DRAG = 8
TICK_MS = 40
NEUTRAL_ENOUGH = 0.05


# -- pure helpers, tested without a display -----------------------------------------------------------


def roi_from_drag(
    start: tuple[float, float], end: tuple[float, float], scale: float, width: int, height: int
) -> Roi | None:
    """The box dragged in window pixels, in frame pixels; ``None`` for a click or a tiny drag."""
    (x0, y0), (x1, y1) = start, end
    if abs(x1 - x0) < MIN_DRAG or abs(y1 - y0) < MIN_DRAG:
        return None
    left, top = min(x0, x1) / scale, min(y0, y1) / scale
    box = (round(left), round(top), round(abs(x1 - x0) / scale), round(abs(y1 - y0) / scale))
    return color_calibration.clamp_roi(box, width, height)


def roi_to_display(roi: Roi, scale: float) -> tuple[float, float, float, float]:
    x, y, w, h = roi
    return x * scale, y * scale, (x + w) * scale, (y + h) * scale


def readout(stats: color_calibration.PatchStats) -> str:
    return f"red/green {stats.red_ratio:.2f} · blue/green {stats.blue_ratio:.2f}"


# -- the window ---------------------------------------------------------------------------------------


class CalibrationApp:
    def __init__(self, root: tk.Tk, cameras: list[CalibrationCamera], initial: CalibrationCamera) -> None:
        self.root = root
        self.cameras = cameras
        self.camera: CalibrationCamera | None = None
        self.source: color_calibration.OpenCVFrameSource | None = None
        self.frame: np.ndarray | None = None
        self.roi: Roi | None = None
        self.scale = 1.0
        self.result: CalibrationResult | None = None
        self.calibrator: color_calibration.Calibrator | None = None
        self.saved_gains: dict[str, float] | None = None
        self.busy = False
        self.unsaved = False  # a calibration is applied to the camera but not yet saved
        self.drag_start: tuple[float, float] | None = None
        self.events: queue.Queue[tuple[Any, ...]] = queue.Queue()
        self.photo: tk.PhotoImage | None = None

        root.title("LeLab camera colour calibration")
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
        self.refresh_button = ttk.Button(top, text="Refresh", command=self._refresh)
        self.refresh_button.pack(side="left")
        self._fill_camera_menu()

        ttk.Label(
            self.root,
            text="Hold a white sheet of paper under the camera, then drag a box over it (or keep the centre box).",
        ).pack(anchor="w", **pad)

        self.canvas = tk.Canvas(self.root, width=640, height=360, bg="black", highlightthickness=0)
        self.canvas.pack(**pad)
        self.image_item = self.canvas.create_image(0, 0, anchor="nw")
        self.roi_item = self.canvas.create_rectangle(0, 0, 0, 0, outline="#ffd400", width=2)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        live = ttk.Frame(self.root)
        live.pack(fill="x", **pad)
        self.readout_label = ttk.Label(live, text="", font=("TkFixedFont", 12))
        self.readout_label.pack(side="left")
        self.corrected = tk.BooleanVar(value=True)
        ttk.Checkbutton(live, text="Show corrected", variable=self.corrected).pack(side="right")

        buttons = ttk.Frame(self.root)
        buttons.pack(fill="x", **pad)
        self.calibrate_button = ttk.Button(buttons, text="Auto calibrate", command=self._calibrate)
        self.calibrate_button.pack(side="left")
        self.save_button = ttk.Button(buttons, text="Save", command=self._save, state="disabled")
        self.save_button.pack(side="left", padx=6)
        self.reset_button = ttk.Button(buttons, text="Reset camera", command=self._reset)
        self.reset_button.pack(side="left")
        self.progress = ttk.Progressbar(buttons, maximum=1.0, length=220)
        self.progress.pack(side="right")

        self.result_label = ttk.Label(self.root, text="", justify="left", wraplength=780)
        self.result_label.pack(anchor="w", **pad)
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

    def _undo_unsaved(self) -> None:
        """Put the camera back as it was if its last calibration was never saved."""
        if self.unsaved and self.calibrator is not None:
            self.calibrator.restore()
        self.unsaved = False

    def _select(self, camera: CalibrationCamera) -> None:
        self._undo_unsaved()
        if self.source is not None:
            self.source.close()
        self.camera, self.source, self.frame, self.roi, self.result = camera, None, None, None, None
        self.camera_menu.current(self.cameras.index(camera))
        self.saved_gains = color_calibration.saved_gains_for(camera, self.cameras)
        self.save_button.configure(state="disabled")
        self.result_label.configure(text="")
        self.progress.configure(value=0)
        try:
            self.source = color_calibration.OpenCVFrameSource(camera.index)
        except color_calibration.CalibrationError as exc:
            self._status(str(exc), error=True)
            return
        saved = "a saved gain" if self.saved_gains else "no saved gain"
        controls = (
            "" if camera.device else " It has no USB white-balance control, so only the gain is measured."
        )
        self._status(f"Showing {camera.label} ({saved}).{controls}")

    # -- the box --

    def _on_press(self, event: tk.Event) -> None:
        if not self.busy and self.frame is not None:
            self.drag_start = (event.x, event.y)

    def _on_drag(self, event: tk.Event) -> None:
        if self.drag_start is not None:
            self.canvas.coords(self.roi_item, *self.drag_start, event.x, event.y)

    def _on_release(self, event: tk.Event) -> None:
        if self.drag_start is None or self.frame is None:
            return
        height, width = self.frame.shape[:2]
        roi = roi_from_drag(self.drag_start, (event.x, event.y), self.scale, width, height)
        self.drag_start = None
        if roi is not None:
            self.roi = roi
        self._draw_roi()

    def _draw_roi(self) -> None:
        if self.roi is not None:
            self.canvas.coords(self.roi_item, *roi_to_display(self.roi, self.scale))

    # -- frames --

    def _tick(self) -> None:
        latest = None
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if event[0] == "frame":
                latest = event[1]
            else:
                self._handle(event)
        if latest is None and not self.busy and self.source is not None:
            try:
                latest = self.source.read()
            except color_calibration.CalibrationError as exc:
                self._status(str(exc), error=True)
        if latest is not None:
            self._show(latest)
        self.root.after(TICK_MS, self._tick)

    def _gains(self) -> dict[str, float] | None:
        return self.result.gains if self.result is not None else self.saved_gains

    def _show(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        if self.frame is None or self.frame.shape != frame.shape:
            self.scale = display_scale(width, height)
            self.canvas.configure(width=round(width * self.scale), height=round(height * self.scale))
            if self.roi is None:
                self.roi = color_calibration.default_roi(width, height)
            self._draw_roi()
        self.frame = frame
        gains = self._gains() if self.corrected.get() else None

        def corrected(image: np.ndarray) -> np.ndarray:
            return color_gain.apply_gains(image, gains) if gains else image

        size = (round(width * self.scale), round(height * self.scale))
        small = corrected(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
        self.photo = tk.PhotoImage(data=to_ppm(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)), format="PPM")
        self.canvas.itemconfigure(self.image_item, image=self.photo)
        if self.roi is not None:
            x, y, w, h = color_calibration.clamp_roi(self.roi, width, height)
            stats = color_calibration.measure_patch([corrected(frame[y : y + h, x : x + w])], (0, 0, w, h))
            neutral = color_calibration.neutrality(stats) < NEUTRAL_ENOUGH
            self.readout_label.configure(
                text=f"{'Corrected box' if gains else 'Box'}: {readout(stats)}{'  ✓ neutral' if neutral else ''}",
                foreground="#1a7f37" if neutral else "",
            )

    # -- actions --

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        state = "disabled" if busy else "normal"
        for button in (self.calibrate_button, self.reset_button, self.refresh_button):
            button.configure(state=state)
        self.camera_menu.configure(state="disabled" if busy else "readonly")
        if busy:
            self.save_button.configure(state="disabled")

    def _calibrate(self) -> None:
        if self.busy or self.source is None or self.roi is None or self.camera is None:
            return
        self._undo_unsaved()
        self._set_busy(True)
        self.result = None
        self.result_label.configure(text="")
        self.calibrator = color_calibration.Calibrator(
            self.source,
            color_calibration.white_balance_for(self.camera),
            brightness=color_calibration.brightness_for(self.camera),
            on_frame=lambda frame: self.events.put(("frame", frame)),
        )
        calibrator, roi = self.calibrator, self.roi

        def work() -> None:
            try:
                result = calibrator.run(
                    roi, progress=lambda f, message: self.events.put(("progress", f, message))
                )
            except Exception as exc:  # every failure has to reach the window, or it stays busy
                self.events.put(("error", str(exc)))
            else:
                self.events.put(("done", result))

        threading.Thread(target=work, name="colour-calibration", daemon=True).start()

    def _handle(self, event: tuple[Any, ...]) -> None:
        kind = event[0]
        if kind == "progress":
            self.progress.configure(value=event[1])
            self._status(event[2])
        elif kind == "done":
            self._set_busy(False)
            self.result = event[1]
            self.unsaved = True
            self.corrected.set(True)
            self.result_label.configure(text="\n".join(color_calibration.describe_result(self.result)))
            self.save_button.configure(state="normal")
            self._status("Calibrated. Check the corrected picture, then press Save to keep it.")
        elif kind == "error":
            self._set_busy(False)
            self.progress.configure(value=0)
            self._status(event[1], error=True)

    def _save(self) -> None:
        if self.result is None or self.calibrator is None or self.camera is None:
            return
        try:
            self.saved_gains = self.calibrator.save(self.result, self.camera.key)
        except Exception as exc:  # shown in the window; the result stays so Save can be retried
            self._status(f"Could not save: {exc}", error=True)
            return
        self.save_button.configure(state="disabled")
        self.unsaved = False
        parts = []
        if self.result.temperature:
            parts.append(f"white balance {self.result.temperature} K")
        if self.result.brightness is not None:
            parts.append(f"brightness {self.result.brightness:+d}")
        temperature = ", ".join(parts) + (" and " if parts else "")
        self._status(
            f"Saved {temperature}the gain for '{self.camera.key}'. LeLab uses it from the next preview, "
            "recording or inference (reload an open browser page)."
        )

    def _reset(self) -> None:
        if self.busy or self.camera is None:
            return
        camera = self.camera
        what = "restore the camera's own settings and " if camera.device else ""
        if not messagebox.askyesno(
            "Reset camera", f"This will {what}forget the saved gain for '{camera.name}'."
        ):
            return
        try:
            if camera.device:
                camera_controls.reset_controls(camera.device)
            color_calibration.forget_gain_for(camera)
        except Exception as exc:  # shown in the window
            self._status(f"Could not reset: {exc}", error=True)
            return
        self.result, self.saved_gains, self.unsaved = None, None, False
        self.result_label.configure(text="")
        self.save_button.configure(state="disabled")
        self._status(f"Reset {camera.name}.")

    def _status(self, text: str, error: bool = False) -> None:
        self.status_label.configure(text=text, foreground="#b42318" if error else "")

    def close(self) -> None:
        if self.busy:
            self._status("Wait for the calibration to finish before closing.", error=True)
            return
        self._undo_unsaved()
        if self.source is not None:
            self.source.close()
        self.root.destroy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lelab-camera-calibrate", description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--camera", help="Camera name or OpenCV index to start with (default: first USB camera)."
    )
    args = parser.parse_args(argv)
    cameras = color_calibration.list_calibration_cameras()
    try:
        initial = color_calibration.find_camera(cameras, args.camera)
    except (color_calibration.CameraNotAttachedError, CameraKeyError) as exc:
        hint = "" if sys.platform == "darwin" else " The camera listing needs macOS."
        print(f"lelab-camera-calibrate: {exc}.{hint}", file=sys.stderr)
        return 4
    root = tk.Tk()
    CalibrationApp(root, cameras, initial)
    root.lift()
    root.attributes("-topmost", True)
    root.after(500, lambda: root.attributes("-topmost", False))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
