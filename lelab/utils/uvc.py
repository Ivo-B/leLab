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

"""USB Video Class (UVC) camera controls for macOS.

OpenCV's AVFoundation backend on macOS ignores white balance, exposure, gain and the other image controls,
but the controls exist in the camera and are reachable with USB control requests sent to the camera's
VideoControl interface (UVC 1.5 spec, section 4). This module sends them through libusb, loaded with
``ctypes`` from the ``libusb-package`` wheel.

The streaming interfaces belong to macOS' video driver and the VideoControl requests only need the device
open, so this never claims an interface: the camera keeps streaming to OpenCV and the browser while its
controls are changed, and no root is needed. pyusb is not usable for this because it claims the interface
before every interface-recipient request, and macOS refuses.

The values live in the camera and survive a ``VideoCapture`` open and close; a power cycle resets them.
"""

from __future__ import annotations

import ctypes
import logging
import platform
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, Protocol

logger = logging.getLogger(__name__)

# Class-specific requests (UVC 1.5, table A-8).
SET_CUR = 0x01
GET_CUR = 0x81
GET_MIN = 0x82
GET_MAX = 0x83
GET_RES = 0x84
GET_INFO = 0x86
GET_DEF = 0x87

_REQUEST_IN = 0xA1  # device-to-host | class | interface
_REQUEST_OUT = 0x21  # host-to-device | class | interface
_INFO_SET_SUPPORTED = 0x02

# Descriptor constants.
_CLASS_VIDEO = 0x0E
_SUBCLASS_VIDEO_CONTROL = 0x01
_DESC_INTERFACE = 0x04
_DESC_CS_INTERFACE = 0x24
_VC_INPUT_TERMINAL = 0x02
_VC_PROCESSING_UNIT = 0x05
_ITT_CAMERA = 0x0201

# Only devices of this class (miscellaneous, interface association) are opened while listing: UVC cameras
# use it, and it keeps hubs and the arms' serial adapters untouched.
_DEVICE_CLASS_MISC = 0xEF

_TIMEOUT_MS = 1000


class UvcUnavailableError(RuntimeError):
    """UVC controls cannot be used here: wrong platform, or libusb is missing."""


class UvcTransferError(OSError):
    """A control request failed on the device."""


class UnsupportedControlError(ValueError):
    """The control is unknown, or this camera does not advertise it."""


class CameraNotFoundError(LookupError):
    """No UVC camera with that USB id is attached."""


@dataclass(frozen=True)
class Control:
    """One UVC control: where it lives and how its value is encoded."""

    name: str
    unit: Literal["pu", "ct"]  # processing unit or camera (input) terminal
    selector: int
    length: int  # bytes
    signed: bool
    bit: int  # position in the unit's bmControls bitmap
    kind: Literal["range", "bool", "enum"]


# bmControls bit positions: UVC 1.5 tables 4-7 (camera terminal) and 4-13 (processing unit).
_CONTROL_LIST = (
    Control("brightness", "pu", 0x02, 2, True, 0, "range"),
    Control("contrast", "pu", 0x03, 2, False, 1, "range"),
    Control("hue", "pu", 0x06, 2, True, 2, "range"),
    Control("saturation", "pu", 0x07, 2, False, 3, "range"),
    Control("sharpness", "pu", 0x08, 2, False, 4, "range"),
    Control("gamma", "pu", 0x09, 2, False, 5, "range"),
    Control("white_balance_temperature", "pu", 0x0A, 2, False, 6, "range"),
    Control("backlight_compensation", "pu", 0x01, 2, False, 8, "range"),
    Control("gain", "pu", 0x04, 2, False, 9, "range"),
    Control("power_line_frequency", "pu", 0x05, 1, False, 10, "enum"),
    Control("white_balance_temperature_auto", "pu", 0x0B, 1, False, 12, "bool"),
    Control("exposure_auto", "ct", 0x02, 1, False, 1, "enum"),
    Control("exposure_time_absolute", "ct", 0x04, 4, False, 3, "range"),
)
CONTROLS: dict[str, Control] = {c.name: c for c in _CONTROL_LIST}

# An auto control gates the manual controls beneath it: while auto is engaged the camera ignores writes to
# them. Each maps to the value that selects manual mode.
GATES: dict[str, int] = {"white_balance_temperature_auto": 0, "exposure_auto": 1}


@dataclass(frozen=True)
class VideoControlInfo:
    """What a camera's configuration descriptor says about its VideoControl interface."""

    interface: int
    processing_unit: int | None
    camera_terminal: int | None
    pu_controls: bytes
    ct_controls: bytes


@dataclass(frozen=True)
class UvcCameraInfo:
    """A UVC camera found on the USB bus."""

    vid: int
    pid: int
    name: str
    controls: list[str]
    location: int | None = None
    """Bus and port path (IOKit ``locationID``); tells two identical cameras apart."""


def _bit_set(bitmap: bytes, bit: int) -> bool:
    byte = bit // 8
    return byte < len(bitmap) and bool((bitmap[byte] >> (bit % 8)) & 1)


def parse_video_control(config: bytes) -> VideoControlInfo | None:
    """Find the VideoControl interface and its unit ids in a configuration descriptor.

    Returns None when the descriptor has no VideoControl interface or no unit to control. A truncated
    descriptor yields whatever was complete before the cut.
    """
    interface: int | None = None
    processing_unit: int | None = None
    camera_terminal: int | None = None
    pu_controls = b""
    ct_controls = b""
    in_video_control = False
    offset = 0
    total = len(config)

    while offset + 2 <= total:
        length, kind = config[offset], config[offset + 1]
        if length < 2 or offset + length > total:
            break
        body = config[offset : offset + length]

        if kind == _DESC_INTERFACE and length >= 9:
            in_video_control = body[5] == _CLASS_VIDEO and body[6] == _SUBCLASS_VIDEO_CONTROL
            if in_video_control and interface is None:
                interface = body[2]
            elif in_video_control:
                in_video_control = False  # only the first VideoControl interface counts
        elif in_video_control and kind == _DESC_CS_INTERFACE and length >= 3:
            subtype = body[2]
            if subtype == _VC_INPUT_TERMINAL and length >= 15:
                if body[4] | (body[5] << 8) == _ITT_CAMERA:
                    camera_terminal = body[3]
                    ct_controls = bytes(body[15 : 15 + body[14]])
            elif subtype == _VC_PROCESSING_UNIT and length >= 8:
                processing_unit = body[3]
                pu_controls = bytes(body[8 : 8 + body[7]])
        offset += length

    if interface is None or (processing_unit is None and camera_terminal is None):
        return None
    return VideoControlInfo(interface, processing_unit, camera_terminal, pu_controls, ct_controls)


class Transport(Protocol):
    """How requests reach the camera. Raises UvcTransferError when a request fails."""

    def control_in(self, request_type: int, request: int, value: int, index: int, length: int) -> bytes: ...

    def control_out(self, request_type: int, request: int, value: int, index: int, data: bytes) -> None: ...


class UvcCamera:
    """The UVC controls of one camera, over any `Transport`."""

    def __init__(
        self, transport: Transport, info: VideoControlInfo, vid: int = 0, pid: int = 0, name: str = ""
    ) -> None:
        self._transport = transport
        self.info = info
        self.vid = vid
        self.pid = pid
        self.name = name

    # -- discovery ----------------------------------------------------------------------------------

    def _unit_and_bitmap(self, control: Control) -> tuple[int | None, bytes]:
        if control.unit == "pu":
            return self.info.processing_unit, self.info.pu_controls
        return self.info.camera_terminal, self.info.ct_controls

    def is_supported(self, name: str) -> bool:
        control = CONTROLS.get(name)
        if control is None:
            return False
        unit, bitmap = self._unit_and_bitmap(control)
        return unit is not None and _bit_set(bitmap, control.bit)

    def supported(self) -> list[str]:
        return [name for name in CONTROLS if self.is_supported(name)]

    def _control(self, name: str) -> Control:
        if name not in CONTROLS:
            raise UnsupportedControlError(f"Unknown camera control '{name}'")
        if not self.is_supported(name):
            raise UnsupportedControlError(f"This camera does not support '{name}'")
        return CONTROLS[name]

    # -- requests -----------------------------------------------------------------------------------

    def _index(self, control: Control) -> int:
        unit, _ = self._unit_and_bitmap(control)
        assert unit is not None
        return (unit << 8) | self.info.interface

    @staticmethod
    def _decode(control: Control, raw: bytes) -> int:
        if len(raw) < control.length:
            raise UvcTransferError(f"Short reply for '{control.name}': {len(raw)} of {control.length} bytes")
        return int.from_bytes(raw[: control.length], "little", signed=control.signed)

    def _get(self, control: Control, request: int) -> int:
        raw = self._transport.control_in(
            _REQUEST_IN, request, control.selector << 8, self._index(control), control.length
        )
        return self._decode(control, raw)

    def _set(self, control: Control, value: int) -> None:
        data = int(value).to_bytes(control.length, "little", signed=control.signed)
        self._transport.control_out(_REQUEST_OUT, SET_CUR, control.selector << 8, self._index(control), data)

    def read(self, name: str) -> int:
        return self._get(self._control(name), GET_CUR)

    def describe(self, name: str) -> dict[str, Any]:
        control = self._control(name)
        info: dict[str, Any] = {
            "name": name,
            "kind": control.kind,
            "value": self._get(control, GET_CUR),
            "default": self._get(control, GET_DEF),
        }
        if control.kind == "range":
            info["min"] = self._get(control, GET_MIN)
            info["max"] = self._get(control, GET_MAX)
            info["step"] = max(self._get(control, GET_RES), 1)
        try:
            raw = self._transport.control_in(
                _REQUEST_IN, GET_INFO, control.selector << 8, self._index(control), 1
            )
            info["writable"] = bool(raw and raw[0] & _INFO_SET_SUPPORTED)
        except UvcTransferError:
            info["writable"] = True  # some cameras do not answer GET_INFO; try the write
        return info

    def write(self, name: str, value: int) -> int:
        """Write one control and return the value the camera reports back.

        Range controls are clamped to the camera's own minimum and maximum, so a bad value cannot push it
        outside what it accepts.
        """
        control = self._control(name)
        value = int(value)
        if control.kind == "range":
            low, high = self._get(control, GET_MIN), self._get(control, GET_MAX)
            if low <= high:
                value = min(max(value, low), high)
        self._set(control, value)
        return self._get(control, GET_CUR)

    def apply(self, controls: Mapping[str, int]) -> dict[str, int]:
        """Write several controls, auto modes first, and return the values read back.

        Every name is checked before the first write, so an unsupported control leaves the camera as it was.
        """
        for name in controls:
            self._control(name)
        ordered = sorted(controls, key=lambda name: name not in GATES)
        return {name: self.write(name, controls[name]) for name in ordered}

    def reset(self) -> dict[str, int]:
        """Restore every supported control to the camera's own default and return the values read back.

        The auto controls drop to manual first, otherwise the camera ignores the manual defaults, and go back
        to their own defaults last.
        """
        supported = self.supported()
        gates = [name for name in supported if name in GATES]
        manual = [name for name in supported if name not in GATES]

        for name in gates:
            self._set(CONTROLS[name], GATES[name])
        for name in manual:
            self._set(CONTROLS[name], self._get(CONTROLS[name], GET_DEF))
        for name in gates:
            self._set(CONTROLS[name], self._get(CONTROLS[name], GET_DEF))
        return {name: self._get(CONTROLS[name], GET_CUR) for name in supported}


# -- libusb transport (macOS) -------------------------------------------------------------------------


class _DeviceDescriptor(ctypes.Structure):
    """libusb_device_descriptor (USB 2.0 spec, table 9-8)."""

    _fields_ = [
        ("bLength", ctypes.c_uint8),
        ("bDescriptorType", ctypes.c_uint8),
        ("bcdUSB", ctypes.c_uint16),
        ("bDeviceClass", ctypes.c_uint8),
        ("bDeviceSubClass", ctypes.c_uint8),
        ("bDeviceProtocol", ctypes.c_uint8),
        ("bMaxPacketSize0", ctypes.c_uint8),
        ("idVendor", ctypes.c_uint16),
        ("idProduct", ctypes.c_uint16),
        ("bcdDevice", ctypes.c_uint16),
        ("iManufacturer", ctypes.c_uint8),
        ("iProduct", ctypes.c_uint8),
        ("iSerialNumber", ctypes.c_uint8),
        ("bNumConfigurations", ctypes.c_uint8),
    ]


def _load_libusb() -> ctypes.CDLL:
    if platform.system() != "Darwin":
        raise UvcUnavailableError(
            "UVC camera controls are only implemented on macOS. On Linux, use `v4l2-ctl` "
            "(white_balance_automatic=0, white_balance_temperature=...)."
        )
    try:
        import libusb_package
    except ImportError as exc:
        raise UvcUnavailableError("The libusb-package wheel is not installed; run `uv sync`.") from exc
    path = libusb_package.find_library("libusb-1.0")
    if not path:
        raise UvcUnavailableError("libusb-package does not contain a libusb for this system.")
    try:
        lib = ctypes.CDLL(path)
    except OSError as exc:
        raise UvcUnavailableError(f"Could not load libusb: {exc}") from exc

    # Every function declares its types: without them ctypes passes pointers as 32-bit ints and the
    # process crashes in libusb_close/libusb_exit.
    void_p, u8, u16 = ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint16
    lib.libusb_init.argtypes = [ctypes.POINTER(void_p)]
    lib.libusb_init.restype = ctypes.c_int
    lib.libusb_exit.argtypes = [void_p]
    lib.libusb_exit.restype = None
    lib.libusb_get_device_list.argtypes = [void_p, ctypes.POINTER(ctypes.POINTER(void_p))]
    lib.libusb_get_device_list.restype = ctypes.c_ssize_t
    lib.libusb_free_device_list.argtypes = [ctypes.POINTER(void_p), ctypes.c_int]
    lib.libusb_free_device_list.restype = None
    lib.libusb_get_device_descriptor.argtypes = [void_p, ctypes.POINTER(_DeviceDescriptor)]
    lib.libusb_get_device_descriptor.restype = ctypes.c_int
    lib.libusb_open.argtypes = [void_p, ctypes.POINTER(void_p)]
    lib.libusb_open.restype = ctypes.c_int
    lib.libusb_close.argtypes = [void_p]
    lib.libusb_close.restype = None
    lib.libusb_control_transfer.argtypes = [void_p, u8, u8, u16, u16, ctypes.c_char_p, u16, ctypes.c_uint]
    lib.libusb_control_transfer.restype = ctypes.c_int
    lib.libusb_get_bus_number.argtypes = [void_p]
    lib.libusb_get_bus_number.restype = u8
    lib.libusb_get_port_numbers.argtypes = [void_p, ctypes.POINTER(u8), ctypes.c_int]
    lib.libusb_get_port_numbers.restype = ctypes.c_int
    lib.libusb_get_string_descriptor_ascii.argtypes = [void_p, u8, ctypes.c_char_p, ctypes.c_int]
    lib.libusb_get_string_descriptor_ascii.restype = ctypes.c_int
    return lib


class _LibusbTransport:
    """Control requests on an open libusb device handle."""

    def __init__(self, lib: ctypes.CDLL, handle: ctypes.c_void_p) -> None:
        self._lib = lib
        self._handle = handle

    def control_in(self, request_type: int, request: int, value: int, index: int, length: int) -> bytes:
        buf = ctypes.create_string_buffer(length)
        n = self._lib.libusb_control_transfer(
            self._handle, request_type, request, value, index, buf, length, _TIMEOUT_MS
        )
        if n < 0:
            raise UvcTransferError(f"libusb control transfer failed ({n})")
        return buf.raw[:n]

    def control_out(self, request_type: int, request: int, value: int, index: int, data: bytes) -> None:
        n = self._lib.libusb_control_transfer(
            self._handle, request_type, request, value, index, data, len(data), _TIMEOUT_MS
        )
        if n < 0:
            raise UvcTransferError(f"libusb control transfer failed ({n})")

    def config_descriptor(self) -> bytes:
        header = self.control_in(0x80, 0x06, 0x0200, 0, 9)  # standard GET_DESCRIPTOR(configuration)
        if len(header) < 4:
            return b""
        return self.control_in(0x80, 0x06, 0x0200, 0, header[2] | (header[3] << 8))

    def product_name(self, string_index: int) -> str:
        if not string_index:
            return ""
        buf = ctypes.create_string_buffer(128)
        n = self._lib.libusb_get_string_descriptor_ascii(self._handle, string_index, buf, len(buf))
        return buf.value.decode("utf-8", "replace").strip() if n > 0 else ""


def _location(lib: ctypes.CDLL, device: Any) -> int:
    """The device's IOKit ``locationID``: the bus number, then one nibble per hub port on the way down."""
    ports = (ctypes.c_uint8 * 7)()
    count = max(lib.libusb_get_port_numbers(device, ports, len(ports)), 0)
    location = lib.libusb_get_bus_number(device) << 24
    for depth, port in enumerate(ports[:count]):
        location |= port << (20 - 4 * depth)
    return location


@contextmanager
def _usb_devices(
    vid: int | None = None, pid: int | None = None
) -> Iterator[list[tuple[Any, _DeviceDescriptor, int]]]:
    """Every attached USB device (optionally one vid:pid) with its descriptor and location, inside a libusb context."""
    lib = _load_libusb()
    ctx = ctypes.c_void_p()
    if lib.libusb_init(ctypes.byref(ctx)) != 0:
        raise UvcUnavailableError("libusb could not be initialised")
    device_list = ctypes.POINTER(ctypes.c_void_p)()
    try:
        count = lib.libusb_get_device_list(ctx, ctypes.byref(device_list))
        found: list[tuple[Any, _DeviceDescriptor, int]] = []
        for i in range(max(count, 0)):
            device = device_list[i]
            descriptor = _DeviceDescriptor()
            if lib.libusb_get_device_descriptor(device, ctypes.byref(descriptor)) != 0:
                continue
            if (vid is None or descriptor.idVendor == vid) and (pid is None or descriptor.idProduct == pid):
                found.append((device, descriptor, _location(lib, device)))
        yield found
    finally:
        if device_list:
            lib.libusb_free_device_list(device_list, 1)
        lib.libusb_exit(ctx)


@contextmanager
def _open_transport(device: Any) -> Iterator[_LibusbTransport]:
    lib = _load_libusb()
    handle = ctypes.c_void_p()
    rc = lib.libusb_open(device, ctypes.byref(handle))
    if rc != 0 or not handle:
        raise UvcTransferError(f"Could not open the USB device ({rc})")
    try:
        yield _LibusbTransport(lib, handle)
    finally:
        lib.libusb_close(handle)


def list_cameras() -> list[UvcCameraInfo]:
    """UVC cameras on the USB bus with the controls each one supports."""
    cameras: list[UvcCameraInfo] = []
    with _usb_devices() as devices:
        for device, descriptor, location in devices:
            if descriptor.bDeviceClass != _DEVICE_CLASS_MISC:
                continue
            try:
                with _open_transport(device) as transport:
                    info = parse_video_control(transport.config_descriptor())
                    if info is None:
                        continue
                    name = transport.product_name(descriptor.iProduct)
                    vid, pid = descriptor.idVendor, descriptor.idProduct
                    supported = UvcCamera(transport, info, vid, pid, name).supported()
            except UvcTransferError as exc:
                logger.debug(
                    "Skipping USB device %04x:%04x: %s", descriptor.idVendor, descriptor.idProduct, exc
                )
                continue
            cameras.append(
                UvcCameraInfo(vid, pid, name or f"USB camera {vid:04x}:{pid:04x}", supported, location)
            )
    return cameras


@contextmanager
def open_camera(vid: int, pid: int, location: int | None = None) -> Iterator[UvcCamera]:
    """Open the attached UVC camera with this USB id, at this ``location`` when given.

    Two identical cameras share a vid:pid, so without a ``location`` that matches only one camera this
    refuses to guess which one to open.
    """
    with _usb_devices(vid, pid) as devices:
        candidates = [
            (device, descriptor, where)
            for device, descriptor, where in devices
            if descriptor.bDeviceClass == _DEVICE_CLASS_MISC and (location is None or where == location)
        ]
        if len(candidates) > 1:
            places = ", ".join(f"{vid:04x}:{pid:04x}@{where:x}" for _, _, where in candidates)
            raise CameraNotFoundError(
                f"{len(candidates)} cameras {vid:04x}:{pid:04x} are attached; name one by its location: {places}"
            )
        for device, descriptor, _ in candidates:
            with _open_transport(device) as transport:
                info = parse_video_control(transport.config_descriptor())
                if info is None:
                    continue
                yield UvcCamera(transport, info, vid, pid, transport.product_name(descriptor.iProduct))
                return
        where_text = "" if location is None else f" at {location:x}"
        raise CameraNotFoundError(f"No UVC camera {vid:04x}:{pid:04x}{where_text} is attached")
