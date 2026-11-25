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

from __future__ import annotations

import math

import numpy as np
import pytest

from lelab import color_gain, lens
from lelab.color_calibration import CalibrationCamera

from .test_color_gain import WHITE_PAPER, _camera_stub

LENS = {"fov": 120.0, "bend": 0.0, "zoom": 1.0}

# The same points, to the same precision, are pinned in frontend/src/lib/cameraLens.test.ts.
FIXTURE_LENS = {"fov": 110.0, "bend": 0.05, "zoom": 0.8}
FIXTURE_SIZE = (640, 480)
FIXTURE_POINTS = [
    ((0, 0), (22.51254984, 17.00080027)),
    ((100, 50), (91.32097101, 42.57550801)),
    ((639, 479), (617.48745016, 462.99919973)),
    ((320, 240), (320.62499912, 240.62499912)),
    ((10, 400), (16.2766783, 397.50433969)),
]

LISTING = [
    CalibrationCamera(1, "USB Camera", (0x0ABD, 0x8050), "0x1300000abd8050", 0x130000),
    CalibrationCamera(2, "USB Camera", (0x0ABD, 0x8050), "0x1410000abd8050", 0x141000),
]


@pytest.fixture(autouse=True)
def _no_lens_left_installed():
    yield
    lens.install({})
    color_gain.install({})


# -- validation -------------------------------------------------------------------------------------


def test_parse_size() -> None:
    assert lens.parse_size("1920x1080") == (1920, 1080)
    assert lens.parse_size("640X480") == (640, 480)
    for bad in ("wide", "640", "0x480", "640x-1"):
        with pytest.raises(lens.LensError):
            lens.parse_size(bad)


def test_a_lens_needs_exactly_fov_bend_and_zoom() -> None:
    assert lens.validate_lens({"fov": 120, "bend": 0, "zoom": 1}) == LENS
    with pytest.raises(lens.LensError, match="exactly"):
        lens.validate_lens({"fov": 120, "zoom": 1})
    with pytest.raises(lens.LensError, match="exactly"):
        lens.validate_lens({**LENS, "k1": 0.1})


@pytest.mark.parametrize(
    "bad",
    [
        {"fov": 5},
        {"fov": 250},
        {"fov": float("nan")},
        {"fov": "120"},
        {"fov": True},
        {"zoom": 0.1},
        {"zoom": 4},
        {"bend": 0.9},
    ],
)
def test_lens_values_must_be_in_range(bad) -> None:
    with pytest.raises(lens.LensError):
        lens.validate_lens({**LENS, **bad})


def test_a_lens_that_folds_the_picture_back_is_refused() -> None:
    with pytest.raises(lens.LensError, match="folds"):
        lens.validate_lens({"fov": 200.0, "bend": -0.15, "zoom": 1.0})
    assert lens.validate_lens({"fov": 60.0, "bend": -0.15, "zoom": 1.0})["bend"] == -0.15


# -- saved lenses -----------------------------------------------------------------------------------


def test_lenses_are_saved_per_normalised_camera_name() -> None:
    lens.save_lens("USB Camera  ", LENS)
    assert lens.load_saved() == {"USB Camera": LENS}
    assert lens.forget_lens("USB Camera") is True
    assert lens.load_saved() == {}
    assert lens.forget_lens("USB Camera") is False
    with pytest.raises(lens.LensError):
        lens.save_lens("   ", LENS)


def test_one_invalid_saved_lens_does_not_hide_the_others(tmp_path) -> None:
    from lelab.utils import config

    lens.save_lens("USB Camera", LENS)
    with open(config.CAMERA_LENS_FILE) as f:
        text = f.read()
    with open(config.CAMERA_LENS_FILE, "w") as f:
        f.write(text.rstrip().removesuffix("}") + ', "Broken": {"fov": 1}}')
    assert lens.load_saved() == {"USB Camera": LENS}


# -- the model --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("pixel", "expected"), FIXTURE_POINTS)
def test_source_coordinates_match_the_shared_fixtures(pixel, expected) -> None:
    x, y = lens.source_coordinates(FIXTURE_LENS, *FIXTURE_SIZE)
    assert (float(x[pixel[1], pixel[0]]), float(y[pixel[1], pixel[0]])) == pytest.approx(expected, abs=1e-6)


def test_a_narrow_field_of_view_is_nearly_the_identity() -> None:
    x, y = lens.source_coordinates({"fov": 10.0, "bend": 0.0, "zoom": 1.0}, 640, 480)
    grid_x, grid_y = np.meshgrid(np.arange(640) + 0.5, np.arange(480) + 0.5)
    assert np.abs(x - grid_x).max() < 1.5 and np.abs(y - grid_y).max() < 1.5


def test_the_centre_keeps_its_scale_and_zoom_magnifies_it() -> None:
    x1, _ = lens.source_coordinates(LENS, 641, 481)
    x2, _ = lens.source_coordinates({**LENS, "zoom": 2.0}, 641, 481)
    # One pixel right of the exact centre pixel is one source pixel at zoom 1, half a pixel at zoom 2.
    assert x1[240, 321] - x1[240, 320] == pytest.approx(1.0, abs=0.01)
    assert x2[240, 321] - x2[240, 320] == pytest.approx(0.5, abs=0.01)


def test_fit_zoom_is_the_widest_view_without_black_borders() -> None:
    size = (160, 120)
    zoom = lens.fit_zoom(LENS, *size)
    white = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
    assert lens.apply_lens(white, {**LENS, "zoom": zoom}).min() == 255
    assert lens.apply_lens(white, {**LENS, "zoom": zoom * 0.9}).min() == 0


def test_fit_zoom_depends_on_the_aspect_ratio_and_not_on_the_resolution() -> None:
    assert lens.fit_zoom(LENS, 1920, 1080) == lens.fit_zoom(LENS, 960, 540)
    # The shorter the picture is compared with its diagonal, the closer its edges sit to the centre.
    assert lens.fit_zoom(LENS, 640, 480) != lens.fit_zoom(LENS, 1920, 1080)
    assert lens.fit_zoom(LENS, 480, 640) == lens.fit_zoom(LENS, 640, 480)


def test_fit_zoom_stays_in_range_for_extreme_lenses() -> None:
    wide = {"fov": 200.0, "bend": 0.0, "zoom": 1.0}
    assert lens.ZOOM_RANGE[0] <= lens.fit_zoom(wide, 640, 480) < 0.7
    assert lens.fit_zoom({"fov": 10.0, "bend": 0.0, "zoom": 1.0}, 640, 480) == pytest.approx(1.0, abs=0.02)


def test_apply_lens_keeps_shape_and_dtype_and_caches_the_maps() -> None:
    frame = np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    out = lens.apply_lens(frame, LENS)
    assert out.shape == frame.shape and out.dtype == np.uint8
    before = lens._maps.cache_info().hits
    lens.apply_lens(frame, LENS)
    assert lens._maps.cache_info().hits == before + 1
    assert lens.apply_lens(frame[:, :, 0], LENS).shape == (48, 64)


def test_a_plane_seen_through_a_fisheye_comes_out_straight() -> None:
    """Render a smooth pattern on a plane through an equidistant lens; correcting it must give the plane back."""
    width, height = 320, 240
    fov, zoom = 100.0, 1.0
    half_diagonal = math.hypot(width, height) / 2
    theta_max = math.radians(fov / 2)

    def plane(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (0.5 + 0.5 * np.sin(60 * x) * np.sin(60 * y)).astype(np.float32)

    u, v = np.meshgrid(
        (np.arange(width) + 0.5 - width / 2) / half_diagonal,
        (np.arange(height) + 0.5 - height / 2) / half_diagonal,
    )
    r_source = np.hypot(u, v)
    theta = r_source * theta_max  # the lens is equidistant
    focal = zoom / theta_max  # the same output focal length the model uses
    plane_radius = focal * np.tan(theta)
    scale = np.divide(
        plane_radius, r_source, out=np.full_like(r_source, focal * theta_max), where=r_source > 1e-12
    )
    fisheye = plane(u * scale, v * scale)

    corrected = lens.apply_lens(fisheye, {"fov": fov, "bend": 0.0, "zoom": zoom})
    inside = np.hypot(u, v) < 0.85
    expected = plane(u, v)
    assert np.abs(corrected - expected)[inside].mean() < 0.03
    assert (
        np.abs(fisheye - expected)[inside].mean() > 0.15
    )  # the test would pass trivially if nothing were bent


# -- attached cameras, the session and the frame hook -----------------------------------------------


def test_a_session_gets_the_lens_saved_for_each_cameras_id() -> None:
    lens.save_lens("USB Camera [0x1300000abd8050]", LENS)
    cameras = {
        "top": {"camera_id": "0x1300000abd8050", "camera_index": 1},
        "side": {"camera_id": "0x1410000abd8050", "camera_index": 2},
    }
    assert lens.lenses_for_session(cameras, LISTING) == {1: LENS}


def test_a_camera_with_lens_correction_off_is_left_alone() -> None:
    lens.save_lens("USB Camera [0x1300000abd8050]", LENS)
    cameras = {"top": {"camera_id": "0x1300000abd8050", "camera_index": 1, "lens_correction": False}}
    assert lens.lenses_for_session(cameras, LISTING) == {}


def test_a_name_only_lens_is_not_applied() -> None:
    lens.save_lens("USB Camera", LENS)
    cameras = {"top": {"camera_id": "0x1300000abd8050", "camera_index": 1}}
    assert lens.lenses_for_session(cameras, LISTING) == {}


def test_the_environment_round_trip_keeps_the_index_map() -> None:
    mapping = {0: LENS, 3: FIXTURE_LENS}
    assert lens.decode_from_env(lens.encode_for_env(mapping)) == mapping


def test_install_straightens_only_the_mapped_camera_before_the_colour_gain() -> None:
    from lerobot.cameras.configs import ColorMode
    from lerobot.cameras.opencv import OpenCVCamera

    frame = np.random.default_rng(1).integers(40, 200, (48, 64, 3), dtype=np.uint8)
    lens.install({0: LENS})
    color_gain.install({0: WHITE_PAPER})

    both = OpenCVCamera._postprocess_image(_camera_stub(0, ColorMode.BGR, (48, 64)), frame)
    other = OpenCVCamera._postprocess_image(_camera_stub(1, ColorMode.BGR, (48, 64)), frame)

    assert np.array_equal(both, color_gain.apply_gains(lens.apply_lens(frame, LENS), WHITE_PAPER))
    assert np.array_equal(other, frame)


def test_an_empty_map_turns_the_lens_off_and_leaves_the_colour_on() -> None:
    from lerobot.cameras.configs import ColorMode
    from lerobot.cameras.opencv import OpenCVCamera

    frame = np.full((8, 8, 3), (136, 243, 168), dtype=np.uint8)
    lens.install({0: LENS})
    color_gain.install({0: WHITE_PAPER})
    lens.install({})
    out = OpenCVCamera._postprocess_image(_camera_stub(0, ColorMode.BGR, (8, 8)), frame)
    assert tuple(out[4, 4]) == (243, 243, 243)


def test_the_rollout_wrapper_installs_gains_and_lenses_before_lerobot_runs(monkeypatch) -> None:
    import lerobot.scripts.lerobot_rollout as lerobot_rollout
    from lelab.scripts import rollout_gain

    seen: dict = {}
    monkeypatch.setenv(color_gain.ENV_VAR, color_gain.encode_for_env({0: WHITE_PAPER}))
    monkeypatch.setenv(lens.ENV_VAR, lens.encode_for_env({0: LENS}))
    monkeypatch.setattr(color_gain, "install", lambda mapping: seen.setdefault("gains", mapping))
    monkeypatch.setattr(lens, "install", lambda mapping: seen.setdefault("lenses", mapping))
    monkeypatch.setattr(lerobot_rollout, "main", lambda: seen.setdefault("ran", True))

    rollout_gain.main()

    assert seen == {"gains": {0: WHITE_PAPER}, "lenses": {0: LENS}, "ran": True}


# -- the command line and the server ----------------------------------------------------------------


def test_cli_sets_shows_and_forgets_a_lens(capsys) -> None:
    from lelab.scripts import camera_controls as cli

    assert cli.main(["lens", "USB Camera", "--fov", "120", "--bend", "0.02", "--zoom", "0.9"]) == cli.EXIT_OK
    assert lens.load_saved() == {"USB Camera": {"fov": 120.0, "bend": 0.02, "zoom": 0.9}}

    assert cli.main(["lens"]) == cli.EXIT_OK
    assert "USB Camera: fov=120° bend=0.02 zoom=0.9" in capsys.readouterr().out

    assert cli.main(["lens", "USB Camera", "--forget"]) == cli.EXIT_OK
    assert lens.load_saved() == {}


def test_cli_fit_chooses_the_zoom_for_the_picture_size() -> None:
    from lelab.scripts import camera_controls as cli

    assert cli.main(["lens", "USB Camera", "--fov", "120", "--fit", "1920x1080"]) == cli.EXIT_OK
    assert lens.load_saved()["USB Camera"]["zoom"] == lens.fit_zoom(LENS, 1920, 1080)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["lens", "USB Camera"], "Give --fov"),
        (["lens", "USB Camera", "--fov", "5"], "between 10 and 200"),
        (["lens", "--fov", "120"], "Name the camera"),
        (["lens", "USB Camera", "--fov", "120", "--zoom", "1", "--fit", "640x480"], "either --zoom or --fit"),
        (["lens", "USB Camera", "--fov", "120", "--fit", "wide"], "like 1920x1080"),
        (["lens", "USB Camera", "--fov", "200", "--bend", "-0.15"], "folds"),
    ],
)
def test_cli_rejects_bad_lens_requests(argv, message, capsys) -> None:
    from lelab.scripts import camera_controls as cli

    assert cli.main(argv) == cli.EXIT_INVALID
    assert message in capsys.readouterr().err
    assert lens.load_saved() == {}


# -- measuring the field of view from a stroke along a straight edge --------------------------------


def _stroke_through_the_lens(
    fov: float, bend: float, start: tuple[float, float], end: tuple[float, float], count: int = 24
) -> list[tuple[float, float]]:
    """Pixels where a straight edge (drawn in the straightened picture) lands in a 1280x720 camera picture.

    The forward model is written out here on purpose: the test must not share the code it checks.
    """
    width, height = 1280, 720
    half_diagonal = math.hypot(width, height) / 2
    theta_max = math.radians(fov / 2)
    theta_dmax = theta_max * (1 + bend * theta_max**2)
    points = []
    for t in np.linspace(0.0, 1.0, count):
        qx = start[0] + t * (end[0] - start[0])
        qy = start[1] + t * (end[1] - start[1])
        rho = math.hypot(qx, qy)
        theta = math.atan(rho * theta_dmax)  # zoom 1
        r_source = theta * (1 + bend * theta**2) / theta_dmax
        scale = r_source / rho
        points.append((width / 2 + qx * scale * half_diagonal, height / 2 + qy * scale * half_diagonal))
    return points


@pytest.mark.parametrize(("fov", "bend"), [(120.0, 0.0), (135.0, 0.0), (100.0, 0.1), (150.0, -0.05)])
def test_a_stroke_along_a_bowed_edge_gives_back_the_field_of_view(fov, bend) -> None:
    stroke = _stroke_through_the_lens(fov, bend, (-0.5, -0.2), (0.6, 0.4))
    found = lens.fit_fov_to_line(stroke, 1280, 720, bend=bend)
    assert found == pytest.approx(fov, abs=0.5)


def test_a_sloppy_hand_stroke_still_gives_the_field_of_view_within_a_degree() -> None:
    rng = np.random.default_rng(0)
    stroke = np.array(_stroke_through_the_lens(130.0, 0.0, (-0.8, -0.3), (0.85, 0.55), count=40))
    stroke += rng.normal(scale=0.7, size=stroke.shape)  # less than a pixel of jitter, as a steady hand gives
    assert lens.fit_fov_to_line(stroke, 1280, 720) == pytest.approx(130.0, abs=1.0)


def test_a_stroke_through_the_middle_cannot_tell_the_fields_of_view_apart() -> None:
    stroke = [
        (640 + 40 * t, 360 + 25 * t) for t in np.linspace(-6, 6, 20)
    ]  # a straight line through the centre
    with pytest.raises(lens.LensError, match="does not bow"):
        lens.fit_fov_to_line(stroke, 1280, 720)


def test_a_stroke_that_is_already_straight_in_the_picture_is_refused() -> None:
    stroke = [(100 + 20 * t, 80 + 5 * t) for t in range(20)]  # straight in the picture, off the centre
    with pytest.raises(lens.LensError, match="does not bow"):
        lens.fit_fov_to_line(stroke, 1280, 720)


def test_a_stroke_that_no_field_of_view_straightens_is_refused() -> None:
    stroke = [(200 + 40 * t, 200 + 40 * t) for t in range(10)] + [
        (600 - 20 * t, 200 + 40 * t) for t in range(1, 10)
    ]  # a bent line with a sharp corner
    with pytest.raises(lens.LensError, match="Could not straighten"):
        lens.fit_fov_to_line(stroke, 1280, 720)


def test_a_stroke_with_too_few_points_is_refused() -> None:
    with pytest.raises(lens.LensError, match="longer stroke"):
        lens.fit_fov_to_line([(100, 100), (300, 200)], 1280, 720)
