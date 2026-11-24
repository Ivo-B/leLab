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

import math
import random
import xml.etree.ElementTree as ET  # nosec B405 - parsing our own generated XML
from dataclasses import asdict
from pathlib import Path

import chess
import numpy as np
import pytest

from lelab.sim.cell import SimCell
from lelab.sim.randomize import (
    SimVariation,
    randomize_cell,
    randomize_joints,
    randomize_variation,
)
from lelab.sim.scene import PIECE_MASS, SimPosition, scene_xml

SEEDS = range(200)


def test_one_seed_reproduces_the_cell_and_the_gate_does_not_randomize() -> None:
    cell = SimCell()
    one = randomize_cell(cell, random.Random(4))
    two = randomize_cell(cell, random.Random(4))
    assert one == two
    assert one.top_camera.pos != cell.top_camera.pos
    assert abs(one.top_camera.pos[0] - cell.top_camera.pos[0]) <= 0.01 + 1e-9
    assert cell == SimCell()  # the frozen original is untouched


def test_the_expert_gate_never_randomizes(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.sim_helpers import arm_assets_or_skip

    arm_assets_or_skip()
    from lelab.sim import generate

    def forbidden(*args, **kwargs):
        raise AssertionError("the gate randomized the scene")

    monkeypatch.setattr(generate, "randomize_cell", forbidden)
    monkeypatch.setattr(generate, "randomize_variation", forbidden)
    monkeypatch.setattr(generate, "TRANSFER_PLAN", generate.TRANSFER_PLAN[:1])
    seen: list[SimVariation | None] = []
    settled = generate.settled_world
    monkeypatch.setattr(
        generate, "settled_world", lambda *a, **k: (seen.append(k.get("variation")), settled(*a, **k))[1]
    )
    (result,) = generate.gate_seed(0, SimCell().to_dict(), asdict(generate.ExpertParams()))
    assert seen == [None] and result[0]


def test_cameras_stay_within_the_specified_ranges() -> None:
    base = SimCell()
    for seed in SEEDS:
        cell = randomize_cell(base, random.Random(seed))
        for name in ("top_camera", "side_camera"):
            before, after = getattr(base, name), getattr(cell, name)
            assert all(abs(a - b) <= 0.01 + 1e-9 for a, b in zip(after.pos, before.pos, strict=True))
            assert abs(after.fovy - before.fovy) <= 2.0 + 1e-9
            for lo in (0, 3):  # each axis turns by the same yaw, within 2 degrees, about the vertical
                old, new = np.array(before.xyaxes[lo : lo + 3]), np.array(after.xyaxes[lo : lo + 3])
                assert new[2] == pytest.approx(old[2])
                assert np.linalg.norm(new) == pytest.approx(np.linalg.norm(old))
                cos = float(old[:2] @ new[:2] / (np.linalg.norm(old[:2]) * np.linalg.norm(new[:2])))
                assert math.degrees(math.acos(min(1.0, cos))) <= 2.0 + 1e-6
    # ... and the draws actually spread over the range rather than sitting at one end
    xs = [randomize_cell(base, random.Random(s)).top_camera.pos[0] - base.top_camera.pos[0] for s in SEEDS]
    assert min(xs) < -0.008 and max(xs) > 0.008


def test_a_randomized_cell_survives_the_dict_round_trip() -> None:
    cell = randomize_cell(SimCell(), random.Random(1))
    assert SimCell.from_dict(cell.to_dict()) == cell


def test_joints_move_by_at_most_two_degrees_and_follow_the_seed() -> None:
    home = (0.0, -100.0, 20.0, 30.0, 0.0, 30.0)
    out = randomize_joints(home, random.Random(2))
    assert out == randomize_joints(home, random.Random(2))
    assert out != home and len(out) == 6
    for seed in SEEDS:
        assert all(
            abs(a - b) <= 2.0 + 1e-9
            for a, b in zip(randomize_joints(home, random.Random(seed)), home, strict=True)
        )


def test_variation_ranges_and_reproducibility() -> None:
    home = (0.0, -100.0, 20.0, 30.0, 0.0, 30.0)
    assert randomize_variation(random.Random(9), home) == randomize_variation(random.Random(9), home)
    nominal = SimVariation()
    seen = set()
    for seed in SEEDS:
        v = randomize_variation(random.Random(seed), home)
        seen.add(v.piece_mass)
        assert abs(v.piece_mass / nominal.piece_mass - 1) <= 0.30 + 1e-9
        assert abs(v.piece_friction / nominal.piece_friction - 1) <= 0.30 + 1e-9
        assert abs(v.actuator_gain - 1) <= 0.15 + 1e-9
        assert 0.7 - 1e-9 <= v.light_intensity <= 1.3 + 1e-9
        assert all(abs(a - b) <= 0.15 + 1e-9 for a, b in zip(v.light_dir, nominal.light_dir, strict=True))
        colours = (*v.board_rgb, v.white_piece_rgb, v.black_piece_rgb, v.table_rgb)
        assert all(0.0 <= c <= 1.0 for rgb in colours for c in rgb)
        assert v.initial_joints is not None and len(v.initial_joints) == 6
        assert min(v.white_piece_rgb) > 0.6 and max(v.black_piece_rgb) < 0.3  # still told apart
    assert len(seen) == len(SEEDS)
    assert randomize_variation(random.Random(1), home) != randomize_variation(random.Random(2), home)


def test_variation_survives_the_dict_round_trip_for_the_process_pool() -> None:
    v = randomize_variation(random.Random(3), (0.0,) * 6)
    assert SimVariation.from_dict(v.to_dict()) == v
    assert SimVariation.from_dict(SimVariation().to_dict()) == SimVariation()
    assert asdict(v) == v.to_dict()


def test_scene_defaults_are_the_nominal_scene_and_arguments_change_it() -> None:
    cell, position = SimCell(), SimPosition(chess.Board())
    assert scene_xml(cell, position) == scene_xml(cell, position, **SimVariation().scene_kwargs())
    nominal = ET.fromstring(scene_xml(cell, position))  # nosec B314
    default_geom = next(g for g in nominal.iter("default") if g.get("class") == "piece").find("geom")
    assert float(default_geom.get("mass")) == pytest.approx(PIECE_MASS)  # type: ignore[union-attr]

    v = randomize_variation(random.Random(5), (0.0,) * 6)
    root = ET.fromstring(scene_xml(cell, position, **v.scene_kwargs()))  # nosec B314
    piece = next(g for g in root.iter("default") if g.get("class") == "piece").find("geom")
    assert float(piece.get("mass")) == pytest.approx(v.piece_mass, rel=1e-4)  # type: ignore[union-attr]
    assert float(piece.get("friction").split()[0]) == pytest.approx(v.piece_friction, rel=1e-4)  # type: ignore[union-attr]
    lights = list(root.iter("light"))
    key = next(light for light in lights if light.get("name") == "key")
    assert [float(x) for x in key.get("dir").split()] == pytest.approx(list(v.light_dir), abs=1e-4)  # type: ignore[union-attr]
    assert float(key.get("diffuse").split()[0]) == pytest.approx(0.25 * v.light_intensity, abs=1e-4)  # type: ignore[union-attr]
    materials = {m.get("name"): m.get("rgba") for m in root.iter("material")}
    for name, rgb in (
        ("white_piece", v.white_piece_rgb),
        ("black_piece", v.black_piece_rgb),
        ("table", v.table_rgb),
    ):
        assert [float(x) for x in materials[name].split()[:3]] == pytest.approx(list(rgb), abs=1e-4)  # type: ignore[union-attr]
    texture = next(root.iter("texture"))
    assert [float(x) for x in texture.get("rgb1").split()] == pytest.approx(list(v.board_rgb[0]), abs=1e-4)  # type: ignore[union-attr]
    assert [float(x) for x in texture.get("rgb2").split()] == pytest.approx(list(v.board_rgb[1]), abs=1e-4)  # type: ignore[union-attr]


def test_randomized_scene_compiles_with_cylinders_and_with_staunton_meshes() -> None:
    mujoco = pytest.importorskip("mujoco")
    v = randomize_variation(random.Random(6), (0.0,) * 6)
    cell = randomize_cell(SimCell(), random.Random(6))
    model = mujoco.MjModel.from_xml_string(scene_xml(cell, SimPosition(chess.Board()), **v.scene_kwargs()))
    pawn = model.body("p_e2_P")
    assert float(model.body_mass[pawn.id]) == pytest.approx(v.piece_mass, rel=1e-4)
    assert model.camera("top").fovy[0] == pytest.approx(cell.top_camera.fovy, abs=0.01)  # written to 0.01


def test_world_applies_the_variation(tmp_path: Path) -> None:
    from tests.sim_helpers import arm_assets_or_skip

    arm_assets_or_skip()
    from lelab.sim.world import PARK, ChessWorld

    position = SimPosition(chess.Board())
    plain = ChessWorld(SimCell(), position, render=False)
    v = randomize_variation(random.Random(8), PARK)
    world = ChessWorld(SimCell(), position, render=False, variation=v)
    ids = world.actuators
    np.testing.assert_allclose(
        world.model.actuator_gainprm[ids, 0], plain.model.actuator_gainprm[ids, 0] * v.actuator_gain
    )
    np.testing.assert_allclose(
        world.model.actuator_biasprm[ids, 1], plain.model.actuator_biasprm[ids, 1] * v.actuator_gain
    )
    np.testing.assert_allclose(plain.joint_state(), np.array(PARK), atol=1e-3)
    assert not np.allclose(world.joint_state(), plain.joint_state(), atol=1e-3)
    # shoulder_lift sits on its joint limit at PARK, so a draw past it is clipped, as any command is
    start = world.from_q(world.to_ctrl(v.initial_joints))
    np.testing.assert_allclose(world.joint_state(), start, atol=1e-3)
    assert world.model.body_mass[world.body_id("p_e2_P")] == pytest.approx(v.piece_mass, rel=1e-4)
