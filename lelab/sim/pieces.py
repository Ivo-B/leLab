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

"""Staunton chess piece meshes for the simulator's visuals.

The meshes are clarkerubber's open-source Staunton set (MIT), fetched once at a pinned
commit into `config.SIM_ASSETS_PATH/staunton`. They are visual only: collisions stay
primitive cylinders (see `scene.py`).

The STL files are in millimetres with Y up, one piece standing on y = 0 with a circular
base centred at (`cx`, `cz`). The base centre and radius below were measured once from
the vertices at y = 0 at `PIECES_REVISION`; `piece_visual_xml` uses them to scale each
mesh so its base diameter equals the collision cylinder's, and to stand it upright and
centred on the piece body.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from lelab.utils import config

from .assets import SimUnavailable

PIECES_REPO = "https://github.com/clarkerubber/Staunton-Pieces"
PIECES_REVISION = "cef2c79445380c5bffec461a80c7c91caf89b3b2"
PIECES_RAW_URL = "https://raw.githubusercontent.com/clarkerubber/Staunton-Pieces"
PIECES_DIR = "staunton"
LICENSE_FILE = "LICENSE"
LICENSE_SHA256 = "9b704b1ddb3ffd8d9323fea402b71dbfd5649c26ecdeb3b84a5cc1a29326cb77"
DOWNLOAD_TIMEOUT = 60  # seconds


@dataclass(frozen=True)
class PieceMesh:
    file: str  # local file name under `PIECES_DIR`
    upstream: str  # path inside the repo at `PIECES_REVISION`
    sha256: str
    cx: float  # base circle centre, mm, in the mesh's x
    cz: float  # ... and z (y is up)
    base_radius: float  # mm


PIECE_MESHES: dict[str, PieceMesh] = {
    "p": PieceMesh(
        "pawn.stl",
        "Source/Staunton/Pawn/Pawn.STL",
        "5af650f34b7547d6e6b53834bbe6753e0593bb9d071cd8666996e8e7a2446a4e",
        12.0,
        12.0,
        12.0,
    ),
    "n": PieceMesh(
        "knight.stl",
        "Source/Staunton/Knight/Knight.STL",
        "1d0b856d369d4a00cab0666780b3b5f344b84ed93ef710a333675c267fbd8e2b",
        15.0,
        15.749,
        15.0,
    ),
    "b": PieceMesh(
        "bishop.stl",
        "Source/Staunton/Bishop/Bishop.STL",
        "584c636d52f1d203a753bc0c7c57f70e457e56cb4d1e42b7a21e8bcf559c7732",
        15.0,
        14.983,
        15.017,
    ),
    "r": PieceMesh(
        "rook.stl",
        "Source/Staunton/Rook/Rook.STL",
        "520bab1d3b38eb4c6ef2df8e3ddf6a69acda47455b35b10f2ee3b479aa2d3b06",
        15.0,
        14.985,
        15.015,
    ),
    "q": PieceMesh(
        "queen.stl",
        "Source/Staunton/Queen/Queen.STL",
        "99aca6b4e40bb7632a2bb248bbd9829bfd1f4bb2c96977c9d3e9bf38bcd6b81f",
        80.9,
        15.0,
        15.0,
    ),
    "k": PieceMesh(
        "king.stl",
        "Source/Staunton/King/King.STL",
        "bd271ed9164d8b2cb1826cfd83a9fb1d4b06bed78caf9d1c0217d8d09c8e7ff5",
        15.0,
        15.0,
        15.0,
    ),
}


def _mesh(symbol: str) -> PieceMesh:
    try:
        return PIECE_MESHES[symbol.lower()]
    except KeyError:
        raise ValueError(f"unknown piece symbol {symbol!r}; expected one of {sorted(PIECE_MESHES)}") from None


def pieces_dir() -> Path:
    return Path(config.SIM_ASSETS_PATH) / PIECES_DIR


def piece_mesh_path(symbol: str) -> Path:
    """Where the mesh of `symbol` (`pnbrqk`, either case) lives; it may not be downloaded yet."""
    return pieces_dir() / _mesh(symbol).file


def _ssl_context() -> ssl.SSLContext:
    """The OS trust store when `truststore` is installed (huggingface_hub pulls it in), as the arm
    download does; behind a TLS-inspecting proxy the bundled CA list alone fails verification."""
    if importlib.util.find_spec("truststore") is not None:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return ssl.create_default_context()


def _fetch(path: str, sha256: str) -> bytes:
    url = f"{PIECES_RAW_URL}/{PIECES_REVISION}/{path}"
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT, context=_ssl_context()) as response:  # nosec B310
        data = response.read()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"{path} does not match its pinned checksum")
    return data


def _write(target: Path, data: bytes) -> None:
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(data)
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)


def ensure_piece_meshes() -> Path:
    """Download the six meshes (and the licence) once; return the directory that holds them."""
    root = pieces_dir()
    wanted = [(root / m.file, m.upstream, m.sha256) for m in PIECE_MESHES.values()]
    wanted.append((root / LICENSE_FILE, LICENSE_FILE, LICENSE_SHA256))
    missing = [item for item in wanted if not item[0].is_file()]
    if not missing:
        return root
    try:
        root.mkdir(parents=True, exist_ok=True)
        for target, upstream, sha256 in missing:
            _write(target, _fetch(upstream, sha256))
    except Exception as e:
        raise SimUnavailable(
            f"Could not download the Staunton piece meshes from {PIECES_REPO}@{PIECES_REVISION[:8]}: {e}"
        ) from e
    return root


def mesh_scale(symbol: str, radius: float) -> float:
    """Factor from the mesh's millimetres to the metres that give a base radius of `radius`."""
    return radius / _mesh(symbol).base_radius


def piece_mesh_assets_xml(radii: dict[str, float]) -> str:
    """`<mesh>` assets for the six pieces; `radii` maps `pnbrqk` to the collision radius in metres.

    Each asset is named after its file, which is what `piece_visual_xml` refers to.
    """
    lines = []
    for symbol, mesh in PIECE_MESHES.items():
        if symbol not in radii:
            continue
        s = mesh_scale(symbol, radii[symbol])
        lines.append(f'<mesh name="{mesh.file}" file="{mesh.file}" scale="{s:.8f} {s:.8f} {s:.8f}"/>')
    return "\n    ".join(lines)


def piece_visual_xml(
    name: str,
    symbol: str,
    mesh_file: str | None,
    *,
    radius: float,
    height: float,
    material: str | None = None,
) -> str:
    """The geoms of one piece body: a collision cylinder, plus a visual mesh when `mesh_file` is set.

    The cylinder is `radius` x `height`, centred on the body origin. The mesh geom keeps the
    mesh asset's base diameter at `2 * radius` (the asset is scaled by `piece_mesh_assets_xml`),
    stands the base on the cylinder's bottom and centres it on the body axis. It carries no
    collision and no mass. Without a mesh the cylinder is the piece's only, visible shape.
    """
    mat = material or ("white_piece" if symbol.isupper() else "black_piece")
    group = "" if mesh_file is None else 'group="3" '  # hidden when the mesh is what renders
    collision = (
        f'<geom name="{name}_collision" class="piece" type="cylinder" '
        f'size="{radius:.5f} {height / 2:.5f}" {group}material="{mat}"/>'
    )
    if mesh_file is None:
        return collision
    mesh = _mesh(symbol)
    s = mesh_scale(symbol, radius)
    # Rotate +90 deg about x (mesh Y-up -> Z-up: (x, y, z) -> (x, -z, y)), then move the base
    # centre to the body axis and the base to the cylinder's bottom face.
    pos = f"{-s * mesh.cx:.6f} {s * mesh.cz:.6f} {-height / 2:.6f}"
    return (
        collision + f'<geom name="{name}_visual" class="piece_visual" type="mesh" mesh="{mesh_file}" '
        f'pos="{pos}" quat="0.70710678 0.70710678 0 0" material="{mat}"/>'
    )
