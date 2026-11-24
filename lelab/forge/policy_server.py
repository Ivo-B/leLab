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

import collections
import contextlib
import json
import queue
import re
import shlex
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from ..utils.forge_config import ForgeConfig
from .certificates import issue_certificates
from .sandbox import SANDBOX_TAG, redact_secrets

POLICY_PORT = 7447
FAILURE_PREFIX = "Could not start the CoreWeave policy server: "
FAILURE_SUFFIX = " Close the evaluation and reopen it with the local policy host."
DEPLOYMENT = "so101-chess"
SEMANTICS = "so101-degrees-gripper-percent-v1"
ROBOT_TYPE = "so101_follower"
ZENOH_CONFIG_PATH = "/work/zenoh.json5"
POLICY_ROOT = "/work/policy"
READY_MARKER = "Policy server ready:"
TAIL_LINES = 20
# The last line of a Python traceback, e.g. `ValueError: ...` or `torch.OutOfMemoryError: ...`.
_TRACEBACK_TAIL = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Exit)\b(: .*)?$")
_DOWNLOAD_CODE = f'import sys, wandb; wandb.Api().artifact(sys.argv[1]).download(root="{POLICY_ROOT}")'
CLIENT_CA_PATH = "/tmp/lelab-policy-ca.pem"
JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


def policy_service() -> dict:
    return {
        "port": POLICY_PORT,
        "name": "policy",
        "visibility": "public",
        "endpoint": {"kind": "tls_passthrough"},
    }


def assigned_endpoint(addresses) -> str:
    for item in addresses:
        if item.name == "policy":
            return f"tls/{item.address}"
    raise ValueError("The Sandbox did not publish a policy endpoint")


def zenoh_server_config() -> str:
    return json.dumps(
        {
            "listen": {"endpoints": ["tls/0.0.0.0:7447"]},
            "transport": {
                "link": {
                    "tls": {
                        "root_ca_certificate": "/work/ca.pem",
                        "listen_private_key": "/work/key.pem",
                        "listen_certificate": "/work/cert.pem",
                        "enable_mtls": False,
                    }
                }
            },
        }
    )


def zenoh_client_config() -> str:
    return json.dumps(
        {
            "transport": {
                "link": {
                    "tls": {
                        "root_ca_certificate": CLIENT_CA_PATH,
                        "verify_name_on_connect": True,
                        "enable_mtls": False,
                    }
                }
            }
        }
    )


FeatureSpecDict = dict[str, Any]
_IMAGE_PREFIX = "observation.images."


def checkpoint_features(config: dict) -> tuple[list[FeatureSpecDict], FeatureSpecDict]:
    """The server's FeatureSpec mappings, read from a checkpoint's `config.json`.

    Images are stored channel-first (`[3, H, W]`); the robot sends them `(H, W, 3)`.
    """
    inputs = config.get("input_features")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError(
            "The checkpoint config has no input_features, so the policy server cannot be set up."
        )

    features: list[FeatureSpecDict] = []
    state: FeatureSpecDict | None = None
    cameras = 0
    for name, spec in inputs.items():
        kind = str(spec.get("type", "")).upper()
        shape = [int(dim) for dim in spec.get("shape", [])]
        if kind == "VISUAL":
            if len(shape) != 3 or shape[0] != 3:
                raise ValueError(f"The checkpoint camera {name} is not a 3-channel image.")
            camera = name.removeprefix(_IMAGE_PREFIX)
            features.append(
                {
                    "name": name,
                    "shape": [shape[1], shape[2], 3],
                    "dtype": "uint8",
                    "kind": "rgb",
                    "semantics": f"rgb-{camera}-v1",
                }
            )
            cameras += 1
        elif name == "observation.state":
            state = _vector_feature(name, shape)
    if cameras == 0:
        raise ValueError("The checkpoint has no camera input, so the policy server cannot be set up.")
    if state is not None:
        features.insert(0, state)

    outputs = config.get("output_features")
    action = outputs.get("action") if isinstance(outputs, dict) else None
    if not isinstance(action, dict):
        raise ValueError("The checkpoint config has no action output, so the policy server cannot be set up.")
    return features, _vector_feature("action", [int(dim) for dim in action.get("shape", [])])


def _vector_feature(name: str, shape: list[int]) -> FeatureSpecDict:
    feature: FeatureSpecDict = {"name": name, "shape": shape, "dtype": "float32"}
    if shape == [len(JOINT_NAMES)]:
        feature["names"] = list(JOINT_NAMES)
    feature["semantics"] = SEMANTICS
    return feature


def _feature_yaml(feature: FeatureSpecDict, *, sequence_item: bool = False) -> list[str]:
    """One FeatureSpec mapping. `sequence_item` emits a YAML list entry."""
    first = "  - " if sequence_item else "  "
    rest = "    " if sequence_item else "  "
    lines = [f"{first}name: {json.dumps(feature['name'])}"]
    for key in ("shape", "dtype", "kind", "names", "semantics"):
        if key in feature:
            lines.append(f"{rest}{key}: {json.dumps(feature[key])}")
    return lines


def policy_server_yaml(model_path: str, features: list[FeatureSpecDict], action: FeatureSpecDict) -> str:
    # JSON strings are valid YAML scalars, so json.dumps quotes values safely.
    return "\n".join(
        [
            f"deployment: {json.dumps(DEPLOYMENT)}",
            f"semantics: {json.dumps(SEMANTICS)}",
            f"robot_type: {json.dumps(ROBOT_TYPE)}",
            "model:",
            f"  repo_or_path: {json.dumps(model_path)}",
            '  device: "cuda"',
            "zenoh:",
            f"  config_file: {json.dumps(ZENOH_CONFIG_PATH)}",
            '  listen_endpoints: ["tls/0.0.0.0:7447"]',
            '  mode: "peer"',
            "features:",
            *(line for feature in features for line in _feature_yaml(feature, sequence_item=True)),
            "action_feature:",
            *_feature_yaml(action),
            "",
        ]
    )


def policy_server_argv() -> list[str]:
    return [
        "lerobot-policy-server",
        "--config_path=/work/server.yaml",
        "--model.device=cuda",
    ]


def _shell(command: str) -> list[str]:
    """LeRobot logs to stderr and the Sandbox exec stream is stdout, so merge them."""
    return ["sh", "-c", f"{command} 2>&1"]


class PolicyHost:
    """Runs the policy server in a CoreWeave Sandbox.

    `client.create(kwargs)` returns a handle with `write_file`, `exec`, `stop`
    and `service_addresses`. `issuer(hostname)` returns `ca.pem`, `key.pem` and
    `cert.pem` as bytes for the Sandbox hostname, which is only known after
    `create`. `ca.pem` is also written to `client_ca_path` for the robot side.
    """

    def __init__(
        self,
        client: Any,
        cfg: ForgeConfig,
        issuer: Callable[[str], dict[str, bytes]] = issue_certificates,
    ) -> None:
        self._client = client
        self._cfg = cfg
        self._issuer = issuer
        self._certificates: dict[str, bytes] = {}
        self._handle: Any = None
        self._server_process: Any = None

    @property
    def client_ca_path(self) -> str:
        return CLIENT_CA_PATH

    def start(
        self,
        artifact_path: str,
        model_subdir: str,
        features: list[FeatureSpecDict],
        action: FeatureSpecDict,
    ) -> str:
        """Create the Sandbox, download the checkpoint into it and launch the policy server.

        Returns the endpoint as soon as the server process is launched; `wait_until_ready`
        reports when it accepts a robot.
        """
        handle: Any = None
        try:
            handle = self._client.create(self._create_kwargs())
            self._handle = handle
            endpoint = assigned_endpoint(handle.service_addresses)
            hostname = endpoint.removeprefix("tls/").rsplit(":", 1)[0]
            self._certificates = self._issuer(hostname)
            for name in ("ca.pem", "key.pem", "cert.pem"):
                handle.write_file(f"/work/{name}", self._certificates[name])
            Path(CLIENT_CA_PATH).write_bytes(self._certificates["ca.pem"])
            handle.write_file(ZENOH_CONFIG_PATH, zenoh_server_config().encode())
            self._download(handle, artifact_path)
            model_path = f"{POLICY_ROOT}/{model_subdir}" if model_subdir else POLICY_ROOT
            handle.write_file("/work/server.yaml", policy_server_yaml(model_path, features, action).encode())
            self._server_process = handle.exec(_shell(f"exec {shlex.join(policy_server_argv())}"))
            return endpoint
        except Exception as exc:
            reason = self._redact(str(exc))
            if handle is not None:
                self._stop_quietly()
            raise RuntimeError(f"{FAILURE_PREFIX}{reason}.{FAILURE_SUFFIX}") from None

    def wait_until_ready(self, timeout_s: float) -> None:
        """Block until the server logs that it is ready, or raise the redacted failure sentence."""
        process = self._server_process
        if process is None:
            raise RuntimeError(f"{FAILURE_PREFIX}the policy server was not started.{FAILURE_SUFFIX}")
        tail: collections.deque[str] = collections.deque(maxlen=TAIL_LINES)
        try:
            reason = self._await_ready(process, timeout_s, tail)
        except Exception as exc:
            reason = f"reading the server output failed: {exc}"
        if reason is None:
            return
        self._stop_quietly()
        output = self._redact("\n".join(tail))
        detail = f" Last output:\n{output}" if output else ""
        raise RuntimeError(f"{FAILURE_PREFIX}{self._redact(reason)}.{detail}{FAILURE_SUFFIX}") from None

    def stop(self) -> None:
        handle, self._handle = self._handle, None
        self._server_process = None
        if handle is not None:
            handle.stop()

    def _download(self, handle: Any, artifact_path: str) -> None:
        command = f"python -c {shlex.quote(_DOWNLOAD_CODE)} {shlex.quote(artifact_path)}"
        process = handle.exec(_shell(command))
        tail: collections.deque[str] = collections.deque(maxlen=TAIL_LINES)
        for line in process.stdout:
            tail.append(line.rstrip("\n"))
        code = process.returncode
        if not isinstance(code, int):
            code = 1
        if code != 0:
            raise RuntimeError(
                f"the checkpoint download failed (exit {code}). Last output:\n" + "\n".join(tail)
            )

    @staticmethod
    def _await_ready(process: Any, timeout_s: float, tail: collections.deque[str]) -> str | None:
        """None when the ready line arrived, otherwise why the server will not become ready."""
        lines: queue.Queue[str | BaseException | None] = queue.Queue()

        def pump(stream: Iterable[str]) -> None:
            try:
                for line in stream:
                    lines.put(line)
            except Exception as exc:
                lines.put(exc)
            finally:
                lines.put(None)

        threading.Thread(
            target=pump, args=(process.stdout,), name="policy-server-output", daemon=True
        ).start()
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                item = lines.get(timeout=max(deadline - time.monotonic(), 0))
            except queue.Empty:
                return f"the policy server was not ready after {timeout_s:g} seconds"
            if item is None:
                return "the policy server stopped before it was ready"
            if isinstance(item, BaseException):
                return f"reading the server output failed: {item}"
            line = item.rstrip("\n")
            tail.append(line)
            if READY_MARKER in line:
                return None
            if _TRACEBACK_TAIL.match(line.strip()):
                return "the policy server failed while starting"

    def _create_kwargs(self) -> dict[str, object]:
        cfg = self._cfg
        env: dict[str, str] = {"PYTHONUNBUFFERED": "1"}
        if cfg.api_key is not None:
            env["WANDB_API_KEY"] = cfg.api_key
        if cfg.entity is not None:
            env["WANDB_ENTITY"] = cfg.entity
        return {
            "args": ["sleep", "infinity"],
            "container_image": cfg.sandbox_image,
            "placement_mode": "serverless",
            "tags": [SANDBOX_TAG],
            "max_lifetime_seconds": cfg.sandbox_lifetime_seconds,
            "resources": {"cpu": cfg.sandbox_cpu, "memory": cfg.sandbox_memory, "gpu": 1},
            "volumes": [{"name": "work", "mount_path": "/work", "size": cfg.sandbox_volume_size}],
            "environment_variables": env,
            "services": [policy_service()],
        }

    def _redact(self, text: str) -> str:
        try:
            text = redact_secrets(text, self._cfg).replace("CWSANDBOX_API_KEY", "[redacted]")
            for blob in self._certificates.values():
                for needle in self._certificate_forms(blob):
                    text = text.replace(needle, "[redacted]")
            return text
        except Exception:
            return "[redacted]"

    @staticmethod
    def _certificate_forms(blob: bytes) -> list[str]:
        if not blob:
            return []
        forms = {repr(blob), repr(blob)[2:-1]}
        try:
            decoded = blob.decode()
        except UnicodeDecodeError:
            pass
        else:
            forms.update({decoded, decoded.strip(), repr(decoded)})
        return sorted((form for form in forms if form), key=len, reverse=True)

    def _stop_quietly(self) -> None:
        # The start error is what the caller needs; the Sandbox also expires on its own.
        with contextlib.suppress(Exception):
            self.stop()
