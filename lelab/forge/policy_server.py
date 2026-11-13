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

import contextlib
import json
from typing import Any

from ..utils.forge_config import ForgeConfig
from .sandbox import SANDBOX_TAG, redact_secrets

POLICY_PORT = 7447
FAILURE_PREFIX = "Could not start the CoreWeave policy server: "
FAILURE_SUFFIX = " Use the local policy host instead."
DEPLOYMENT = "so101-kitting"
SEMANTICS = "so101-degrees-gripper-percent-v1"
ROBOT_TYPE = "so101_follower"
ZENOH_CONFIG_PATH = "/work/zenoh.json5"


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
                        "root_ca_certificate": "/tmp/lelab-policy-ca.pem",
                        "verify_name_on_connect": True,
                        "enable_mtls": False,
                    }
                }
            }
        }
    )


def policy_server_yaml(checkpoint: str) -> str:
    # JSON strings are valid YAML scalars, so json.dumps quotes values safely.
    return "\n".join(
        [
            f"deployment: {json.dumps(DEPLOYMENT)}",
            f"semantics: {json.dumps(SEMANTICS)}",
            f"robot_type: {json.dumps(ROBOT_TYPE)}",
            "model:",
            f"  repo_or_path: {json.dumps(checkpoint)}",
            '  device: "cuda"',
            "zenoh:",
            f"  config_file: {json.dumps(ZENOH_CONFIG_PATH)}",
            '  listen_endpoints: ["tls/0.0.0.0:7447"]',
            '  mode: "peer"',
            "",
        ]
    )


def policy_server_argv() -> list[str]:
    return [
        "lerobot-policy-server",
        "--config_path=/work/server.yaml",
        "--model.device=cuda",
    ]


class PolicyHost:
    """Runs the policy server in a CoreWeave Sandbox.

    `client.create(kwargs)` returns a handle with `write_file`, `exec`, `stop`
    and `service_addresses`. `certificates` maps `ca.pem`, `key.pem` and
    `cert.pem` to their bytes.
    """

    def __init__(self, client: Any, cfg: ForgeConfig, certificates: dict[str, bytes]) -> None:
        self._client = client
        self._cfg = cfg
        self._certificates = certificates
        self._handle: Any = None

    def start(self, checkpoint: str) -> str:
        handle: Any = None
        try:
            handle = self._client.create(self._create_kwargs())
            self._handle = handle
            for name in ("ca.pem", "key.pem", "cert.pem"):
                handle.write_file(f"/work/{name}", self._certificates[name])
            handle.write_file(ZENOH_CONFIG_PATH, zenoh_server_config().encode())
            handle.write_file("/work/server.yaml", policy_server_yaml(checkpoint).encode())
            endpoint = assigned_endpoint(handle.service_addresses)
            handle.exec(policy_server_argv())
            return endpoint
        except Exception as exc:
            reason = self._redact(str(exc))
            if handle is not None:
                self._stop_quietly()
            raise RuntimeError(f"{FAILURE_PREFIX}{reason}.{FAILURE_SUFFIX}") from None

    def stop(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            handle.stop()

    def _create_kwargs(self) -> dict[str, object]:
        cfg = self._cfg
        return {
            "args": ["sleep", "infinity"],
            "container_image": cfg.sandbox_image,
            "placement_mode": "serverless",
            "tags": [SANDBOX_TAG],
            "max_lifetime_seconds": cfg.sandbox_lifetime_seconds,
            "resources": {"cpu": cfg.sandbox_cpu, "memory": cfg.sandbox_memory, "gpu": 1},
            "volumes": [{"name": "work", "mount_path": "/work", "size": cfg.sandbox_volume_size}],
            "environment_variables": {"PYTHONUNBUFFERED": "1"},
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
