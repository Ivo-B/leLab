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

"""The real `cwsandbox` client behind `SandboxJobRunner`, plus the startup sweep.

`cwsandbox` is imported inside each function so that importing LeLab, or
running with no Sandbox settings, never needs the SDK.
"""

from __future__ import annotations

import logging
from typing import Any

from ..forge.sandbox import SANDBOX_TAG, redact_secrets
from ..utils.forge_config import load_forge_config, sandbox_configured

logger = logging.getLogger(__name__)


class SandboxHandle:
    """The slice of a `cwsandbox.Sandbox` that `SandboxJobRunner` uses."""

    def __init__(self, sandbox: Any) -> None:
        self._sandbox = sandbox

    def write_file(self, path: str, contents: bytes) -> None:
        self._sandbox.write_file(path, contents).result()

    def exec(self, command: list[str]) -> Any:
        return self._sandbox.exec(command)

    def stop(self) -> None:
        self._sandbox.stop(missing_ok=True).result()


class CwsandboxClient:
    def create(self, kwargs: dict[str, Any]) -> SandboxHandle:
        from cwsandbox import AuthStrategy, Sandbox, ScratchVolumeOptions

        sandbox = Sandbox.run(
            *kwargs["args"],
            auth=AuthStrategy.COREWEAVE_API_KEY,
            container_image=kwargs["container_image"],
            placement_mode=kwargs["placement_mode"],
            tags=kwargs["tags"],
            max_lifetime_seconds=kwargs["max_lifetime_seconds"],
            resources=kwargs["resources"],
            volumes=[ScratchVolumeOptions(**volume) for volume in kwargs["volumes"]],
            environment_variables=kwargs["environment_variables"],
        ).wait()
        return SandboxHandle(sandbox)


def sweep_lelab_sandboxes() -> int:
    """Stop every Sandbox tagged `lelab`. Returns how many were stopped.

    A stop failure is logged (redacted) and the sweep moves on. Nothing here
    logs the environment.
    """
    cfg = load_forge_config()
    if not sandbox_configured(cfg):
        return 0
    stopped = 0
    try:
        from cwsandbox import AuthStrategy, Sandbox

        sandboxes = Sandbox.list(tags=[SANDBOX_TAG], auth=AuthStrategy.COREWEAVE_API_KEY).result()
    except Exception as exc:
        logger.warning("Could not list CoreWeave Sandboxes: %s", redact_secrets(str(exc), cfg))
        return 0
    for sandbox in sandboxes:
        try:
            sandbox.stop(missing_ok=True).result()
            stopped += 1
        except Exception as exc:
            logger.warning("Could not stop a CoreWeave Sandbox: %s", redact_secrets(str(exc), cfg))
    return stopped
