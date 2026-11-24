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

"""A JSON file of settings keyed by camera name: the colour gain and the lens model use one each.

Cameras are keyed by the name the OS and the browser show, because OpenCV indices move when cameras are
replugged. Entries that fail validation are skipped on load, so one bad entry never hides the others.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Mapping
from typing import Any

logger = logging.getLogger(__name__)


def clean_name(name: str, error: type[ValueError]) -> str:
    """Camera names come from the OS with stray spaces (``"Innomaker-U20CAM-720P  "``); compare without them."""
    cleaned = name.strip()
    if not cleaned:
        raise error("The camera name is empty")
    return cleaned


class CameraStore[V]:
    def __init__(
        self,
        path: Callable[[], str],
        validate: Callable[[Mapping[str, Any]], V],
        normalize: Callable[[str], str],
        noun: str,
    ) -> None:
        # ``path`` is called on every access so tests and ``lelab.utils.config`` can redirect the file.
        self._path = path
        self._validate = validate
        self._normalize = normalize
        self._noun = noun

    def load(self) -> dict[str, V]:
        path = self._path()
        if not os.path.isfile(path):
            return {}
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            logger.warning("Ignoring unreadable camera %s file %s", self._noun, path)
            return {}
        if not isinstance(data, dict):
            return {}
        saved: dict[str, V] = {}
        for name, values in data.items():
            if not isinstance(values, dict):
                continue
            try:
                saved[self._normalize(str(name))] = self._validate(values)
            except ValueError:
                logger.warning("Ignoring invalid %s for '%s' in %s", self._noun, name, path)
        return saved

    def _write(self, saved: Mapping[str, V]) -> None:
        path = self._path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(saved, f, indent=2, sort_keys=True)
        os.replace(tmp, path)

    def save(self, name: str, values: Mapping[str, Any]) -> V:
        key = self._normalize(name)
        checked = self._validate(values)
        saved = self.load()
        saved[key] = checked
        self._write(saved)
        return checked

    def forget(self, name: str) -> bool:
        key = self._normalize(name)
        saved = self.load()
        if key not in saved:
            return False
        del saved[key]
        self._write(saved)
        return True
