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

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Repo root when editable; ignored if the file is missing (wheel install).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _REPO_ROOT / ".env"
# Load once at import so tests can monkeypatch after import without .env refilling.
load_dotenv(_ENV_FILE, override=False)

# Keys the Setup form may write. Order matches `.env.template`.
_FORGE_ENV_KEYS = (
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
    "CWSANDBOX_API_KEY",
    "CWSANDBOX_TRAIN_IMAGE",
)


@dataclass(frozen=True)
class ForgeConfig:
    api_key: str | None
    entity: str | None
    project: str
    sandbox_api_key: str | None = None
    sandbox_image: str | None = None
    sandbox_lifetime_seconds: int = 14400
    sandbox_cpu: str = "8"
    sandbox_memory: str = "32Gi"
    sandbox_volume_size: str = "100Gi"
    hf_token: str | None = None
    judge_model: str = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B"
    planner_model: str = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B"


def _blank_to_none(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return value.strip()


def _positive_int(value: str | None, default: int) -> int:
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value.strip())
    except ValueError:
        return default
    if parsed <= 0 or parsed > 2_592_000:
        return default
    return parsed


def load_forge_config() -> ForgeConfig:
    return ForgeConfig(
        api_key=_blank_to_none(os.environ.get("WANDB_API_KEY")),
        entity=_blank_to_none(os.environ.get("WANDB_ENTITY")),
        project=_blank_to_none(os.environ.get("WANDB_PROJECT")) or "so101-kitting",
        sandbox_api_key=_blank_to_none(os.environ.get("CWSANDBOX_API_KEY")),
        sandbox_image=_blank_to_none(os.environ.get("CWSANDBOX_TRAIN_IMAGE")),
        sandbox_lifetime_seconds=_positive_int(os.environ.get("CWSANDBOX_MAX_LIFETIME_SECONDS"), 14400),
        hf_token=_blank_to_none(os.environ.get("HF_TOKEN"))
        or _blank_to_none(os.environ.get("HUGGING_FACE_HUB_TOKEN")),
        judge_model=_blank_to_none(os.environ.get("FORGE_JUDGE_MODEL"))
        or "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B",
        planner_model=_blank_to_none(os.environ.get("FORGE_PLANNER_MODEL"))
        or "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B",
    )


def forge_configured(cfg: ForgeConfig) -> bool:
    return cfg.api_key is not None and cfg.entity is not None


def sandbox_configured(cfg: ForgeConfig) -> bool:
    return cfg.sandbox_api_key is not None and cfg.sandbox_image is not None


def forge_status_dict(cfg: ForgeConfig | None = None) -> dict[str, Any]:
    """Public Forge status — never includes secret values."""
    resolved = cfg if cfg is not None else load_forge_config()
    registry_block: dict[str, Any] = {"ready": False, "collections": []}
    if forge_configured(resolved):
        try:
            from lelab.forge.registry import RegistryStore

            collections = RegistryStore().list_collections(resolved.entity)
            registry_block = {"ready": True, "collections": collections}
        except Exception:
            registry_block = {"ready": False, "collections": []}

    return {
        "forge_configured": forge_configured(resolved),
        "entity": resolved.entity,
        "project": resolved.project,
        "sandbox_configured": sandbox_configured(resolved),
        "sandbox_image": resolved.sandbox_image,
        "registry": registry_block,
        # Per-field presence only — secrets stay out of the payload.
        "fields": {
            "api_key": resolved.api_key is not None,
            "entity": resolved.entity is not None,
            "project": _blank_to_none(os.environ.get("WANDB_PROJECT")) is not None,
            "sandbox_api_key": resolved.sandbox_api_key is not None,
            "sandbox_image": resolved.sandbox_image is not None,
        },
    }


def env_file_path() -> Path:
    return _ENV_FILE


def upsert_dotenv(path: Path, updates: dict[str, str]) -> None:
    """Replace or append KEY=value lines. Preserves comments and unrelated keys."""
    if not updates:
        return
    for key, value in updates.items():
        if "\n" in value or "\r" in value:
            raise ValueError(f"{key} must be a single-line value")
    lines = path.read_text().splitlines() if path.exists() else []
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(line)
    for key, value in updates.items():
        if key not in seen:
            out.append(f"{key}={value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + ("\n" if out else ""))
    try:
        path.chmod(0o600)
    except OSError:
        pass


def save_forge_credentials(
    *,
    api_key: str | None = None,
    entity: str | None = None,
    project: str | None = None,
    sandbox_api_key: str | None = None,
    sandbox_image: str | None = None,
    env_path: Path | None = None,
) -> dict[str, Any]:
    """Persist non-blank credential fields to `.env` and the process environment.

    Blank / missing fields leave the existing value unchanged. Secrets are never
    returned in the status payload.
    """
    updates: dict[str, str] = {}
    mapping = {
        "WANDB_API_KEY": api_key,
        "WANDB_ENTITY": entity,
        "WANDB_PROJECT": project,
        "CWSANDBOX_API_KEY": sandbox_api_key,
        "CWSANDBOX_TRAIN_IMAGE": sandbox_image,
    }
    for key, raw in mapping.items():
        value = _blank_to_none(raw)
        if value is None:
            continue
        if key not in _FORGE_ENV_KEYS:
            continue
        updates[key] = value

    if not updates:
        raise ValueError("Provide at least one Forge credential field to save")

    target = env_path if env_path is not None else env_file_path()
    upsert_dotenv(target, updates)
    for key, value in updates.items():
        os.environ[key] = value
    return forge_status_dict()
