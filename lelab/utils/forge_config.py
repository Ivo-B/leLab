import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ForgeConfig:
    api_key: str | None
    entity: str | None
    project: str


def _blank_to_none(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return value.strip()


def load_forge_config() -> ForgeConfig:
    return ForgeConfig(
        api_key=_blank_to_none(os.environ.get("WANDB_API_KEY")),
        entity=_blank_to_none(os.environ.get("WANDB_ENTITY")),
        project=_blank_to_none(os.environ.get("WANDB_PROJECT")) or "so101-kitting",
    )


def forge_configured(cfg: ForgeConfig) -> bool:
    return cfg.api_key is not None and cfg.entity is not None
