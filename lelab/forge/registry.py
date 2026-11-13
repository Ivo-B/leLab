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

import re
from pathlib import Path

_SKILL_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def collection_name(policy_type: str, skill: str | None) -> str:
    if policy_type == "smolvla":
        return "so101-kitting-smolvla"
    if policy_type == "act":
        if skill is None or _SKILL_SLUG.fullmatch(skill) is None:
            raise ValueError("ACT registry linking needs a skill slug such as bolt, bracket, or gear")
        return f"so101-kitting-act-{skill}"
    raise ValueError(f"No registry collection for policy type {policy_type}")


def registry_path(entity: str, collection: str) -> str:
    return f"{entity}/wandb-registry-model/{collection}"


class RegistryStore:
    """W&B / CoreWeave Model Registry client. Inject ``api`` in tests."""

    KNOWN = (
        "so101-kitting-smolvla",
        "so101-kitting-act-bolt",
        "so101-kitting-act-bracket",
        "so101-kitting-act-gear",
    )

    def __init__(self, api=None):
        self._api = api

    def _client(self):
        if self._api is not None:
            return self._api

        import wandb

        from lelab.utils.forge_config import forge_configured, load_forge_config

        cfg = load_forge_config()
        if not forge_configured(cfg):
            raise ValueError("Registry needs WANDB_API_KEY and WANDB_ENTITY")
        return wandb.Api(api_key=cfg.api_key)

    def resolve(self, entity: str, collection: str, alias: str) -> str:
        return f"{registry_path(entity, collection)}:{alias}"

    def download(
        self,
        entity: str,
        collection: str,
        alias: str,
        root: Path | None = None,
    ) -> Path:
        api = self._client()
        path = self.resolve(entity, collection, alias)
        try:
            artifact = api.artifact(path)
        except Exception as exc:
            raise ValueError(f"Registry artifact is missing: {path}") from exc
        downloaded = Path(artifact.download(root=str(root) if root else None))
        return downloaded

    def promote(
        self,
        entity: str,
        collection: str,
        *,
        from_alias: str = "candidate",
        to_alias: str = "production",
    ) -> dict:
        api = self._client()
        path = self.resolve(entity, collection, from_alias)
        try:
            artifact = api.artifact(path)
        except Exception as exc:
            raise ValueError(f"Registry artifact is missing: {path}") from exc
        if to_alias not in artifact.aliases:
            artifact.aliases.append(to_alias)
            artifact.save()
        return {
            "collection": collection,
            "from_alias": from_alias,
            "to_alias": to_alias,
            "version": getattr(artifact, "version", None),
        }

    def list_collections(self, entity: str) -> list[dict]:
        api = self._client()
        rows = []
        for name in self.KNOWN:
            aliases: list[str] = []
            for probe in ("latest", "candidate", "production"):
                try:
                    artifact = api.artifact(self.resolve(entity, name, probe))
                    for alias in getattr(artifact, "aliases", []) or []:
                        if alias not in aliases:
                            aliases.append(alias)
                except Exception as exc:
                    message = str(exc).lower()
                    if not any(
                        marker in message
                        for marker in ("not found", "does not exist", "missing")
                    ):
                        raise
            rows.append({"name": name, "aliases": aliases, "linked": bool(aliases)})
        return rows


def checkpoint_dir_for(
    policy_family: str,
    skill: str | None,
    alias: str,
    store: RegistryStore | None = None,
) -> Path:
    """Download the Registry version behind ``alias`` and return its pretrained_model dir.

    Policy artifacts are logged from a checkpoint's ``pretrained_model`` directory, so the
    downloaded artifact root is the directory a rollout loads. SmolVLA has one collection;
    ACT has one per skill. A missing alias raises ``ValueError``.
    """
    from lelab.utils.forge_config import load_forge_config

    cfg = load_forge_config()
    collection = collection_name(policy_family, skill if policy_family == "act" else None)
    root = Path.home() / ".cache" / "huggingface" / "lerobot" / "forge-registry" / collection / alias
    return (store or RegistryStore()).download(cfg.entity, collection, alias, root=root)


def episode_counts(dataset_dir: Path) -> dict[str, int]:
    from lelab.episode_media import read_episode_index

    counts: dict[str, int] = {}
    for row in read_episode_index(dataset_dir):
        for task in row.tasks:
            counts[task] = counts.get(task, 0) + 1
    return counts
