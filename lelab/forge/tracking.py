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

import logging
import os
from pathlib import Path

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE

from lelab.forge.registry import collection_name, episode_counts, registry_path
from lelab.utils.forge_config import forge_configured, load_forge_config
from lerobot.utils.constants import PRETRAINED_MODEL_DIR

logger = logging.getLogger(__name__)


def log_dataset_if_configured(dataset_dir: Path, repo_id: str) -> str | None:
    cfg = load_forge_config()
    if not forge_configured(cfg):
        return None
    try:
        import wandb

        counts = episode_counts(dataset_dir)
        run = wandb.init(
            entity=cfg.entity,
            project=cfg.project,
            job_type="record",
            name=f"dataset-{repo_id.replace('/', '-')}",
        )
        try:
            artifact = wandb.Artifact(
                "so101-kitting",
                type="dataset",
                metadata={"repo_id": repo_id, "episode_counts": counts},
            )
            artifact.add_dir(str(dataset_dir))
            run.log_artifact(artifact)
        finally:
            run.finish()
        return run.url
    except Exception:
        logger.exception("Dataset artifact was not logged for %s", repo_id)
        return None


class WandbLogger:
    def __init__(self, cfg) -> None:
        import wandb

        self._wandb = wandb
        self._cfg = cfg
        self._run = wandb.init(
            entity=load_forge_config().entity,
            project=cfg.trackio.project,
            job_type="train",
            config=cfg.to_dict(),
        )
        self._last_artifact = None
        logger.info("Track this run --> %s", self._run.url)

    def log_policy(self, checkpoint_dir) -> None:
        try:
            weights_dir = Path(checkpoint_dir) / PRETRAINED_MODEL_DIR
            adapter = weights_dir / "adapter_model.safetensors"
            standard = weights_dir / SAFETENSORS_SINGLE_FILE
            if not adapter.exists() and not standard.exists():
                logger.warning("No model weights in %s", weights_dir)
                return
            artifact = self._wandb.Artifact(f"policy-{Path(checkpoint_dir).name}", type="model")
            artifact.add_dir(str(weights_dir))
            self._run.log_artifact(artifact)
            self._last_artifact = artifact
        except Exception:
            logger.exception("Model artifact was not logged for %s", checkpoint_dir)

    def log_dict(self, d, step=None, mode="train", custom_step_key=None) -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")
        payload = {}
        for key, value in d.items():
            if isinstance(value, (int, float, str)):
                payload[f"{mode}/{key}"] = value
        if not payload:
            return
        if custom_step_key is not None:
            self._run.log(payload)
        else:
            self._run.log(payload, step=step)

    def log_video(self, video_path, step, mode="train") -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        video = self._wandb.Video(video_path, caption=f"{mode} step {step}")
        self._run.log({f"{mode}/video": video}, step=step)

    def finish(self) -> None:
        try:
            self._link_last_checkpoint()
        except Exception:
            logger.exception("Checkpoint was not linked into the Registry")
        finally:
            self._run.finish()

    def _link_last_checkpoint(self) -> None:
        if self._last_artifact is None:
            return
        cfg = load_forge_config()
        if not forge_configured(cfg):
            return
        skill = os.environ.get("FORGE_SKILL")
        skill = skill.strip() if skill else None
        try:
            collection = collection_name(self._cfg.policy.type, skill or None)
        except ValueError as exc:
            logger.warning("%s", exc)
            return
        self._run.link_artifact(
            self._last_artifact,
            registry_path(cfg.entity, collection),
            aliases=["candidate"],
        )
