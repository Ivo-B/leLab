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
from pathlib import Path

from lelab.forge.registry import episode_counts
from lelab.utils.forge_config import forge_configured, load_forge_config

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
