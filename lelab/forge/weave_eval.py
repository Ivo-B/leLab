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

from dataclasses import dataclass


@dataclass(frozen=True)
class TrialRow:
    skill: str
    stage: str | None
    verdict_source: str
    duration_s: float
    chunk_latency_s: float | None
    policy_host: str
    top_video: str | None
    side_video: str | None


class WeaveTrialLog:
    def __init__(self, logger, entity: str, project: str) -> None:
        self._logger = logger
        self._entity = entity
        self._project = project
        self._count = 0

    def log_trial(self, row: TrialRow) -> None:
        prediction = self._logger.log_prediction(
            inputs={"skill": row.skill, "policy_host": row.policy_host},
            output={
                "stage": row.stage,
                "verdict_source": row.verdict_source,
                "policy_host": row.policy_host,
                "duration_s": row.duration_s,
                "chunk_latency_s": row.chunk_latency_s,
                "top_video": row.top_video,
                "side_video": row.side_video,
            },
        )
        prediction.log_score("needs_review", 1 if row.stage is None else 0)
        prediction.finish()
        self._count += 1

    def finish(self) -> str:
        self._logger.log_summary({"trials": self._count})
        return f"https://wandb.ai/{self._entity}/{self._project}/weave"
