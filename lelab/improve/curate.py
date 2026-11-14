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

from statistics import median

IDLE_START_S = 1.0


def flag_episode(
    start_idle_s: float,
    frame_gaps_s: list[float],
    duration_s: float,
    peer_durations: list[float],
) -> list[str]:
    reasons: list[str] = []
    if start_idle_s > IDLE_START_S:
        reasons.append("idle_start")
    positive = [gap for gap in frame_gaps_s if gap > 0]
    if positive and any(gap > 2 * median(positive) for gap in positive):
        reasons.append("dropped_frames")
    if len(peer_durations) >= 4:
        middle = median(peer_durations)
        if duration_s < 0.5 * middle or duration_s > 1.5 * middle:
            reasons.append("outlier_length")
    return reasons
