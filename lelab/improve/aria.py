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

AUTOMATION_NAME = "so101-chess evaluation"
AUTOMATION_EVENT = "A run reaches FINISHED"


def aria_prompt() -> str:
    return (
        "Look at the Weave evaluation so101-chess-sim-eval in this project. "
        "Each trial is one transfer of a chess piece, with a transfer class (near, mid, far, bin, park), "
        "a domain (sim or real) and an outcome: success, approach, grasp, drop, place, topple or disturbed. "
        "Name the transfer class and the failure outcome that fail most, separately for sim and real, "
        "and state the gap between sim and real success per transfer class. "
        "Propose the next experiment: which transfer classes need real correction episodes, "
        "whether the sim randomization or the sim cell should change, "
        "and whether to retrain ACT or SmolVLA. "
        "Do not change an alias and do not launch a job."
    )
