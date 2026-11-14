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

AUTOMATION_NAME = "so101-kitting evaluation"
AUTOMATION_EVENT = "A run reaches FINISHED"


def aria_prompt() -> str:
    return (
        "Look at the Weave evaluation that just finished in this project. "
        "Name the skill and the stage where trials failed. "
        "The stages are started, reached, grasped, and delivered. "
        "A trial with no stage needs review and counts as a failure. "
        "Propose the next experiment: which skill needs more demonstrations, "
        "and whether to retrain SmolVLA or the ACT policy for that skill. "
        "Do not change an alias and do not launch a job."
    )
