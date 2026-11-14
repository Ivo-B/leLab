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

from pathlib import Path


def test_notebook_downloads_the_dataset_artifact_and_flags_episodes() -> None:
    text = Path("notebooks/curate_dataset.py").read_text()
    assert "flag_episode" in text
    assert 'use_artifact("so101-kitting:latest")' in text
    assert "artifact.download()" in text
    assert "temporary" in text.lower()
    assert "wandb.init(" not in text
