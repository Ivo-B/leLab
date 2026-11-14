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

import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Curate the SO-101 kitting dataset

        Files written inside this notebook sandbox are temporary. They vanish when
        the sandbox stops, so keep anything you need as a W&B artifact.
        CoreWeave Notebooks provides marimo and does not include LeLab, so install
        this repo in the notebook environment before the `flag_episode` import.
        """
    )
    return


@app.cell
def _():
    from lelab.improve.curate import flag_episode

    return (flag_episode,)


@app.cell
def _():
    def download_dataset(run):
        """Download the latest kitting dataset artifact for an open W&B run."""
        artifact = run.use_artifact("so101-kitting:latest")
        return artifact.download()

    return (download_dataset,)


@app.cell
def _(flag_episode):
    def review_episode(start_idle_s, frame_gaps_s, duration_s, peer_durations):
        """Return the reasons an episode should be flagged, if any."""
        return flag_episode(start_idle_s, frame_gaps_s, duration_s, peer_durations)

    return (review_episode,)


if __name__ == "__main__":
    app.run()
