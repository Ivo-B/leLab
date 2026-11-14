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

import pytest

from lelab.forge.registry import (
    RegistryStore,
    checkpoint_dir_for,
    collection_name,
    episode_counts,
    registry_path,
)


class FakeArtifact:
    def __init__(self, name: str, aliases: list[str], version: str = "v3"):
        self.name = name
        self.aliases = list(aliases)
        self.version = version
        self._saved = False

    def download(self, root=None):
        target = Path(root or ".") / "artifact"
        target.mkdir(parents=True, exist_ok=True)
        (target / "pretrained_model").mkdir(exist_ok=True)
        (target / "pretrained_model" / "config.json").write_text("{}")
        return str(target)

    def save(self):
        self._saved = True


class _Collection:
    def __init__(self, name: str) -> None:
        self.name = name


class _Registry:
    def __init__(self, names: list[str]) -> None:
        self._names = names

    def collections(self) -> list[_Collection]:
        return [_Collection(name) for name in self._names]


class FakeApi:
    def __init__(self, artifacts: dict[str, FakeArtifact]):
        self.artifacts = artifacts

    def artifact(self, name: str):
        if name not in self.artifacts:
            raise ValueError(f"artifact not found: {name}")
        return self.artifacts[name]

    def registry(self, name: str) -> _Registry:
        if name != "model":
            raise ValueError(f"unknown registry: {name}")
        names = sorted({artifact.name.rsplit(":", 1)[0].rsplit("/", 1)[-1] for artifact in self.artifacts.values()})
        return _Registry(names)

    def artifact_exists(self, name: str) -> bool:
        if name in self.artifacts:
            return True
        base, _, alias = name.rpartition(":")
        return any(
            artifact.name.rsplit(":", 1)[0] == base and alias in artifact.aliases
            for artifact in self.artifacts.values()
        )


def test_smolvla_has_one_collection() -> None:
    assert collection_name("smolvla", None) == "so101-kitting-smolvla"
    assert collection_name("smolvla", "bolt") == "so101-kitting-smolvla"


def test_act_collection_uses_the_skill_slug() -> None:
    assert collection_name("act", "bracket") == "so101-kitting-act-bracket"


def test_act_without_a_slug_is_rejected() -> None:
    with pytest.raises(ValueError, match="skill slug"):
        collection_name("act", None)
    with pytest.raises(ValueError, match="skill slug"):
        collection_name("act", "Pick the bolt")


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="groot"):
        collection_name("groot", "bolt")


def test_registry_path() -> None:
    assert registry_path("coreweave-demo", "so101-kitting-smolvla") == (
        "coreweave-demo/wandb-registry-model/so101-kitting-smolvla"
    )


def test_episode_counts_sum_tasks(tmp_path: Path, monkeypatch) -> None:
    from lelab import episode_media

    class Row:
        def __init__(self, tasks: tuple[str, ...]) -> None:
            self.tasks = tasks

    monkeypatch.setattr(
        episode_media,
        "read_episode_index",
        lambda directory: [Row(("pick the bolt",)), Row(("pick the bolt",)), Row(("pick the bracket",))],
    )
    assert episode_counts(tmp_path) == {"pick the bolt": 2, "pick the bracket": 1}


def test_resolve_builds_the_registry_alias_path():
    store = RegistryStore(api=FakeApi({}))
    assert store.resolve("acme", "so101-kitting-smolvla", "candidate") == (
        "acme/wandb-registry-model/so101-kitting-smolvla:candidate"
    )


def test_download_returns_the_pretrained_model_parent(tmp_path):
    art = FakeArtifact("acme/wandb-registry-model/so101-kitting-smolvla:candidate", ["candidate"])
    store = RegistryStore(api=FakeApi({art.name: art}))
    path = store.download("acme", "so101-kitting-smolvla", "candidate", root=tmp_path)
    assert (path / "pretrained_model" / "config.json").is_file()


def test_checkpoint_dir_for_returns_nested_pretrained_model(tmp_path, monkeypatch):
    downloaded = tmp_path / "artifact"
    pretrained_model = downloaded / "pretrained_model"
    pretrained_model.mkdir(parents=True)
    (pretrained_model / "config.json").write_text("{}")

    class Store:
        def download(self, entity, collection, alias, root=None):
            return downloaded

    monkeypatch.setattr(
        "lelab.utils.forge_config.load_forge_config",
        lambda: type("Config", (), {"entity": "acme"})(),
    )
    assert checkpoint_dir_for("smolvla", None, "candidate", store=Store()) == pretrained_model


def test_download_and_promote_reject_missing_artifacts():
    store = RegistryStore(api=FakeApi({}))

    with pytest.raises(ValueError, match="artifact not found"):
        store.download("acme", "so101-kitting-smolvla", "candidate")
    with pytest.raises(ValueError, match="artifact not found"):
        store.promote("acme", "so101-kitting-smolvla")


def test_promote_moves_production_onto_the_candidate_version(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "k")
    monkeypatch.setenv("WANDB_ENTITY", "acme")
    candidate = FakeArtifact(
        "acme/wandb-registry-model/so101-kitting-smolvla:candidate",
        ["candidate", "latest"],
        version="v3",
    )
    store = RegistryStore(api=FakeApi({candidate.name: candidate}))
    result = store.promote("acme", "so101-kitting-smolvla")
    assert result == {
        "collection": "so101-kitting-smolvla",
        "from_alias": "candidate",
        "to_alias": "production",
        "version": "v3",
    }
    assert "production" in candidate.aliases
    assert candidate._saved is True


def test_list_collections_reports_aliases_for_known_names(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "k")
    monkeypatch.setenv("WANDB_ENTITY", "acme")
    arts = {
        "acme/wandb-registry-model/so101-kitting-smolvla:latest": FakeArtifact(
            "acme/wandb-registry-model/so101-kitting-smolvla:latest",
            ["latest", "candidate"],
        ),
    }
    store = RegistryStore(api=FakeApi(arts))
    rows = store.list_collections("acme")
    smol = next(r for r in rows if r["name"] == "so101-kitting-smolvla")
    assert "candidate" in smol["aliases"]
