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


def test_train_entry_swaps_the_logger_before_main(monkeypatch) -> None:
    import lerobot.scripts.lerobot_train as train_mod
    from lelab.forge.remote import train_entry
    from lelab.forge.tracking import WandbLogger

    called = {}

    def fake_main() -> None:
        called["logger"] = train_mod.TrackioLogger

    original = train_mod.TrackioLogger
    monkeypatch.setattr(train_mod, "main", fake_main)
    try:
        train_entry.main()
        assert called["logger"] is WandbLogger
    finally:
        train_mod.TrackioLogger = original
