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

import json

import pytest
import yaml

from lelab.forge.policy_server import (
    POLICY_PORT,
    assigned_endpoint,
    checkpoint_features,
    policy_server_argv,
    policy_server_yaml,
    policy_service,
    zenoh_client_config,
    zenoh_server_config,
)


class _Address:
    def __init__(self, name: str, address: str) -> None:
        self.name = name
        self.address = address


def test_service_is_tls_passthrough_on_the_zenoh_port() -> None:
    service = policy_service()
    assert service["port"] == POLICY_PORT == 7447
    assert service["name"] == "policy"
    assert service["visibility"] == "public"
    assert service["endpoint"] == {"kind": "tls_passthrough"}
    assert "auth" not in service["endpoint"]


def test_assigned_endpoint_uses_the_platform_address() -> None:
    endpoint = assigned_endpoint([_Address("policy", "8443-abc.ep2.cwsandbox.com:443")])
    assert endpoint == "tls/8443-abc.ep2.cwsandbox.com:443"


def test_zenoh_configs_name_the_documented_tls_fields() -> None:
    server = json.loads(zenoh_server_config())
    tls = server["transport"]["link"]["tls"]
    assert server["listen"]["endpoints"] == ["tls/0.0.0.0:7447"]
    assert tls["listen_private_key"] == "/work/key.pem"
    assert tls["listen_certificate"] == "/work/cert.pem"
    assert tls["root_ca_certificate"] == "/work/ca.pem"
    assert tls["enable_mtls"] is False
    client = json.loads(zenoh_client_config())
    assert client["transport"]["link"]["tls"]["verify_name_on_connect"] is True
    assert client["transport"]["link"]["tls"]["root_ca_certificate"] == "/tmp/lelab-policy-ca.pem"


def test_zenoh_loads_both_configs(tmp_path) -> None:
    from lerobot.transport import ZenohConfig

    client_path = tmp_path / "client.json5"
    client_path.write_text(zenoh_client_config())
    server_path = tmp_path / "server.json5"
    server_path.write_text(zenoh_server_config())

    ZenohConfig(connect_endpoints=["tls/policy.example:443"], config_file=str(client_path)).build()
    ZenohConfig(listen_endpoints=["tls/0.0.0.0:7447"], config_file=str(server_path)).build()


def test_policy_server_argv_pins_cuda() -> None:
    assert policy_server_argv() == [
        "lerobot-policy-server",
        "--config_path=/work/server.yaml",
        "--model.device=cuda",
    ]


JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def _checkpoint_config() -> dict:
    return {
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [6]},
            "observation.images.wrist": {"type": "VISUAL", "shape": [3, 240, 320]},
            "observation.images.top": {"type": "VISUAL", "shape": [3, 480, 640]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [6]}},
    }


def test_checkpoint_features_come_from_the_checkpoint_config() -> None:
    features, action = checkpoint_features(_checkpoint_config())

    assert features == [
        {
            "name": "observation.state",
            "shape": [6],
            "dtype": "float32",
            "names": JOINTS,
            "semantics": "so101-degrees-gripper-percent-v1",
        },
        {
            "name": "observation.images.wrist",
            "shape": [240, 320, 3],
            "dtype": "uint8",
            "kind": "rgb",
            "semantics": "rgb-wrist-v1",
        },
        {
            "name": "observation.images.top",
            "shape": [480, 640, 3],
            "dtype": "uint8",
            "kind": "rgb",
            "semantics": "rgb-top-v1",
        },
    ]
    assert action == {
        "name": "action",
        "shape": [6],
        "dtype": "float32",
        "names": JOINTS,
        "semantics": "so101-degrees-gripper-percent-v1",
    }


def test_checkpoint_features_omit_joint_names_when_the_state_is_not_six_wide() -> None:
    config = _checkpoint_config()
    config["input_features"]["observation.state"]["shape"] = [7]
    config["output_features"]["action"]["shape"] = [7]

    features, action = checkpoint_features(config)

    assert "names" not in features[0]
    assert features[0]["shape"] == [7]
    assert "names" not in action


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c.pop("input_features"), "input_features"),
        (lambda c: c.update(input_features={"observation.state": {"type": "STATE", "shape": [6]}}), "camera"),
        (lambda c: c.pop("output_features"), "action"),
    ],
)
def test_checkpoint_features_reject_incomplete_configs(mutate, message) -> None:
    config = _checkpoint_config()
    mutate(config)

    with pytest.raises(ValueError, match=message):
        checkpoint_features(config)


def test_policy_server_yaml_uses_the_given_model_path_and_features() -> None:
    features, action = checkpoint_features(_checkpoint_config())

    parsed = yaml.safe_load(policy_server_yaml("/work/policy/pretrained_model", features, action))

    assert parsed["model"] == {"repo_or_path": "/work/policy/pretrained_model", "device": "cuda"}
    assert parsed["features"] == features
    assert parsed["action_feature"] == action
    assert parsed["deployment"] == "so101-chess"


def test_generated_yaml_parses_into_lerobots_server_config(tmp_path) -> None:
    import draccus

    from lerobot.remote_inference.configs import ServerConfig

    features, action = checkpoint_features(_checkpoint_config())
    path = tmp_path / "server.yaml"
    path.write_text(policy_server_yaml("/work/policy/pretrained_model", features, action))

    cfg = draccus.parse(ServerConfig, config_path=str(path), args=[])

    assert cfg.model.repo_or_path == "/work/policy/pretrained_model"
    assert [feature.name for feature in cfg.features] == [
        "observation.state",
        "observation.images.wrist",
        "observation.images.top",
    ]
    assert cfg.features[1].shape == (240, 320, 3)
    assert cfg.features[1].kind == "rgb"
    assert cfg.features[0].names == tuple(JOINTS)
    assert cfg.action_feature is not None and cfg.action_feature.names == tuple(JOINTS)
