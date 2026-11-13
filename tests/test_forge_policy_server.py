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

from lelab.forge.policy_server import (
    POLICY_PORT,
    assigned_endpoint,
    policy_server_argv,
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


def test_policy_server_argv_pins_cuda() -> None:
    assert policy_server_argv() == [
        "lerobot-policy-server",
        "--config_path=/work/server.yaml",
        "--model.device=cuda",
    ]
