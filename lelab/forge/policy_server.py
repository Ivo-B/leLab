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

POLICY_PORT = 7447


def policy_service() -> dict:
    return {
        "port": POLICY_PORT,
        "name": "policy",
        "visibility": "public",
        "endpoint": {"kind": "tls_passthrough"},
    }


def assigned_endpoint(addresses) -> str:
    for item in addresses:
        if item.name == "policy":
            return f"tls/{item.address}"
    raise ValueError("The Sandbox did not publish a policy endpoint")


def zenoh_server_config() -> str:
    return json.dumps(
        {
            "listen": {"endpoints": ["tls/0.0.0.0:7447"]},
            "transport": {
                "link": {
                    "tls": {
                        "root_ca_certificate": "/work/ca.pem",
                        "listen_private_key": "/work/key.pem",
                        "listen_certificate": "/work/cert.pem",
                        "enable_mtls": False,
                    }
                }
            },
        }
    )


def zenoh_client_config() -> str:
    return json.dumps(
        {
            "transport": {
                "link": {
                    "tls": {
                        "root_ca_certificate": "/tmp/lelab-policy-ca.pem",
                        "verify_name_on_connect": True,
                        "enable_mtls": False,
                    }
                }
            }
        }
    )


def policy_server_argv() -> list[str]:
    return [
        "lerobot-policy-server",
        "--config_path=/work/server.yaml",
        "--model.device=cuda",
    ]
