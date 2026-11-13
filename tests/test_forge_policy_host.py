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

import sys
import types
from types import SimpleNamespace

import pytest
import yaml

from lelab.forge.policy_server import (
    PolicyHost,
    policy_server_argv,
    policy_server_yaml,
    policy_service,
    zenoh_server_config,
)
from lelab.utils.forge_config import ForgeConfig

CERTIFICATES = {"ca.pem": b"CA", "key.pem": b"KEY", "cert.pem": b"CERT"}
FAILURE_PREFIX = "Could not start the CoreWeave policy server: "
FAILURE_SUFFIX = " Use the local policy host instead."


def _cfg() -> ForgeConfig:
    return ForgeConfig(
        api_key="wandb-secret",
        entity="coreweave-demo",
        project="so101-kitting",
        sandbox_api_key="CWSANDBOX_API_KEY-secret",
        sandbox_image="registry.example/lelab-train:cuda",
        hf_token="hf-secret",
    )


class FakeHandle:
    def __init__(self, addresses=None, exec_error: Exception | None = None) -> None:
        self.service_addresses = (
            [SimpleNamespace(name="policy", address="policy.example:443")] if addresses is None else addresses
        )
        self.exec_error = exec_error
        self.writes: list[tuple[str, bytes]] = []
        self.execs: list[list[str]] = []
        self.stops = 0

    def write_file(self, path, contents):
        self.writes.append((path, contents))

    def exec(self, command):
        if self.exec_error is not None:
            raise self.exec_error
        self.execs.append(command)
        return object()

    def stop(self):
        self.stops += 1


class FakeClient:
    def __init__(self, handle: FakeHandle | None = None, error: Exception | None = None) -> None:
        self.handle = handle
        self.error = error
        self.created: list[dict] = []

    def create(self, kwargs):
        self.created.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.handle


def test_start_creates_the_sandbox_writes_files_and_returns_the_endpoint() -> None:
    handle = FakeHandle()
    client = FakeClient(handle)
    cfg = _cfg()
    host = PolicyHost(client, cfg, CERTIFICATES)

    endpoint = host.start("me/checkpoint")

    assert endpoint == "tls/policy.example:443"
    kwargs = client.created[0]
    assert kwargs["services"] == [policy_service()]
    assert kwargs["tags"] == ["lelab"]
    assert kwargs["args"] == ["sleep", "infinity"]
    assert kwargs["container_image"] == cfg.sandbox_image
    assert kwargs["max_lifetime_seconds"] == cfg.sandbox_lifetime_seconds
    assert kwargs["resources"]["cpu"] == cfg.sandbox_cpu
    assert kwargs["resources"]["memory"] == cfg.sandbox_memory
    assert kwargs["volumes"] == [{"name": "work", "mount_path": "/work", "size": cfg.sandbox_volume_size}]
    assert "CWSANDBOX_API_KEY" not in kwargs["environment_variables"]
    assert cfg.sandbox_api_key not in kwargs["environment_variables"].values()
    assert [path for path, _ in handle.writes] == [
        "/work/ca.pem",
        "/work/key.pem",
        "/work/cert.pem",
        "/work/zenoh.json5",
        "/work/server.yaml",
    ]
    assert dict(handle.writes) == {
        "/work/ca.pem": b"CA",
        "/work/key.pem": b"KEY",
        "/work/cert.pem": b"CERT",
        "/work/zenoh.json5": zenoh_server_config().encode(),
        "/work/server.yaml": policy_server_yaml("me/checkpoint").encode(),
    }
    assert handle.execs == [policy_server_argv()]
    assert handle.stops == 0


def test_start_writes_the_server_config_before_exec() -> None:
    handle = FakeHandle()
    order: list[str] = []
    handle.write_file = lambda path, contents: (order.append(path), handle.writes.append((path, contents)))[1]
    handle.exec = lambda command: order.append("exec")
    PolicyHost(FakeClient(handle), _cfg(), CERTIFICATES).start("me/checkpoint")

    assert "/work/server.yaml" in dict(handle.writes)
    assert order.index("/work/server.yaml") < order.index("exec")
    text = dict(handle.writes)["/work/server.yaml"].decode()
    assert "me/checkpoint" in text
    assert "/work/zenoh.json5" in text
    parsed = yaml.safe_load(text)
    assert parsed == {
        "deployment": "so101-kitting",
        "semantics": "so101-degrees-gripper-percent-v1",
        "robot_type": "so101_follower",
        "model": {"repo_or_path": "me/checkpoint", "device": "cuda"},
        "zenoh": {
            "config_file": "/work/zenoh.json5",
            "listen_endpoints": ["tls/0.0.0.0:7447"],
            "mode": "peer",
        },
    }


def test_stop_stops_the_sandbox() -> None:
    handle = FakeHandle()
    host = PolicyHost(FakeClient(handle), _cfg(), CERTIFICATES)
    host.start("me/checkpoint")
    host.stop()
    assert handle.stops == 1


def test_stop_without_a_started_sandbox_does_nothing() -> None:
    PolicyHost(FakeClient(FakeHandle()), _cfg(), CERTIFICATES).stop()


def test_create_failure_is_redacted_and_points_at_the_local_host() -> None:
    client = FakeClient(error=RuntimeError("bad CWSANDBOX_API_KEY CWSANDBOX_API_KEY-secret wandb-secret"))
    host = PolicyHost(client, _cfg(), CERTIFICATES)

    with pytest.raises(RuntimeError) as excinfo:
        host.start("me/checkpoint")

    message = str(excinfo.value)
    assert message.startswith(FAILURE_PREFIX)
    assert message.endswith(FAILURE_SUFFIX)
    assert "CWSANDBOX_API_KEY" not in message
    assert "wandb-secret" not in message
    assert "[redacted]" in message


def test_create_failure_does_not_leak_the_private_key_bytes() -> None:
    key = b"-----BEGIN PRIVATE KEY-----\nabc123\n-----END PRIVATE KEY-----\n"
    certificates = {**CERTIFICATES, "key.pem": key}
    for message in (f"failed {key!r}", f"failed {key.decode()}", f"failed {key.decode()!r}"):
        host = PolicyHost(FakeClient(error=RuntimeError(message)), _cfg(), certificates)
        with pytest.raises(RuntimeError) as excinfo:
            host.start("me/checkpoint")
        assert "abc123" not in str(excinfo.value)
        assert "PRIVATE KEY" not in str(excinfo.value)
        assert "[redacted]" in str(excinfo.value)


def test_create_failure_does_not_log_the_exception(caplog) -> None:
    client = FakeClient(error=RuntimeError("boom CWSANDBOX_API_KEY-secret"))
    with caplog.at_level("DEBUG"), pytest.raises(RuntimeError):
        PolicyHost(client, _cfg(), CERTIFICATES).start("me/checkpoint")
    assert "boom" not in caplog.text
    assert "CWSANDBOX_API_KEY-secret" not in caplog.text


def test_missing_policy_address_stops_the_sandbox_and_raises() -> None:
    handle = FakeHandle(addresses=[SimpleNamespace(name="other", address="x:1")])
    host = PolicyHost(FakeClient(handle), _cfg(), CERTIFICATES)

    with pytest.raises(RuntimeError) as excinfo:
        host.start("me/checkpoint")

    assert str(excinfo.value).startswith(FAILURE_PREFIX)
    assert str(excinfo.value).endswith(FAILURE_SUFFIX)
    assert handle.stops == 1


def test_exec_failure_stops_the_sandbox_and_redacts() -> None:
    handle = FakeHandle(exec_error=RuntimeError("exec failed hf-secret"))
    host = PolicyHost(FakeClient(handle), _cfg(), CERTIFICATES)

    with pytest.raises(RuntimeError) as excinfo:
        host.start("me/checkpoint")

    assert str(excinfo.value).startswith(FAILURE_PREFIX)
    assert str(excinfo.value).endswith(FAILURE_SUFFIX)
    assert "hf-secret" not in str(excinfo.value)
    assert handle.stops == 1


def test_write_failure_stops_the_sandbox() -> None:
    handle = FakeHandle()

    def failing_write(path, contents):
        raise OSError("disk full")

    handle.write_file = failing_write
    host = PolicyHost(FakeClient(handle), _cfg(), CERTIFICATES)

    with pytest.raises(RuntimeError, match="^Could not start the CoreWeave policy server: "):
        host.start("me/checkpoint")
    assert handle.stops == 1


def test_a_stop_failure_while_cleaning_up_still_raises_the_start_error() -> None:
    handle = FakeHandle(addresses=[])

    def failing_stop():
        raise RuntimeError("cannot stop CWSANDBOX_API_KEY-secret")

    handle.stop = failing_stop
    host = PolicyHost(FakeClient(handle), _cfg(), CERTIFICATES)

    with pytest.raises(RuntimeError) as excinfo:
        host.start("me/checkpoint")
    assert str(excinfo.value).startswith(FAILURE_PREFIX)
    assert "CWSANDBOX_API_KEY-secret" not in str(excinfo.value)


# -- CwsandboxClient.create services handling --


def _install_fake_cwsandbox(monkeypatch):
    run_calls: list[dict] = []

    class Future:
        def wait(self):
            return "sandbox"

    class FakeSandbox:
        @staticmethod
        def run(*args, **kwargs):
            run_calls.append({"args": args, **kwargs})
            return Future()

    class Service:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Endpoint:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake = types.SimpleNamespace(
        Sandbox=FakeSandbox,
        AuthStrategy=types.SimpleNamespace(COREWEAVE_API_KEY="cw-key"),
        ScratchVolumeOptions=lambda **kw: ("volume", kw),
        Service=Service,
        Endpoint=Endpoint,
        EndpointKind=types.SimpleNamespace(TLS_PASSTHROUGH="tls_passthrough"),
        ServiceVisibility=types.SimpleNamespace(PUBLIC="public"),
    )
    monkeypatch.setitem(sys.modules, "cwsandbox", fake)
    return run_calls


def _base_kwargs() -> dict:
    return {
        "args": ["sleep", "infinity"],
        "container_image": "img",
        "placement_mode": "serverless",
        "tags": ["lelab"],
        "max_lifetime_seconds": 60,
        "resources": {"cpu": "1", "memory": "1Gi", "gpu": 1},
        "volumes": [{"name": "work", "mount_path": "/work", "size": "1Gi"}],
        "environment_variables": {},
    }


def test_client_omits_services_when_the_key_is_absent(monkeypatch) -> None:
    run_calls = _install_fake_cwsandbox(monkeypatch)
    from lelab.runners.cwsandbox_client import CwsandboxClient

    CwsandboxClient().create(_base_kwargs())

    assert "services" not in run_calls[0]


def test_client_omits_services_when_the_list_is_empty(monkeypatch) -> None:
    run_calls = _install_fake_cwsandbox(monkeypatch)
    from lelab.runners.cwsandbox_client import CwsandboxClient

    CwsandboxClient().create({**_base_kwargs(), "services": []})

    assert "services" not in run_calls[0]


def test_client_passes_a_tls_passthrough_service(monkeypatch) -> None:
    run_calls = _install_fake_cwsandbox(monkeypatch)
    from lelab.runners.cwsandbox_client import CwsandboxClient

    CwsandboxClient().create({**_base_kwargs(), "services": [policy_service()]})

    (service,) = run_calls[0]["services"]
    assert service.kwargs["port"] == 7447
    assert service.kwargs["name"] == "policy"
    assert service.kwargs["visibility"] == "public"
    assert service.kwargs["endpoint"].kwargs == {"kind": "tls_passthrough"}


def test_sandbox_handle_exposes_the_service_addresses() -> None:
    from lelab.runners.cwsandbox_client import SandboxHandle

    addresses = (SimpleNamespace(name="policy", address="h:443"),)
    handle = SandboxHandle(SimpleNamespace(service_addresses=addresses))
    assert handle.service_addresses == addresses
