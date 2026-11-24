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

import shlex
import sys
import time
import types
from types import SimpleNamespace

import pytest
import yaml

from lelab.forge.policy_server import (
    PolicyHost,
    checkpoint_features,
    policy_server_argv,
    policy_server_yaml,
    policy_service,
    zenoh_server_config,
)
from lelab.utils.forge_config import ForgeConfig

CERTIFICATES = {"ca.pem": b"CA", "key.pem": b"KEY", "cert.pem": b"CERT"}
FAILURE_PREFIX = "Could not start the CoreWeave policy server: "
FAILURE_SUFFIX = " Close the evaluation and reopen it with the local policy host."
READY_LINE = "Policy server ready: deployment=so101-chess instance=abc; awaiting one client"
ARTIFACT = "acme/wandb-registry-model/so101-chess-act:candidate"
FEATURES, ACTION = checkpoint_features(
    {
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [6]},
            "observation.images.top": {"type": "VISUAL", "shape": [3, 480, 640]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [6]}},
    }
)
START = (ARTIFACT, "pretrained_model", FEATURES, ACTION)


def _cfg() -> ForgeConfig:
    return ForgeConfig(
        api_key="wandb-secret",
        entity="coreweave-demo",
        project="so101-chess",
        sandbox_api_key="CWSANDBOX_API_KEY-secret",
        sandbox_image="registry.example/lelab-train:cuda",
        hf_token="hf-secret",
    )


@pytest.fixture(autouse=True)
def _client_ca_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("lelab.forge.policy_server.CLIENT_CA_PATH", str(tmp_path / "client-ca.pem"))


class FakeProcess:
    def __init__(self, lines=(), returncode: int | None = 0) -> None:
        self.stdout = iter(lines)
        self.returncode = returncode


class FakeHandle:
    """`processes` are returned by successive `exec` calls: download, then server."""

    def __init__(self, addresses=None, exec_error: Exception | None = None, processes=None) -> None:
        self.service_addresses = (
            [SimpleNamespace(name="policy", address="policy.example:443")] if addresses is None else addresses
        )
        self.exec_error = exec_error
        self.processes = (
            list(processes) if processes is not None else [FakeProcess(), FakeProcess([READY_LINE])]
        )
        self.writes: list[tuple[str, bytes]] = []
        self.execs: list[list[str]] = []
        self.stops = 0

    def write_file(self, path, contents):
        self.writes.append((path, contents))

    def exec(self, command):
        if self.exec_error is not None:
            raise self.exec_error
        self.execs.append(command)
        return self.processes.pop(0) if self.processes else FakeProcess()

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


def _download_command(path: str) -> list[str]:
    code = 'import sys, wandb; wandb.Api().artifact(sys.argv[1]).download(root="/work/policy")'
    return ["sh", "-c", f"python -c {shlex.quote(code)} {shlex.quote(path)} 2>&1"]


def _server_command() -> list[str]:
    return ["sh", "-c", f"exec {shlex.join(policy_server_argv())} 2>&1"]


def test_start_creates_the_sandbox_writes_files_and_returns_the_endpoint() -> None:
    handle = FakeHandle()
    client = FakeClient(handle)
    cfg = _cfg()
    host = PolicyHost(client, cfg, issuer=lambda hostname: CERTIFICATES)

    endpoint = host.start(*START)

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
    env = kwargs["environment_variables"]
    assert env["WANDB_API_KEY"] == cfg.api_key
    assert env["WANDB_ENTITY"] == cfg.entity
    assert "CWSANDBOX_API_KEY" not in env
    assert cfg.sandbox_api_key not in env.values()
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
        "/work/server.yaml": policy_server_yaml("/work/policy/pretrained_model", FEATURES, ACTION).encode(),
    }
    assert handle.execs == [_download_command(ARTIFACT), _server_command()]
    assert handle.stops == 0


def test_start_downloads_the_checkpoint_before_writing_the_server_config_and_starting_the_server() -> None:
    handle = FakeHandle()
    order: list[str] = []
    handle.write_file = lambda path, contents: (order.append(path), handle.writes.append((path, contents)))[1]
    original_exec = handle.exec
    handle.exec = lambda command: (order.append("exec"), original_exec(command))[1]
    PolicyHost(FakeClient(handle), _cfg(), issuer=lambda hostname: CERTIFICATES).start(*START)

    assert order == [
        "/work/ca.pem",
        "/work/key.pem",
        "/work/cert.pem",
        "/work/zenoh.json5",
        "exec",
        "/work/server.yaml",
        "exec",
    ]
    for command in handle.execs:
        assert command[:2] == ["sh", "-c"]
        assert command[2].endswith(" 2>&1")
    assert "WANDB_API_KEY" not in handle.execs[0][2]
    assert "wandb-secret" not in handle.execs[0][2]


def test_start_writes_a_server_config_for_the_checkpoint() -> None:
    handle = FakeHandle()
    PolicyHost(FakeClient(handle), _cfg(), issuer=lambda hostname: CERTIFICATES).start(*START)

    parsed = yaml.safe_load(dict(handle.writes)["/work/server.yaml"].decode())
    assert parsed["model"] == {"repo_or_path": "/work/policy/pretrained_model", "device": "cuda"}
    assert parsed["zenoh"] == {
        "config_file": "/work/zenoh.json5",
        "listen_endpoints": ["tls/0.0.0.0:7447"],
        "mode": "peer",
    }
    assert parsed["deployment"] == "so101-chess"
    assert parsed["features"] == FEATURES
    assert parsed["action_feature"] == ACTION
    assert [feature["name"] for feature in parsed["features"]] == [
        "observation.state",
        "observation.images.top",
    ]


def test_the_model_path_has_no_trailing_slash_without_a_subdir() -> None:
    handle = FakeHandle()
    PolicyHost(FakeClient(handle), _cfg(), issuer=lambda hostname: CERTIFICATES).start(
        ARTIFACT, "", FEATURES, ACTION
    )

    parsed = yaml.safe_load(dict(handle.writes)["/work/server.yaml"].decode())
    assert parsed["model"]["repo_or_path"] == "/work/policy"


def test_the_download_command_quotes_the_artifact_path() -> None:
    handle = FakeHandle()
    path = "acme/wandb-registry-model/x; touch /tmp/pwned:candidate"
    PolicyHost(FakeClient(handle), _cfg(), issuer=lambda hostname: CERTIFICATES).start(
        path, "pretrained_model", FEATURES, ACTION
    )

    assert handle.execs[0] == _download_command(path)
    assert shlex.quote(path) in handle.execs[0][2]


def test_a_failed_download_is_redacted_and_stops_the_sandbox() -> None:
    lines = [f"line {number}\n" for number in range(30)] + ["401 for wandb-secret CWSANDBOX_API_KEY\n"]
    handle = FakeHandle(processes=[FakeProcess(lines, returncode=1)])
    host = PolicyHost(FakeClient(handle), _cfg(), issuer=lambda hostname: CERTIFICATES)

    with pytest.raises(RuntimeError) as excinfo:
        host.start(*START)

    message = str(excinfo.value)
    assert message.startswith(FAILURE_PREFIX)
    assert message.endswith(FAILURE_SUFFIX)
    assert "wandb-secret" not in message
    assert "CWSANDBOX_API_KEY" not in message
    assert "[redacted]" in message
    assert "line 29" in message
    assert "line 0" not in message
    assert handle.stops == 1
    assert len(handle.execs) == 1
    assert "/work/server.yaml" not in dict(handle.writes)


def test_a_download_without_an_exit_code_counts_as_a_failure() -> None:
    handle = FakeHandle(processes=[FakeProcess([], returncode=None)])
    host = PolicyHost(FakeClient(handle), _cfg(), issuer=lambda hostname: CERTIFICATES)

    with pytest.raises(RuntimeError, match="^Could not start the CoreWeave policy server: "):
        host.start(*START)
    assert handle.stops == 1


def test_a_download_stream_error_is_redacted_and_stops_the_sandbox() -> None:
    def broken():
        yield "starting\n"
        raise OSError("stream dropped wandb-secret")

    handle = FakeHandle(processes=[FakeProcess(broken(), returncode=0)])
    host = PolicyHost(FakeClient(handle), _cfg(), issuer=lambda hostname: CERTIFICATES)

    with pytest.raises(RuntimeError) as excinfo:
        host.start(*START)
    assert "wandb-secret" not in str(excinfo.value)
    assert handle.stops == 1


# -- wait_until_ready --


def _started_host(handle: FakeHandle) -> PolicyHost:
    host = PolicyHost(FakeClient(handle), _cfg(), issuer=lambda hostname: CERTIFICATES)
    host.start(*START)
    return host


def test_wait_until_ready_returns_when_the_ready_line_appears() -> None:
    handle = FakeHandle(
        processes=[FakeProcess(), FakeProcess(["loading\n", READY_LINE + "\n", "later noise\n"])]
    )
    host = _started_host(handle)

    host.wait_until_ready(5)

    assert handle.stops == 0


def test_a_stream_that_ends_without_the_ready_line_fails_and_stops_the_sandbox() -> None:
    handle = FakeHandle(
        processes=[FakeProcess(), FakeProcess(["loading CWSANDBOX_API_KEY-secret\n", "bye\n"])]
    )
    host = _started_host(handle)

    with pytest.raises(RuntimeError) as excinfo:
        host.wait_until_ready(5)

    message = str(excinfo.value)
    assert message.startswith(FAILURE_PREFIX)
    assert message.endswith(FAILURE_SUFFIX)
    assert "bye" in message
    assert "CWSANDBOX_API_KEY-secret" not in message
    assert handle.stops == 1


def test_a_traceback_tail_fails_without_waiting_for_the_stream_to_end() -> None:
    def lines():
        yield "loading\n"
        yield "Traceback (most recent call last):\n"
        yield "ValueError: wandb-secret is not a checkpoint\n"
        time.sleep(30)  # the server hangs after the error; the host must not wait for it
        yield READY_LINE + "\n"

    handle = FakeHandle(processes=[FakeProcess(), FakeProcess(lines())])
    host = _started_host(handle)
    started = time.monotonic()

    with pytest.raises(RuntimeError) as excinfo:
        host.wait_until_ready(10)

    assert time.monotonic() - started < 5
    assert "ValueError" in str(excinfo.value)
    assert "wandb-secret" not in str(excinfo.value)
    assert handle.stops == 1


def test_wait_until_ready_times_out_and_stops_the_sandbox() -> None:
    def lines():
        yield "loading\n"
        time.sleep(30)
        yield READY_LINE + "\n"

    handle = FakeHandle(processes=[FakeProcess(), FakeProcess(lines())])
    host = _started_host(handle)

    with pytest.raises(RuntimeError) as excinfo:
        host.wait_until_ready(0.2)

    assert str(excinfo.value).startswith(FAILURE_PREFIX)
    assert "loading" in str(excinfo.value)
    assert handle.stops == 1


def test_wait_until_ready_keeps_only_the_last_twenty_lines() -> None:
    lines = [f"line {number}\n" for number in range(40)]
    handle = FakeHandle(processes=[FakeProcess(), FakeProcess(lines)])
    host = _started_host(handle)

    with pytest.raises(RuntimeError) as excinfo:
        host.wait_until_ready(5)

    assert "line 39" in str(excinfo.value)
    assert "line 20" in str(excinfo.value)
    assert "line 19" not in str(excinfo.value)


def test_wait_until_ready_before_start_fails_cleanly() -> None:
    host = PolicyHost(FakeClient(FakeHandle()), _cfg(), issuer=lambda hostname: CERTIFICATES)

    with pytest.raises(RuntimeError, match="^Could not start the CoreWeave policy server: "):
        host.wait_until_ready(1)


def test_stop_stops_the_sandbox() -> None:
    handle = FakeHandle()
    host = PolicyHost(FakeClient(handle), _cfg(), issuer=lambda hostname: CERTIFICATES)
    host.start(*START)
    host.stop()
    assert handle.stops == 1


def test_stop_without_a_started_sandbox_does_nothing() -> None:
    PolicyHost(FakeClient(FakeHandle()), _cfg(), issuer=lambda hostname: CERTIFICATES).stop()


def test_create_failure_is_redacted_and_points_at_the_local_host() -> None:
    client = FakeClient(error=RuntimeError("bad CWSANDBOX_API_KEY CWSANDBOX_API_KEY-secret wandb-secret"))
    host = PolicyHost(client, _cfg(), issuer=lambda hostname: CERTIFICATES)

    with pytest.raises(RuntimeError) as excinfo:
        host.start(*START)

    message = str(excinfo.value)
    assert message.startswith(FAILURE_PREFIX)
    assert message.endswith(FAILURE_SUFFIX)
    assert "CWSANDBOX_API_KEY" not in message
    assert "wandb-secret" not in message
    assert "[redacted]" in message


def test_start_issues_the_certificate_for_the_assigned_host(tmp_path, monkeypatch) -> None:
    seen: list[str] = []

    def issuer(hostname: str) -> dict[str, bytes]:
        seen.append(hostname)
        return {"ca.pem": b"CA-for-" + hostname.encode(), "key.pem": b"KEY", "cert.pem": b"CERT"}

    handle = FakeHandle()
    ca_path = tmp_path / "ca.pem"
    monkeypatch.setattr("lelab.forge.policy_server.CLIENT_CA_PATH", str(ca_path))
    endpoint = PolicyHost(FakeClient(handle), _cfg(), issuer=issuer).start(*START)

    assert endpoint == "tls/policy.example:443"
    assert seen == ["policy.example"]
    assert dict(handle.writes)["/work/ca.pem"] == b"CA-for-policy.example"
    assert ca_path.read_bytes() == b"CA-for-policy.example"


def test_a_failure_after_issuing_does_not_leak_the_private_key_bytes() -> None:
    key = b"-----BEGIN PRIVATE KEY-----\nabc123\n-----END PRIVATE KEY-----\n"
    certificates = {**CERTIFICATES, "key.pem": key}
    for message in (f"failed {key!r}", f"failed {key.decode()}", f"failed {key.decode()!r}"):
        handle = FakeHandle(exec_error=RuntimeError(message))
        host = PolicyHost(FakeClient(handle), _cfg(), issuer=lambda hostname: certificates)
        with pytest.raises(RuntimeError) as excinfo:
            host.start(*START)
        assert "abc123" not in str(excinfo.value)
        assert "PRIVATE KEY" not in str(excinfo.value)
        assert "[redacted]" in str(excinfo.value)
        assert handle.stops == 1


def test_create_failure_does_not_log_the_exception(caplog) -> None:
    client = FakeClient(error=RuntimeError("boom CWSANDBOX_API_KEY-secret"))
    with caplog.at_level("DEBUG"), pytest.raises(RuntimeError):
        PolicyHost(client, _cfg(), issuer=lambda hostname: CERTIFICATES).start(*START)
    assert "boom" not in caplog.text
    assert "CWSANDBOX_API_KEY-secret" not in caplog.text


def test_missing_policy_address_stops_the_sandbox_and_raises() -> None:
    handle = FakeHandle(addresses=[SimpleNamespace(name="other", address="x:1")])
    host = PolicyHost(FakeClient(handle), _cfg(), issuer=lambda hostname: CERTIFICATES)

    with pytest.raises(RuntimeError) as excinfo:
        host.start(*START)

    assert str(excinfo.value).startswith(FAILURE_PREFIX)
    assert str(excinfo.value).endswith(FAILURE_SUFFIX)
    assert handle.stops == 1


def test_exec_failure_stops_the_sandbox_and_redacts() -> None:
    handle = FakeHandle(exec_error=RuntimeError("exec failed hf-secret"))
    host = PolicyHost(FakeClient(handle), _cfg(), issuer=lambda hostname: CERTIFICATES)

    with pytest.raises(RuntimeError) as excinfo:
        host.start(*START)

    assert str(excinfo.value).startswith(FAILURE_PREFIX)
    assert str(excinfo.value).endswith(FAILURE_SUFFIX)
    assert "hf-secret" not in str(excinfo.value)
    assert handle.stops == 1


def test_write_failure_stops_the_sandbox() -> None:
    handle = FakeHandle()

    def failing_write(path, contents):
        raise OSError("disk full")

    handle.write_file = failing_write
    host = PolicyHost(FakeClient(handle), _cfg(), issuer=lambda hostname: CERTIFICATES)

    with pytest.raises(RuntimeError, match="^Could not start the CoreWeave policy server: "):
        host.start(*START)
    assert handle.stops == 1


def test_a_stop_failure_while_cleaning_up_still_raises_the_start_error() -> None:
    handle = FakeHandle(addresses=[])

    def failing_stop():
        raise RuntimeError("cannot stop CWSANDBOX_API_KEY-secret")

    handle.stop = failing_stop
    host = PolicyHost(FakeClient(handle), _cfg(), issuer=lambda hostname: CERTIFICATES)

    with pytest.raises(RuntimeError) as excinfo:
        host.start(*START)
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
