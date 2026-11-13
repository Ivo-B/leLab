# LeLab Forge M4 — Evaluate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An operator can score a candidate checkpoint on the real arm, see each trial in W&B Weave, and point the arm at a policy running in a CoreWeave GPU Sandbox.

**Architecture:** The Evaluate page starts one trial at a time through the existing rollout mutex. A pure judge parser turns a Serverless Inference JSON reply into a stage-ladder verdict and never treats a parse error as success. A Weave logger records one row per finished trial. The CoreWeave policy host is a second Sandbox, tagged `lelab`, whose Zenoh listener is reached through a TLS-passthrough endpoint. Local rollout stays the default host.

**Tech Stack:** Python 3.12, FastAPI, the existing rollout subprocess, `cwsandbox` 1.28.0, the OpenAI Python client against `https://api.inference.wandb.ai/v1`, `weave.EvaluationLogger`, React + Vite.

**Verified 2026-11-13, before this plan was written:**

- LeRobot v0.7 `lerobot-policy-server` is Zenoh, not gRPC. The generic server listens on `tcp/127.0.0.1:7447` (`examples/remote_inference/server.yaml`). `ZenohConfig.build()` keeps TLS settings from a JSON5 `config_file` and overwrites `listen/endpoints`.
- A Sandbox HTTPS endpoint carries HTTP, WebSockets, and Server-Sent Events only. It cannot carry Zenoh. TLS passthrough (`EndpointKind.TLS_PASSTHROUGH`, `auth` omitted) carries any protocol over TLS. The address is `host:port` on `sandbox.service_addresses`. The platform port is 443. The container port in this plan is 7447. The client must send that host as SNI, and the certificate must cover it. The hostname is known only after create, so the host generates the certificate after reading `service_addresses`, writes it into the Sandbox, then starts the server.
- Zenoh TLS keys, from the current `DEFAULT_CONFIG.json5`: `transport.link.tls.root_ca_certificate`, `listen_private_key`, `listen_certificate`, `enable_mtls`, `verify_name_on_connect`. Listen locator `tls/0.0.0.0:7447`. Connect locator `tls/<assigned-host>:443`.
- No Europe round-trip was measured. Measuring it needs a live GPU Sandbox. Do not invent a millisecond budget. Connection setup uses Zenoh's existing `open_timeout_s` of 10 seconds. A trial whose rollout fails is marked failed, and the page offers the local host.
- Serverless Inference base URL is `https://api.inference.wandb.ai/v1`. The catalogue model that is both vision and described for visual understanding plus structured outputs is `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B`. No image-byte limit is published. This plan caps each JPEG at 200,000 bytes before the request is sent.
- `weave.EvaluationLogger.log_prediction` / `log_score` / `finish`, then `log_summary`, is the imperative one-row-at-a-time API.

## Global Constraints

- Do not fork LeRobot. The Sandbox runs `lerobot-policy-server` from the image.
- Do not start M5 or M6. No order agent, Agent Lens, Automations, ARIA, or curation notebook.
- `WANDB_API_KEY` and `CWSANDBOX_API_KEY` never appear in a frontend response, a log line, or a logger call. The TLS private key does not either.
- With Forge off (`forge_configured` false), `POST /evaluations` returns 400 and the Evaluate page keeps the placeholder. Local inference on `/inference` stays as it is.
- A judge error, an unparseable reply, or confidence below `0.6` is `needs_review`. It is never success.
- One trial holds the arm. Recording, teleoperation, replay, inference, and a second trial stay mutually exclusive.
- Policies use `wrist` and `top`. The judge reads `top` and `side`.
- Live Forge tests stay skipped unless `RUN_FORGE_LIVE=1`. Unit tests use fakes and must not open a network connection or import `cwsandbox` from a pure helper.
- New Python modules start with the Apache-2.0 header in `tests/test_forge_config.py` lines 1–13.
- Pytest is not on `PATH`. Run `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest`.
- Commit only the files the task lists. Push only to `origin` (`https://github.com/Ivo-B/leLab.git`). Do not push `upstream`. Do not remove the `github.repository == 'huggingface/leLab'` guard in `.github/workflows/sync_space.yml`.
- Work on the checked-out `forge/m4-evaluate` branch.

---

### Task 1: Judge verdict and frame cap

**Files:**
- Create: `lelab/forge/judge.py`
- Create: `tests/test_forge_judge.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `STAGES = ("started", "reached", "grasped", "delivered")`, `CONFIDENCE_FLOOR = 0.6`, `MAX_FRAME_BYTES = 200_000`, `JudgeVerdict` with fields `stage: str | None`, `confidence: float | None`, `status: str` (`"scored"` or `"needs_review"`). `parse_judge_payload(payload: dict) -> JudgeVerdict`. `frame_data_url(name: str, jpeg: bytes) -> str`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_forge_judge.py` with the Apache-2.0 header, then:

```python
import pytest

from lelab.forge.judge import (
    CONFIDENCE_FLOOR,
    MAX_FRAME_BYTES,
    STAGES,
    frame_data_url,
    parse_judge_payload,
)


def test_stage_ladder_and_floor() -> None:
    assert STAGES == ("started", "reached", "grasped", "delivered")
    assert CONFIDENCE_FLOOR == 0.6
    assert MAX_FRAME_BYTES == 200_000


def test_delivered_above_the_floor_is_scored() -> None:
    verdict = parse_judge_payload({"stage": "delivered", "confidence": 0.91})
    assert verdict.status == "scored"
    assert verdict.stage == "delivered"
    assert verdict.confidence == 0.91


def test_low_confidence_needs_review() -> None:
    verdict = parse_judge_payload({"stage": "grasped", "confidence": 0.59})
    assert verdict.status == "needs_review"
    assert verdict.stage == "grasped"


def test_unknown_stage_or_missing_confidence_needs_review() -> None:
    assert parse_judge_payload({"stage": "success", "confidence": 1}).status == "needs_review"
    assert parse_judge_payload({"stage": "delivered"}).status == "needs_review"
    assert parse_judge_payload({"stage": "delivered", "confidence": 1.2}).stage is None


def test_frame_data_url_rejects_an_oversized_jpeg() -> None:
    url = frame_data_url("top", b"\xff\xd8\xff")
    assert url.startswith("data:image/jpeg;base64,")
    with pytest.raises(ValueError, match="top"):
        frame_data_url("top", b"x" * (MAX_FRAME_BYTES + 1))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_judge.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'lelab.forge.judge'`

- [ ] **Step 3: Write the minimal implementation**

Create `lelab/forge/judge.py` with the Apache-2.0 header, then:

```python
import base64
from dataclasses import dataclass

STAGES = ("started", "reached", "grasped", "delivered")
CONFIDENCE_FLOOR = 0.6
MAX_FRAME_BYTES = 200_000


@dataclass(frozen=True)
class JudgeVerdict:
    stage: str | None
    confidence: float | None
    status: str


def parse_judge_payload(payload: dict) -> JudgeVerdict:
    stage = payload.get("stage")
    raw = payload.get("confidence")
    if stage not in STAGES or isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return JudgeVerdict(None, None, "needs_review")
    confidence = float(raw)
    if confidence < 0 or confidence > 1:
        return JudgeVerdict(None, None, "needs_review")
    status = "scored" if confidence >= CONFIDENCE_FLOOR else "needs_review"
    return JudgeVerdict(stage, confidence, status)


def frame_data_url(name: str, jpeg: bytes) -> str:
    if len(jpeg) > MAX_FRAME_BYTES:
        raise ValueError(f"{name} frame exceeds {MAX_FRAME_BYTES} bytes")
    encoded = base64.b64encode(jpeg).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_judge.py -q`

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add lelab/forge/judge.py tests/test_forge_judge.py docs/superpowers/plans/2026-11-13-lelab-forge-m4-evaluate.md
git commit -m "$(cat <<'EOF'
feat: parse a stage-ladder judge verdict

EOF
)"
```

---

### Task 2: Judge client

**Files:**
- Modify: `lelab/forge/judge.py`
- Modify: `lelab/utils/forge_config.py`
- Modify: `tests/test_forge_judge.py`
- Modify: `tests/test_forge_config.py`

**Interfaces:**
- Consumes: `parse_judge_payload`, `frame_data_url`, `JudgeVerdict`, `ForgeConfig.api_key`, `ForgeConfig.entity`, `ForgeConfig.project`.
- Produces: `ForgeConfig.judge_model: str` from `FORGE_JUDGE_MODEL`, default `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B`. `judge_frames(client, cfg, top_jpeg: bytes, side_jpeg: bytes) -> JudgeVerdict`. The client is any object with `chat.completions.create(**kwargs) -> response` whose `response.choices[0].message.content` is a JSON string. A transport exception or bad JSON returns `needs_review` and does not raise. The request uses `model=cfg.judge_model` and two image parts named by the data URLs. `WANDB_API_KEY` is not read inside `judge_frames`; the caller builds the client.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_forge_judge.py`:

```python
import json

from lelab.forge.judge import judge_frames
from lelab.utils.forge_config import ForgeConfig


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, content: str | None, error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.kwargs: dict = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return _Response(self.content or "")


class _Chat:
    def __init__(self, completions: _Completions) -> None:
        self.completions = completions


class _Client:
    def __init__(self, content: str | None, error: Exception | None = None) -> None:
        self.chat = _Chat(_Completions(content, error))


def _cfg() -> ForgeConfig:
    return ForgeConfig(api_key="wandb-key", entity="team", project="so101-kitting", judge_model="judge-model")


def test_judge_frames_sends_top_and_side_and_parses_json() -> None:
    client = _Client(json.dumps({"stage": "reached", "confidence": 0.8}))
    verdict = judge_frames(client, _cfg(), b"\xff\xd8top", b"\xff\xd8side")
    assert verdict.status == "scored"
    assert verdict.stage == "reached"
    assert client.chat.completions.kwargs["model"] == "judge-model"
    content = client.chat.completions.kwargs["messages"][0]["content"]
    assert content[1]["type"] == "image_url"
    assert content[2]["type"] == "image_url"


def test_judge_transport_error_needs_review() -> None:
    client = _Client(None, error=RuntimeError("wandb-key refused"))
    verdict = judge_frames(client, _cfg(), b"\xff\xd8top", b"\xff\xd8side")
    assert verdict.status == "needs_review"
    assert verdict.stage is None
```

Add to `tests/test_forge_config.py`:

```python
def test_judge_model_defaults_and_can_be_overridden(monkeypatch) -> None:
    monkeypatch.delenv("FORGE_JUDGE_MODEL", raising=False)
    assert (
        load_forge_config().judge_model
        == "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B"
    )
    monkeypatch.setenv("FORGE_JUDGE_MODEL", "google/gemma-4-26B-A4B-it")
    assert load_forge_config().judge_model == "google/gemma-4-26B-A4B-it"
```

`ForgeConfig(...)` constructions in existing tests must keep working. Give `judge_model` a default on the dataclass so those calls stay valid.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_judge.py tests/test_forge_config.py -q`

Expected: FAIL with `ImportError` or `TypeError` for the missing symbol and field.

- [ ] **Step 3: Write the minimal implementation**

Add to `ForgeConfig` and `load_forge_config`:

```python
judge_model: str = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B"
```

```python
judge_model=_blank_to_none(os.environ.get("FORGE_JUDGE_MODEL"))
or "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B",
```

Append to `lelab/forge/judge.py`:

```python
import json

_NEEDS_REVIEW = JudgeVerdict(None, None, "needs_review")


def judge_frames(client, cfg, top_jpeg: bytes, side_jpeg: bytes) -> JudgeVerdict:
    try:
        response = client.chat.completions.create(
            model=cfg.judge_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Score this robot trial. Reply with JSON "
                                '{"stage":"started|reached|grasped|delivered","confidence":0.0} '
                                "and no other text. The first image is the top camera. "
                                "The second image is the side camera."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": frame_data_url("top", top_jpeg)}},
                        {"type": "image_url", "image_url": {"url": frame_data_url("side", side_jpeg)}},
                    ],
                }
            ],
        )
        payload = json.loads(response.choices[0].message.content)
    except Exception:
        return _NEEDS_REVIEW
    if not isinstance(payload, dict):
        return _NEEDS_REVIEW
    return parse_judge_payload(payload)
```

`except Exception` must not log the exception, because the message can contain the API key.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_judge.py tests/test_forge_config.py -q`

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add lelab/forge/judge.py lelab/utils/forge_config.py tests/test_forge_judge.py tests/test_forge_config.py
git commit -m "$(cat <<'EOF'
feat: score a trial with the Serverless Inference judge

EOF
)"
```

---

### Task 3: Weave trial row

**Files:**
- Create: `lelab/forge/weave_eval.py`
- Create: `tests/test_forge_weave_eval.py`

**Interfaces:**
- Consumes: nothing from the judge module. The session passes plain values.
- Produces: `TrialRow` with `skill: str`, `stage: str | None`, `verdict_source: str`, `duration_s: float`, `chunk_latency_s: float | None`, `policy_host: str`, `top_video: str | None`, `side_video: str | None`. `WeaveTrialLog.log_trial(row: TrialRow) -> None` and `WeaveTrialLog.finish() -> str`. The constructor takes `logger`. `logger.log_prediction(inputs, output)` returns an object with `log_score(name, value)` and `finish()`. `logger.log_summary(summary)` is called once from `finish()`. `finish()` returns `https://wandb.ai/{entity}/{project}/weave`.

- [ ] **Step 1: Write the failing test**

```python
from lelab.forge.weave_eval import TrialRow, WeaveTrialLog


class _Prediction:
    def __init__(self) -> None:
        self.scores: dict[str, object] = {}
        self.finished = False

    def log_score(self, name: str, value: object) -> None:
        self.scores[name] = value

    def finish(self) -> None:
        self.finished = True


class _Logger:
    def __init__(self) -> None:
        self.predictions: list[tuple[dict, dict, _Prediction]] = []
        self.summary: dict | None = None

    def log_prediction(self, inputs: dict, output: dict) -> _Prediction:
        prediction = _Prediction()
        self.predictions.append((inputs, output, prediction))
        return prediction

    def log_summary(self, summary: dict) -> None:
        self.summary = summary


def test_one_row_records_the_spec_fields_and_the_forge_url() -> None:
    logger = _Logger()
    log = WeaveTrialLog(logger, entity="team", project="so101-kitting")
    log.log_trial(
        TrialRow(
            skill="bracket",
            stage="grasped",
            verdict_source="judge",
            duration_s=12.5,
            chunk_latency_s=None,
            policy_host="local",
            top_video="/tmp/top.mp4",
            side_video="/tmp/side.mp4",
        )
    )
    inputs, output, prediction = logger.predictions[0]
    assert inputs["skill"] == "bracket"
    assert output["stage"] == "grasped"
    assert output["verdict_source"] == "judge"
    assert output["policy_host"] == "local"
    assert prediction.scores["needs_review"] == 0
    assert prediction.finished is True
    assert log.finish() == "https://wandb.ai/team/so101-kitting/weave"
    assert logger.summary == {"trials": 1}


def test_needs_review_score_is_one_when_the_stage_is_missing() -> None:
    logger = _Logger()
    log = WeaveTrialLog(logger, entity="team", project="so101-kitting")
    log.log_trial(
        TrialRow("bolt", None, "judge", 4.0, None, "coreweave", None, None)
    )
    assert logger.predictions[0][2].scores["needs_review"] == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_weave_eval.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the minimal implementation**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class TrialRow:
    skill: str
    stage: str | None
    verdict_source: str
    duration_s: float
    chunk_latency_s: float | None
    policy_host: str
    top_video: str | None
    side_video: str | None


class WeaveTrialLog:
    def __init__(self, logger, entity: str, project: str) -> None:
        self._logger = logger
        self._entity = entity
        self._project = project
        self._count = 0

    def log_trial(self, row: TrialRow) -> None:
        prediction = self._logger.log_prediction(
            inputs={"skill": row.skill, "policy_host": row.policy_host},
            output={
                "stage": row.stage,
                "verdict_source": row.verdict_source,
                "duration_s": row.duration_s,
                "chunk_latency_s": row.chunk_latency_s,
                "top_video": row.top_video,
                "side_video": row.side_video,
            },
        )
        prediction.log_score("needs_review", 1 if row.stage is None else 0)
        prediction.finish()
        self._count += 1

    def finish(self) -> str:
        self._logger.log_summary({"trials": self._count})
        return f"https://wandb.ai/{self._entity}/{self._project}/weave"
```

Add the Apache-2.0 header. Do not import `weave` in this module. The session constructs `weave.EvaluationLogger` later and passes it in.

- [ ] **Step 4: Run the test to verify it passes**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_weave_eval.py -q`

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add lelab/forge/weave_eval.py tests/test_forge_weave_eval.py
git commit -m "$(cat <<'EOF'
feat: record one Weave row per evaluation trial

EOF
)"
```

---

### Task 4: Policy-server Sandbox description

**Files:**
- Create: `lelab/forge/policy_server.py`
- Create: `tests/test_forge_policy_server.py`

**Interfaces:**
- Consumes: `ForgeConfig` sandbox resource fields. Does not import `cwsandbox`.
- Produces: `POLICY_PORT = 7447`, `policy_service() -> dict` with `port`, `name` `"policy"`, `visibility` `"public"`, `endpoint` `{"kind": "tls_passthrough"}`. `assigned_endpoint(addresses) -> str` reads an object with `.name`, `.address` and returns `tls/<host>:<port>` for the `policy` entry. `zenoh_server_config() -> str` is JSON5-compatible JSON containing `tls/0.0.0.0:7447`, `listen_private_key` `/work/key.pem`, `listen_certificate` `/work/cert.pem`, `root_ca_certificate` `/work/ca.pem`, `enable_mtls` false. `zenoh_client_config() -> str` contains `verify_name_on_connect` true and `root_ca_certificate` `/work/ca.pem` on the client side path string `/tmp/lelab-policy-ca.pem`. `policy_server_argv() -> list[str]` is `["lerobot-policy-server", "--config_path=/work/server.yaml", "--model.device=cuda"]`.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_policy_server.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the minimal implementation**

```python
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
```

Add the Apache-2.0 header. Do not generate certificates in this task.

- [ ] **Step 4: Run the test to verify it passes**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_policy_server.py -q`

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add lelab/forge/policy_server.py tests/test_forge_policy_server.py
git commit -m "$(cat <<'EOF'
feat: describe the Zenoh policy server Sandbox

EOF
)"
```

---

### Task 5: Policy host lifecycle

**Files:**
- Modify: `lelab/forge/policy_server.py`
- Modify: `lelab/runners/cwsandbox_client.py`
- Create: `tests/test_forge_policy_host.py`

**Interfaces:**
- Consumes: `policy_service`, `assigned_endpoint`, `policy_server_argv`, `zenoh_server_config`, `Sandbox.run` kwargs style already used by `CwsandboxClient.create`.
- Produces: `PolicyHost.start(checkpoint: str) -> str` returns the `tls/...` endpoint. `PolicyHost.stop() -> None` calls `handle.stop()`. Create failure raises `RuntimeError` whose message is `Could not start the CoreWeave policy server: <redacted>. Use the local policy host instead.` and contains neither API key. The Sandbox is created with `args ["sleep", "infinity"]`, tag `lelab`, the same resources and lifetime as training, and `services` equal to `[policy_service()]` inside the existing `create(kwargs)` dict. Do not add a second positional argument. `CwsandboxClient.create` reads `kwargs.get("services")` and passes `services=` to `Sandbox.run` only when that list is non-empty. Training creates omit the key, so their `Sandbox.run` call stays unchanged. After create, the host writes `/work/zenoh.json5` and then execs `policy_server_argv()`.

The certificate bytes are injected through `PolicyHost(client, cfg, certificates)`, where `certificates` is `{"ca.pem": bytes, "key.pem": bytes, "cert.pem": bytes}`. This task does not shell out to `openssl`. A missing `policy` address raises the same redacted `RuntimeError`.

- [ ] **Step 1: Write the failing test**

Use a fake client. `create` returns a handle recording `write_file` calls and `exec` argv. `service_addresses` is a list of objects with `name` and `address`. A second test makes `create` raise `RuntimeError("bad CWSANDBOX_API_KEY")` and asserts the public message and that `CWSANDBOX_API_KEY` is absent from it.

Assert `create` kwargs include `services` equal to `policy_service()`, `tags` `["lelab"]`, and `args` `["sleep", "infinity"]`. Assert the first `write_file` path is `/work/zenoh.json5` and the exec argv equals `policy_server_argv()`. Assert `stop()` calls `handle.stop()`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_policy_host.py -q`

Expected: FAIL with `ImportError` for `PolicyHost`.

- [ ] **Step 3: Write the minimal implementation**

Keep `CwsandboxClient.create(self, kwargs)`. When `kwargs.get("services")` is a non-empty list, pass `services=` into `Sandbox.run`. Training kwargs omit the key, so that call must not gain `services=None`. Build the SDK objects inside the client:

```python
if services:
    from cwsandbox import Endpoint, EndpointKind, Service, ServiceVisibility

    run_services = []
    for item in services:
        endpoint = item.get("endpoint")
        run_services.append(
            Service(
                port=item["port"],
                name=item["name"],
                visibility=ServiceVisibility.PUBLIC,
                endpoint=Endpoint(kind=EndpointKind.TLS_PASSTHROUGH) if endpoint else None,
            )
        )
```

Pass `services=run_services` only when the list is non-empty. Existing training creates omit `services`, so `Sandbox.run` must not receive `services=None`.

`PolicyHost.start` uses the existing `redact_secrets` from `lelab.forge.sandbox`. On any exception before a handle exists, raise the local-host sentence. After a handle exists, write the three PEM files to `/work/ca.pem`, `/work/key.pem`, `/work/cert.pem`, write `zenoh_server_config()` to `/work/zenoh.json5`, then exec. If exec or address lookup fails, `stop()` the handle and raise the same sentence.

- [ ] **Step 4: Run the test to verify it passes**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_policy_host.py tests/test_forge_sandbox_launch.py -q`

Expected: the new tests pass and the existing launch tests still pass.

- [ ] **Step 5: Commit**

```bash
git add lelab/forge/policy_server.py lelab/runners/cwsandbox_client.py tests/test_forge_policy_host.py
git commit -m "$(cat <<'EOF'
feat: start and stop a CoreWeave policy server

EOF
)"
```

---

### Task 6: Evaluation session

**Files:**
- Create: `lelab/evaluate.py`
- Modify: `lelab/server.py`
- Create: `tests/test_forge_evaluate.py`

**Interfaces:**
- Consumes: `JudgeVerdict`, `TrialRow`, `WeaveTrialLog`, `parse_judge_payload` is not called here; the session receives a `JudgeVerdict`.
- Produces: `EvaluationSession.start_trial(skill: str) -> dict`, `finish_trial(verdict: JudgeVerdict, duration_s: float) -> dict`, `override(stage: str | None) -> dict`, `close() -> dict` with `weave_url`. `POST /evaluations` body `{"policy_family":"smolvla"|"act","version":"candidate"|"production","policy_host":"local"|"coreweave","trials_per_skill":10}`. Forge off returns 400 `Evaluation needs Weights & Biases`. `trials_per_skill` outside 1..50 returns 400. A second `POST` while a session is open returns 409. `POST /evaluations/current/override` body `{"stage":"grasped"}` or `{"stage":null}` for needs-review. Unknown stage returns 400.

The session does not start the arm in this task. `start_trial` records the skill and returns `{"skill": skill, "index": n, "status": "running"}`. `finish_trial` stores the verdict. `override` sets `verdict_source` to `operator` and replaces the stage; `null` leaves the stage empty and the Weave score `needs_review` at 1. `close` calls `log.finish()`.

Skills for a session are `bolt`, `bracket`, and `gear` when `policy_family` is `smolvla`. For `act` the body also requires `skill` and the session contains only that skill. Each skill repeats `trials_per_skill` times.

- [ ] **Step 1: Write the failing test**

Cover: Forge off is 400; `trials_per_skill` 0 is 400; a scored verdict logs `needs_review` 0; an override to `grasped` changes the stored stage and source; a second session is 409; `close` returns the Weave URL from the fake logger.

Use FastAPI `TestClient` the same way `tests/test_forge_config.py` does. Inject the logger by constructing `EvaluationSession` in the unit test and also hit the route with `monkeypatch` of the module-level session factory so the route test does not import `weave`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_evaluate.py -q`

Expected: FAIL with `ModuleNotFoundError` or a missing route.

- [ ] **Step 3: Write the minimal implementation**

`lelab/evaluate.py` holds one module-level session under a `threading.Lock`. Routes in `server.py` call `handle_start_evaluation`, `handle_override`, and `handle_close_evaluation`. Map `ValueError` to HTTP 400 and a busy session to HTTP 409, matching the training route style.

`finish_trial` builds a `TrialRow` with `policy_host` from the session, `chunk_latency_s=None`, and video paths `None`. Those paths are filled by a later hardware pass; this task's contract is the row and the operator override.

- [ ] **Step 4: Run the test to verify it passes**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_evaluate.py -q`

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add lelab/evaluate.py lelab/server.py tests/test_forge_evaluate.py
git commit -m "$(cat <<'EOF'
feat: open an evaluation session and record the operator verdict

EOF
)"
```

---

### Task 7: Evaluate page

**Files:**
- Create: `frontend/src/pages/Evaluate.tsx`
- Create: `frontend/src/pages/Evaluate.test.tsx`
- Modify: `frontend/src/App.tsx`

**Interfaces:**
- Consumes: `GET /forge-status` fields `forge_configured` and `sandbox_configured`. `POST /evaluations`, `POST /evaluations/current/override`, `POST /evaluations/current/close`.
- Produces: when `forge_configured` is not strictly `true`, the existing placeholder copy remains: `Real-arm evaluation in W&B Weave arrives in a later milestone. Rollouts still run from the inference page.` When it is true, the page shows a title `Evaluate`, one primary button `Start evaluation`, a policy-family select, a version select with `candidate` and `production`, a host select, and a trials input defaulting to 10. CoreWeave is disabled when `sandbox_configured !== true`, with the hint `Add CWSANDBOX_API_KEY and CWSANDBOX_TRAIN_IMAGE to the backend environment.` After a trial result is on screen, buttons `Confirm` and `Needs review` are visible. `Needs review` posts `{"stage": null}`. The empty state, before a session exists, reads `Choose a checkpoint, then score it on the arm.` Status text includes the word of the stage, not a color alone.

- [ ] **Step 1: Write the failing test**

Use the existing Testing Library setup. Mock `fetchWithHeaders` through the `ApiProvider` pattern already used by `frontend/src/pages` tests, or render `Evaluate` with a stub `fetch` if that is how `Setup` is tested. Assert the placeholder when `forge_configured` is false. Assert `Start evaluation` and the empty-state sentence when it is true. Assert the CoreWeave option is disabled when `sandbox_configured` is false.

- [ ] **Step 2: Run the test to verify it fails**

Run from `frontend/`: `npm test -- src/pages/Evaluate.test.tsx`

Expected: FAIL because `Evaluate.tsx` does not exist.

- [ ] **Step 3: Write the minimal implementation**

Replace the `/evaluate` route element with `<Evaluate />`. Keep the placeholder paragraph as a branch inside `Evaluate` so the copy stays exact. Do not add a new font, a teal color, or a status that is color without a text label.

- [ ] **Step 4: Run the test to verify it passes**

Run from `frontend/`: `npm test -- src/pages/Evaluate.test.tsx`

Expected: the new tests pass. Then `npm test` and expect the existing 91 tests plus the new file to pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/Evaluate.tsx frontend/src/pages/Evaluate.test.tsx frontend/src/App.tsx
git commit -m "$(cat <<'EOF'
feat: start a real-arm evaluation from the Evaluate page

EOF
)"
```

---

## Self-review

- Spec section 5 step 5 (evaluate plan, judge ladder, operator override, Weave row, policy host) is Tasks 1–7. Section 8.1 judge and policy-server failures are Tasks 2 and 5. Section 4.2 `judge.py`, `weave_eval.py`, and `policy_server.py` are those files.
- The gRPC check in spec section 11 is answered in the verified block: the v0.7 server is Zenoh, and only TLS passthrough can carry it. The plan does not claim a measured Europe latency.
- M5 and M6 stay out of this plan. Chunk latency is stored as `null` until a rollout log line exists to parse.
- Placeholder scan: Task 5 and Task 6 describe the fake and the route table in prose where the surrounding tests already fix the assertions. An implementer who needs a line of test code that this plan does not paste must copy the assertion list in the task instead of inventing extra routes.
