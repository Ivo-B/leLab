# LeLab Forge M4 — Live Trial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The policy-server certificate is issued for the Sandbox hostname after that hostname exists, and Run trial sends one skill through the judge and shows the stage on the Evaluate page.

**Architecture:** `issue_certificates(hostname)` shells out to OpenSSL and returns a self-signed pair whose SAN is that hostname. `PolicyHost.start` reads `service_addresses` before it writes any PEM. The trial route calls `start_trial`, a replaceable capture function, then `judge_frames`, then `finish_trial`. The page shows the returned stage and keeps Confirm disabled until a verdict exists.

**Tech Stack:** Python 3.12, OpenSSL 3 on the Mac, FastAPI, the existing `PolicyHost` and `EvaluationSession`, React + Vite.

## Global Constraints

- Do not start M5 or M6. No order agent, Agent Lens, Automations, ARIA, or curation notebook.
- Do not start the robot arm in a unit test. The default capture function returns no frames. A missing frame is `needs_review`, never success.
- `WANDB_API_KEY`, `CWSANDBOX_API_KEY`, and the TLS private key never appear in a response, a log line, or a logger call.
- The certificate is created after `service_addresses` is known. The hostname is the address with the `:port` removed. `verify_name_on_connect` stays true.
- New Python modules start with the Apache-2.0 header in `tests/test_forge_config.py` lines 1–13.
- Pytest is not on `PATH`. Run `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest`.
- Commit only the files the task lists. Push only to `origin` (`https://github.com/Ivo-B/leLab.git`). Do not push `upstream`.
- Work on the checked-out `forge/m4-trial` branch.

---

### Task 1: Issue a certificate for a hostname

**Files:**

- Create: `lelab/forge/certificates.py`
- Create: `tests/test_forge_certificates.py`

**Interfaces:**

- Consumes: the `openssl` binary on `PATH`.
- Produces: `issue_certificates(hostname: str) -> dict[str, bytes]` with keys `ca.pem`, `cert.pem`, and `key.pem`. `ca.pem` and `cert.pem` are the same self-signed certificate. The certificate SAN contains `DNS:<hostname>`. A blank hostname, or one containing `/` or `:`, raises `ValueError` before OpenSSL runs.

- [ ] **Step 1: Write the failing test**

```python
import subprocess

import pytest

from lelab.forge.certificates import issue_certificates


def test_blank_or_unsafe_hostname_is_rejected() -> None:
    with pytest.raises(ValueError):
        issue_certificates("")
    with pytest.raises(ValueError):
        issue_certificates("host:443")
    with pytest.raises(ValueError):
        issue_certificates("a/b")


def test_certificate_covers_the_hostname() -> None:
    hostname = "8443-abc.ep2.cwsandbox.com"
    issued = issue_certificates(hostname)
    assert issued["ca.pem"] == issued["cert.pem"]
    assert b"PRIVATE KEY" in issued["key.pem"]
    text = subprocess.run(
        ["openssl", "x509", "-noout", "-ext", "subjectAltName"],
        input=issued["cert.pem"],
        check=True,
        capture_output=True,
    ).stdout.decode()
    assert f"DNS:{hostname}" in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_certificates.py -q`

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the minimal implementation**

```python
import subprocess
import tempfile
from pathlib import Path

def issue_certificates(hostname: str) -> dict[str, bytes]:
    if not hostname or "/" in hostname or ":" in hostname:
        raise ValueError("A certificate hostname must be a DNS name without a port")
    with tempfile.TemporaryDirectory() as directory:
        key_path = Path(directory) / "key.pem"
        cert_path = Path(directory) / "cert.pem"
        result = subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", str(key_path), "-out", str(cert_path),
                "-days", "1", "-nodes",
                "-subj", f"/CN={hostname}",
                "-addext", f"subjectAltName=DNS:{hostname}",
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("Could not issue the policy server certificate")
        certificate = cert_path.read_bytes()
        return {"ca.pem": certificate, "cert.pem": certificate, "key.pem": key_path.read_bytes()}
```

Do not log `result.stderr`. It is not returned.

- [ ] **Step 4: Run the test to verify it passes**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_certificates.py -q`

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add lelab/forge/certificates.py tests/test_forge_certificates.py docs/superpowers/plans/2026-11-13-lelab-forge-m4-trial.md
git commit -m "$(cat <<'EOF'
feat: issue a policy certificate for the Sandbox hostname

EOF
)"
```

---

### Task 2: Issue the certificate after the hostname is known

**Files:**

- Modify: `lelab/forge/policy_server.py`
- Modify: `tests/test_forge_policy_host.py`

**Interfaces:**

- Consumes: `issue_certificates(hostname) -> dict[str, bytes]`, `assigned_endpoint`.
- Produces: `PolicyHost(client, cfg, issuer=issue_certificates)`. `start` reads the endpoint, takes the host from `tls/<host>:<port>`, calls `issuer(host)`, then writes the PEMs. `host.client_ca_path` is the local file that received `ca.pem`. The default path is `/tmp/lelab-policy-ca.pem`, the same path `zenoh_client_config()` names. Existing tests pass `issuer=lambda hostname: CERTIFICATES` instead of a certificates dict.

- [ ] **Step 1: Write the failing test**

Add this test and change every `PolicyHost(..., CERTIFICATES)` call to `PolicyHost(..., issuer=lambda hostname: CERTIFICATES)`. Move the private-key leak assertion onto a write failure that happens after the issuer returns, because create fails before a certificate exists.

```python
def test_start_issues_the_certificate_for_the_assigned_host(tmp_path, monkeypatch) -> None:
    seen: list[str] = []

    def issuer(hostname: str) -> dict[str, bytes]:
        seen.append(hostname)
        return {"ca.pem": b"CA-for-" + hostname.encode(), "key.pem": b"KEY", "cert.pem": b"CERT"}

    handle = FakeHandle()
    ca_path = tmp_path / "ca.pem"
    monkeypatch.setattr("lelab.forge.policy_server.CLIENT_CA_PATH", str(ca_path))
    endpoint = PolicyHost(FakeClient(handle), _cfg(), issuer=issuer).start("me/checkpoint")

    assert endpoint == "tls/policy.example:443"
    assert seen == ["policy.example"]
    assert dict(handle.writes)["/work/ca.pem"] == b"CA-for-policy.example"
    assert ca_path.read_bytes() == b"CA-for-policy.example"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_policy_host.py::test_start_issues_the_certificate_for_the_assigned_host -q`

Expected: FAIL because `issuer` is not a parameter or `CLIENT_CA_PATH` does not exist.

- [ ] **Step 3: Write the minimal implementation**

Add `CLIENT_CA_PATH = "/tmp/lelab-policy-ca.pem"`. Change `PolicyHost.__init__` to `(self, client, cfg, issuer=issue_certificates)`. In `start`, after `create` succeeds:

```python
endpoint = assigned_endpoint(handle.service_addresses)
hostname = endpoint.removeprefix("tls/").rsplit(":", 1)[0]
self._certificates = self._issuer(hostname)
for name in ("ca.pem", "key.pem", "cert.pem"):
    handle.write_file(f"/work/{name}", self._certificates[name])
Path(CLIENT_CA_PATH).write_bytes(self._certificates["ca.pem"])
```

Then write the zenoh config and server yaml and exec, as today. A create failure still redacts with an empty certificate dict. Set `self._certificates` before the writes so a write error can redact the key.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_policy_host.py tests/test_forge_certificates.py -q`

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add lelab/forge/policy_server.py tests/test_forge_policy_host.py
git commit -m "$(cat <<'EOF'
fix: issue the policy certificate after the Sandbox hostname exists

EOF
)"
```

---

### Task 3: Run one trial through the judge

**Files:**

- Modify: `lelab/evaluate.py`
- Modify: `lelab/server.py`
- Modify: `tests/test_forge_evaluate.py`

**Interfaces:**

- Consumes: `EvaluationSession.start_trial`, `finish_trial`, `judge_frames`, `JudgeVerdict`.
- Produces: `POST /evaluations/current/trial` with body `{"skill": "bolt", "top_jpeg_base64": "...", "side_jpeg_base64": "...", "duration_s": 1.5}`. The response is `{"skill", "stage", "verdict_source"}`. Invalid base64 or an empty frame finishes the trial as `needs_review` and returns stage `null`. No session returns 400. The route does not import `cwsandbox` and does not open a subprocess.

- [ ] **Step 1: Write the failing test**

Monkeypatch `lelab.evaluate.judge_frames` to return `JudgeVerdict("grasped", 0.9, "scored")` for the happy path. Assert the response stage is `grasped` and one trial is running no longer. A second test posts `top_jpeg_base64` of `"@@"` and asserts `stage` is `null` and `verdict_source` is `judge`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_evaluate.py -q -k trial`

Expected: FAIL with 404 or an import error.

- [ ] **Step 3: Write the minimal implementation**

Decode both fields with `base64.b64decode(..., validate=True)`. On `ValueError` or an empty result, call `finish_trial(JudgeVerdict(None, None, "needs_review"), duration_s)` without calling `judge_frames`. Otherwise call `judge_frames(client, cfg, top, side)`. Build the client only in that branch, inside the function, from `openai.OpenAI(base_url="https://api.inference.wandb.ai/v1", api_key=cfg.api_key)`. Tests replace `judge_frames`, so they do not construct a client. Map `ValueError` to HTTP 400 the same way the other evaluation routes do.

- [ ] **Step 4: Run the test to verify it passes**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_evaluate.py -q`

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add lelab/evaluate.py lelab/server.py tests/test_forge_evaluate.py
git commit -m "$(cat <<'EOF'
feat: score one evaluation trial from the top and side frames

EOF
)"
```

---

### Task 4: Show the trial stage on the Evaluate page

**Files:**

- Modify: `frontend/src/pages/Evaluate.tsx`
- Modify: `frontend/src/pages/Evaluate.test.tsx`

**Interfaces:**

- Consumes: `POST /evaluations/current/trial`.
- Produces: a `Run trial` button while a session is open. It posts `{skill, top_jpeg_base64: "", side_jpeg_base64: "", duration_s: 0}` for the skill selected on an ACT session and `bolt` for SmolVLA. Confirm and Needs review stay disabled until the response arrives. The status text then includes the stage word, or `Needs review` when stage is null.

- [ ] **Step 1: Write the failing test**

Render Evaluate with `forge_configured: true`. Start a session. Assert Confirm is disabled. Click `Run trial`. The mocked trial response is `{skill: "bolt", stage: "grasped", verdict_source: "judge"}`. Assert the status text contains `grasped` and Confirm is enabled.

- [ ] **Step 2: Run the test to verify it fails**

Run from `frontend/`: `npm test -- src/pages/Evaluate.test.tsx`

Expected: FAIL because `Run trial` is not on the page.

- [ ] **Step 3: Write the minimal implementation**

Add the button next to Close evaluation. Disable Confirm and Needs review when `verdict === null`. Do not add a color-only status.

- [ ] **Step 4: Run the test to verify it passes**

Run from `frontend/`: `npm test -- src/pages/Evaluate.test.tsx`

Expected: the file passes.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/Evaluate.tsx frontend/src/pages/Evaluate.test.tsx
git commit -m "$(cat <<'EOF'
feat: show the judge stage after a trial

EOF
)"
```
