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
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-keyout",
                str(key_path),
                "-out",
                str(cert_path),
                "-days",
                "1",
                "-nodes",
                "-subj",
                f"/CN={hostname}",
                "-addext",
                f"subjectAltName=DNS:{hostname}",
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("Could not issue the policy server certificate")
        certificate = cert_path.read_bytes()
        return {"ca.pem": certificate, "cert.pem": certificate, "key.pem": key_path.read_bytes()}
