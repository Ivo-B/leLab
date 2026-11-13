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
