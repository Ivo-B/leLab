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

"""CoreWeave Sandbox runner: streams a GPU Sandbox training into the job log.

The runner talks to an injected client (`client.create(kwargs) -> handle`), so
it never imports `cwsandbox`. Startup runs on a worker thread because creating
a Sandbox can take a while and `start()` must return immediately.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from ..forge.sandbox import dumps_command, redact_secrets, sandbox_create_kwargs, training_command
from ..jobs import LogLine, SubprocessJobRunner, TrainingMetrics
from ..train import TrainingRequest
from ..utils.forge_config import ForgeConfig, load_forge_config

logger = logging.getLogger(__name__)

ARGV_PATH = "/work/train_argv.json"
LOCAL_FALLBACK = "Run this training on this machine instead."
_SIGTERM_RETURNCODE = -15


class SandboxJobRunner(SubprocessJobRunner):
    """Run a training in a CoreWeave Sandbox. Single-shot: one runner per job."""

    def __init__(self, metrics: TrainingMetrics, log_file_path: Path, client: Any) -> None:
        super().__init__(metrics, log_file_path)
        self._client = client
        # Guards this runner's log writes and the close of `_log_file`.
        self._log_lock = threading.Lock()
        self._handle: Any = None
        self._worker: threading.Thread | None = None
        self._returncode: int | None = None
        self._failure: str | None = None
        # Set once the worker has stored a handle or has exited without one.
        self._handle_settled = threading.Event()

    def start(self, job_id: str, config: TrainingRequest, output_dir: str) -> None:
        if self._worker is not None:
            raise RuntimeError(f"{type(self).__name__} already started")
        self._worker = threading.Thread(
            target=self._run, args=(config,), name=f"sandbox-runner-{job_id}", daemon=True
        )
        self._worker.start()

    def failure_message(self) -> str | None:
        return self._failure

    def is_running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def returncode(self) -> int | None:
        if self._worker is None or self._worker.is_alive():
            return None
        return self._returncode

    def pid(self) -> int | None:
        return None

    def stop(self, *, wait_seconds: float = 60) -> None:
        if self._worker is None:
            return
        self._stop_event.set()
        # A create() that never returns must not hang the caller forever.
        if not self._handle_settled.wait(timeout=wait_seconds):
            return
        handle = self._handle
        if handle is None:
            return
        try:
            handle.stop()
        except Exception as exc:
            self._report(f"Could not stop the CoreWeave Sandbox: {self._redact(str(exc))}")

    # -- internals --

    def _run(self, config: TrainingRequest) -> None:
        cfg: ForgeConfig | None = None
        try:
            self._open_log_file()
            cfg = load_forge_config()
            try:
                kwargs = sandbox_create_kwargs(cfg, config.skill)
                handle = self._client.create(kwargs)
            except Exception as exc:
                self._fail(
                    f"Could not start a CoreWeave Sandbox: {redact_secrets(str(exc), cfg)}. {LOCAL_FALLBACK}"
                )
                return
            self._handle = handle
            self._handle_settled.set()
            if self._stop_event.is_set():
                self._returncode = _SIGTERM_RETURNCODE
                return
            try:
                command = training_command(config)
                handle.write_file(ARGV_PATH, dumps_command(command))
                process = handle.exec(command)
                stream_errors: list[str] = []
                self._consume_lines(self._redacted(process.stdout, cfg, stream_errors))
                if stream_errors:
                    self._fail(f"The CoreWeave Sandbox training failed: {stream_errors[0]}")
                    return
                code = process.returncode
                if isinstance(code, int):
                    self._returncode = code
                else:
                    self._returncode = _SIGTERM_RETURNCODE if self._stop_event.is_set() else 1
            except Exception as exc:
                self._fail(f"The CoreWeave Sandbox training failed: {redact_secrets(str(exc), cfg)}")
        except Exception as exc:  # never let the worker die silently
            self._fail(f"The CoreWeave Sandbox runner failed: {self._redact(str(exc), cfg)}")
        finally:
            self._close_log_file()
            self._handle_settled.set()

    def _redacted(self, lines: Iterable[str], cfg: ForgeConfig, errors: list[str]) -> Iterator[str]:
        """Redact each line. A stream error is recorded (redacted) in `errors`
        and ends the iteration, so `_consume_lines` never logs the raw exception."""
        try:
            for line in lines:
                yield redact_secrets(line, cfg)
        except Exception as exc:
            errors.append(self._redact(str(exc), cfg))

    def _redact(self, text: str, cfg: ForgeConfig | None = None) -> str:
        try:
            return redact_secrets(text, cfg or load_forge_config())
        except Exception:
            return "[redacted]"

    def _fail(self, message: str) -> None:
        self._returncode = 1
        self._failure = message
        self._report(message)

    def _report(self, message: str) -> None:
        """Surface an already-redacted message in the log and the live queue."""
        logger.warning("%s", message)
        line = LogLine(timestamp=time.time(), message=message)
        payload = line.model_dump_json() + "\n"
        with self._log_lock:
            file = self._log_file
            try:
                if file is not None and not file.closed:
                    file.write(payload)
                    file.flush()
                elif self._log_file_path is not None:
                    # The worker already closed the log (or stop() raced it): append
                    # the line through a short-lived handle so it is not lost.
                    with self._log_file_path.open("a") as reopened:
                        reopened.write(payload)
                        reopened.flush()
            except Exception:  # best-effort persist; the message is already redacted
                logger.warning("Could not write to the job log file")
        self._log_queue.put(line)

    def _close_log_file(self) -> None:
        with self._log_lock:
            file = self._log_file
            self._log_file = None
            if file is not None:
                with contextlib.suppress(Exception):
                    file.close()
