# SPDX-FileCopyrightText: Copyright (c) 2026 AuricTW
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original one-off Long-WAM contributor validation.
"""Repeat gated transport races against the original module, without ML imports."""

import hashlib
import json
import multiprocessing as mp
import os
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = Path(os.environ["LONGWAM_WORKER_PATH"])
sys.path.insert(0, str(MODULE.parent))
import worker


def initialize(gates):
    """Provide a gated CPU echo handler in the real child process."""

    def handle(operation, payload):
        if operation == "block":
            gates[0].set()
            if not gates[1].wait(10):
                raise RuntimeError("gate was not released")
        return payload

    return handle, {}


class BadPayload:
    """Fail inside Connection.send's real pickler."""

    def __reduce__(self):
        raise TypeError("gated serialization failure")


class DeferredSend:
    """Ensure the receiver is already polling before permitting serialization."""

    def __init__(self, connection):
        self.connection = connection
        self.release = threading.Event()
        self.waiting = threading.Event()

    def send(self, message):
        if not self.release.wait(10):
            raise RuntimeError("send gate was not released")
        self.connection.send(message)

    def poll(self, timeout=0):
        if timeout > 0:
            self.waiting.set()
        return self.connection.poll(timeout)

    def __getattr__(self, name):
        return getattr(self.connection, name)


def blocked_failure_trial():
    """Wake a result(None) that was already blocked before the sender failed."""
    client = worker.SpawnedInferenceProcess(
        initialize, None, name="stress-wake", startup_timeout_s=10
    )
    connection = DeferredSend(client._connection)
    client._connection = connection
    pending = client.submit("echo", BadPayload())
    finished, errors = threading.Event(), []

    def receive():
        try:
            pending.result()
        except worker.RemoteInferenceError as error:
            errors.append(error)
        finally:
            finished.set()

    receiver = threading.Thread(target=receive, daemon=True)
    receiver.start()
    try:
        assert connection.waiting.wait(3), "receive did not enter a blocking poll"
        assert not finished.is_set()
        connection.release.set()
        assert finished.wait(3), "sender failure did not wake the blocked receiver"
        assert len(errors) == 1 and isinstance(errors[0], worker.RemoteInferenceError)
        assert isinstance(errors[0].__cause__, TypeError)
        assert pending.done()
    finally:
        connection.release.set()
        client.close(force=True)
        receiver.join(timeout=3)
    assert (
        not receiver.is_alive()
        and not client._sender.is_alive()
        and not client._process.is_alive()
    )


def finite_timeout_trial():
    """Keep a request retryable across a deadline spanning several poll periods."""
    context = mp.get_context("spawn")
    gates = context.Event(), context.Event()
    client = worker.SpawnedInferenceProcess(
        initialize, gates, name="stress-timeout", startup_timeout_s=10
    )
    try:
        pending = client.submit("block", "completed")
        assert gates[0].wait(3)
        try:
            pending.result(timeout=0.12)
        except TimeoutError:
            pass
        else:
            raise AssertionError("blocked request did not time out")
        assert not pending.done()
        gates[1].set()
        assert pending.result(timeout=3) == "completed"
        assert client.call("echo", "next") == "next"
    finally:
        gates[1].set()
        client.close()
    assert not client._sender.is_alive() and not client._process.is_alive()


def main():
    """Retain exact source identity and observed, bounded race-test outcomes."""
    for _ in range(10):
        blocked_failure_trial()
        finite_timeout_trial()
    report = {
        "python": sys.version,
        "source_sha256_lf": hashlib.sha256(
            MODULE.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        "blocked_receive_sender_failure": {"passed": 10, "failed": 0},
        "multi_poll_timeout_and_retry": {"passed": 10, "failed": 0},
        "method": "Actual unchanged worker module imported directly; real spawned CPU processes and pipes; gates inject timing and serialization faults; no ML imports or GPU/model validation",
    }
    (Path(os.environ["LONGWAM_EVIDENCE_DIR"]) / "stress-results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
