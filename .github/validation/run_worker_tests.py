# SPDX-FileCopyrightText: Copyright (c) 2026 AuricTW
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original one-off Long-WAM contributor validation.
"""Validate the exact PR commit on Linux without modifying its source."""

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path.cwd()
CANDIDATE = ROOT / "source"
BASELINE = ROOT / "baseline"
EVIDENCE = ROOT / "evidence"
EVIDENCE.mkdir(exist_ok=True)
FILES = {
    "Long-WAM/src/longwam/runtime/worker.py": "3e29763e05028694539abb416bf6cfe5645311b377fc083c50e95933ee08f150",
    "Long-WAM/src/longwam/runtime/README.md": "753073cc4297536ef211ae27464e95a34993225ea8a830ebf62c68ee28b138c8",
    "Long-WAM/tests/runtime/test_worker_transport.py": "d41bd2af95e266d05f4bd33f6785f93c82f6375b2bd6378acea67057367b8eab",
}


def write_json(name, value):
    (EVIDENCE / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run(name, args, cwd, env=None, expected=0):
    print(f"Running {name}: {args}", flush=True)
    with (
        (EVIDENCE / f"{name}.log").open("w", encoding="utf-8") as log,
        subprocess.Popen(
            args,
            cwd=cwd,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as process,
    ):
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        status = process.wait()
    assert status == expected, (name, status, expected)


def results(name):
    cases = {}
    for case in ET.parse(EVIDENCE / name).iter("testcase"):
        identity = case.get("classname", "") + "::" + case.get("name", "")
        status = "passed"
        for kind in ("failure", "error", "skipped"):
            if case.find(kind) is not None:
                status = kind
                break
        cases[identity] = status
    return cases


def main():
    assert sys.platform == "linux", sys.platform
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=CANDIDATE, text=True
        ).strip()
        == os.environ["CANDIDATE_SHA"]
    )
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=BASELINE, text=True
        ).strip()
        == os.environ["BASELINE_SHA"]
    )
    identities = {}
    for name, expected in FILES.items():
        digest = hashlib.sha256(
            (CANDIDATE / name).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        assert digest == expected, name
        identities[name] = digest
    write_json(
        "source-manifest.json",
        {
            "baseline_commit": os.environ["BASELINE_SHA"],
            "candidate_commit": os.environ["CANDIDATE_SHA"],
            "validation_workflow_commit": os.environ["GITHUB_SHA"],
            "normalized_source_sha256": identities,
        },
    )
    environment = {
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "python": sys.version,
        "distribution": platform.freedesktop_os_release(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "torchvision",
                "numpy",
                "pytest",
                "pytest-timeout",
                "hydra-core",
                "omegaconf",
            )
        },
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "github_run_id": os.environ["GITHUB_RUN_ID"],
        "github_run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
    }
    write_json("environment.json", environment)
    (EVIDENCE / "pip-freeze.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
        encoding="utf-8",
    )
    print(json.dumps(environment, indent=2), flush=True)
    # Add only the regression tests to the unmodified production baseline.
    test = "Long-WAM/tests/runtime/test_worker_transport.py"
    shutil.copyfile(CANDIDATE / test, BASELINE / test)
    baseline_env = dict(os.environ, PYTHONPATH=str(BASELINE / "Long-WAM/src"))
    candidate_env = dict(os.environ, PYTHONPATH=str(CANDIDATE / "Long-WAM/src"))
    selectors = [
        "tests/runtime/test_async_policy.py",
        "tests/runtime/test_worker_transport.py::test_sender_failure_completes_future_and_is_cached[serialization]",
        "tests/runtime/test_worker_transport.py::test_sender_failure_completes_future_and_is_cached[broken_pipe]",
        "tests/runtime/test_worker_transport.py::test_full_queue_rejects_request_without_poisoning_single_flight",
    ]
    run(
        "baseline",
        [
            sys.executable,
            "-m",
            "pytest",
            *selectors,
            "-q",
            "--color=no",
            "--timeout=30",
            f"--junitxml={EVIDENCE / 'baseline.xml'}",
        ],
        BASELINE / "Long-WAM",
        baseline_env,
        expected=1,
    )
    baseline_cases = results("baseline.xml")
    expected_failures = {
        "tests.runtime.test_worker_transport::test_sender_failure_completes_future_and_is_cached[serialization]",
        "tests.runtime.test_worker_transport::test_sender_failure_completes_future_and_is_cached[broken_pipe]",
        "tests.runtime.test_worker_transport::test_full_queue_rejects_request_without_poisoning_single_flight",
    }
    assert {
        name for name, status in baseline_cases.items() if status == "failure"
    } == expected_failures
    assert Counter(baseline_cases.values()) == {"passed": 22, "failure": 3}
    run(
        "candidate",
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/runtime/test_async_policy.py",
            test.removeprefix("Long-WAM/"),
            "-q",
            "--color=no",
            "--timeout=30",
            "-W",
            "error",
            f"--junitxml={EVIDENCE / 'candidate.xml'}",
        ],
        CANDIDATE / "Long-WAM",
        candidate_env,
    )
    candidate_cases = results("candidate.xml")
    assert Counter(candidate_cases.values()) == {"passed": 38}
    stress_env = dict(
        candidate_env,
        LONGWAM_WORKER_PATH=str(CANDIDATE / "Long-WAM/src/longwam/runtime/worker.py"),
        LONGWAM_EVIDENCE_DIR=str(EVIDENCE),
    )
    run("stress", [sys.executable, str(ROOT / "stress_transport.py")], ROOT, stress_env)
    stress = json.loads((EVIDENCE / "stress-results.json").read_text())
    assert stress["source_sha256_lf"] == FILES["Long-WAM/src/longwam/runtime/worker.py"]
    assert stress["blocked_receive_sender_failure"] == {"passed": 10, "failed": 0}
    assert stress["multi_poll_timeout_and_retry"] == {"passed": 10, "failed": 0}
    assert (
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=CANDIDATE, text=True
        )
        == ""
    )
    report = {
        "baseline_commit": os.environ["BASELINE_SHA"],
        "candidate_commit": os.environ["CANDIDATE_SHA"],
        "baseline": dict(Counter(baseline_cases.values())),
        "baseline_expected_failure_ids": sorted(expected_failures),
        "candidate": dict(Counter(candidate_cases.values())),
        "stress": stress,
        "source_unchanged": True,
        "scope": "Linux CPU worker transport and scheduling only. Full suite, GPU, released models, simulators and hardware not exercised.",
    }
    write_json("validation-summary.json", report)
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as summary:
        summary.write("## Long-WAM Linux worker transport validation\n\n")
        summary.write(f"Exact candidate: `{os.environ['CANDIDATE_SHA']}`.\n\n")
        summary.write(
            "- Unmodified baseline: 22 scheduling tests pass; 3 selected regression cases fail as expected.\n"
        )
        summary.write(
            "- Candidate: 38 scheduling and transport tests pass, warnings treated as errors.\n"
        )
        summary.write("- Real spawned-process stress: 20 trials pass.\n")
        summary.write(
            "- All three candidate source hashes match the reviewed Windows-tested commit; source remains clean.\n"
        )
        summary.write(
            "- This is contributor-fork CPU validation, not upstream PR CI or GPU/model/hardware validation.\n"
        )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
