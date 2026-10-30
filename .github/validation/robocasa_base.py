# SPDX-FileCopyrightText: Copyright (c) 2026 AuricTW
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original one-off Long-WAM contributor validation.
"""Compare the exact candidate and baseline in fresh Linux environments."""

from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path.cwd()
SOURCE = ROOT / "source"
BASELINE = ROOT / "baseline"
EVIDENCE = ROOT / "evidence"
EVIDENCE.mkdir(exist_ok=True)
MODE = os.environ["VALIDATION_MODE"]


def write_json(name, value):
    (EVIDENCE / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run(name, args, checkout, allowed=(0,)):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(checkout / "Long-WAM/src")
    print(f"Running {name}: {args}", flush=True)
    with (EVIDENCE / f"{name}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            args, cwd=checkout / "Long-WAM", env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    assert code in allowed, (name, code, allowed)


def cases(name):
    result = {}
    for case in ET.parse(EVIDENCE / f"{name}.xml").iter("testcase"):
        identity = case.get("classname", "") + "::" + case.get("name", "")
        assert identity not in result, identity
        result[identity] = next(
            (kind for kind in ("failure", "error", "skipped") if case.find(kind) is not None),
            "passed",
        )
    return result


def pytest_run(name, checkout, paths, allowed=(0,)):
    run(name, [sys.executable, "-m", "pytest", "-c", "pyproject.toml", *paths,
               "-q", "--tb=short", "--timeout=120", "--timeout-method=signal",
               "--junitxml=" + str(EVIDENCE / f"{name}.xml")], checkout, allowed)
    return cases(name)


def main():
    assert sys.platform == "linux"
    assert sys.prefix != sys.base_prefix
    import torch
    assert torch.__version__ == "2.7.1+cpu"
    assert not torch.cuda.is_available()
    optional = ("datasets", "pyarrow", "pandas", "git", "boto3", "wandb", "jsonlines", "lerobot")
    optional_present = {name: importlib.util.find_spec(name) is not None for name in optional}
    if MODE == "base":
        assert not any(optional_present.values()), optional_present
    else:
        assert all(optional_present[name] for name in optional if name != "lerobot")
        assert not optional_present["lerobot"]
    write_json("environment.json", {
        "mode": MODE, "platform": platform.platform(), "os_release": platform.freedesktop_os_release(),
        "python": sys.version, "venv_config": (Path(sys.prefix) / "pyvenv.cfg").read_text(),
        "training_packages_present": optional_present,
        "installed": sorted((dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions()),
    })
    run("pip-check", [sys.executable, "-m", "pip", "check"], SOURCE)
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", os.environ["BASELINE_SHA"], os.environ["CANDIDATE_SHA"]],
        cwd=SOURCE, text=True,
    ).splitlines()
    manifest = {}
    for checkout, key in ((BASELINE, "BASELINE_SHA"), (SOURCE, "CANDIDATE_SHA")):
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
        assert sha == os.environ[key]
    for name in changed:
        manifest[name] = hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
    write_json("source-manifest.json", {
        "baseline": os.environ["BASELINE_SHA"], "candidate": os.environ["CANDIDATE_SHA"],
        "workflow_commit": os.environ["GITHUB_SHA"], "sha256": manifest,
    })

    if MODE == "base":
        regression = EVIDENCE / "regression"
        regression.mkdir()
        test_path = regression / "test_robocasa_base_install.py"
        shutil.copy2(SOURCE / "Long-WAM/tests/test_robocasa_base_install.py", test_path)
        before = pytest_run("baseline-new-tests", BASELINE, [str(test_path)], (1,))
        assert Counter(before.values()) == {"failure": 4}, before
        baseline_log = (EVIDENCE / "baseline-new-tests.log").read_text()
        assert "No module named" in baseline_log and "datasets" in baseline_log and "git" in baseline_log
        after = pytest_run("candidate-base-tests", SOURCE, [
            "tests/test_robocasa_base_install.py", "tests/test_dependency_contract.py",
        ])
        assert sum(value == "passed" for key, value in after.items() if "robocasa_base_install" in key) == 4
        assert not any(value in ("error", "failure") for value in after.values())
        write_json("summary.json", {"baseline_regression_tests": dict(Counter(before.values())),
                                   "candidate_tests": dict(Counter(after.values())),
                                   "actual_training_packages_absent": True})
    else:
        before = pytest_run("baseline-full", BASELINE, ["tests"], (0, 1))
        after = pytest_run("candidate-full", SOURCE, ["tests"], (0, 1))
        assert before.keys() <= after.keys()
        regressions = [key for key, status in before.items() if status == "passed" and after[key] != "passed"]
        failures_before = {key for key, status in before.items() if status in ("failure", "error")}
        failures_after = {key for key, status in after.items() if status in ("failure", "error")}
        assert not regressions, regressions
        assert failures_before == failures_after, (failures_before, failures_after)
        added = set(after) - set(before)
        assert len(added) == 4 and all(after[key] == "passed" for key in added), added
        write_json("summary.json", {"baseline": dict(Counter(before.values())),
                                   "candidate": dict(Counter(after.values())),
                                   "new_test_ids": sorted(added),
                                   "unchanged_failures": sorted(failures_after),
                                   "regressions": regressions})
    run("ruff", [sys.executable, "-m", "ruff", "check", *[
        str(SOURCE / name) for name in changed if name.endswith(".py")
    ]], SOURCE)
    for checkout in (BASELINE, SOURCE):
        assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout, text=True).strip()
    print("VALIDATION PASSED: " + MODE, flush=True)


if __name__ == "__main__":
    main()
