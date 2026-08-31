#!/usr/bin/env python3
"""Reproduce sealed publication artifacts from a clean repository checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "evidence" / "snapshots" / "2026-08-31_pusher_reacher_final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="also run the complete pytest suite",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="skip loading the two Stable-Baselines model archives",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="optional JSON report destination",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.relative_to(ROOT)}")
    return value


def verify_checksum_file(root: Path, checksum_file: Path) -> int:
    checked = 0
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"manifest entry is missing: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"checksum mismatch: {relative}")
        checked += 1
    return checked


def verify_protocol_hashes() -> int:
    checked = 0
    names = ("FROZEN_PROTOCOL.json", "FROZEN_FRESH_PROTOCOL.json")
    for name in names:
        for protocol in SNAPSHOT.rglob(name):
            expected = protocol.with_suffix(".sha256").read_text().strip()
            if sha256(protocol) != expected:
                raise ValueError(
                    f"protocol checksum mismatch: {protocol.relative_to(ROOT)}"
                )
            checked += 1
    return checked


def run_checked(command: list[str]) -> None:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        details = (result.stdout + "\n" + result.stderr).strip()
        raise RuntimeError(f"command failed: {' '.join(command)}\n{details}")


def compare_generated_artifacts(
    generated: Path, expected: Path
) -> dict[str, str]:
    expected_manifest = read_json(expected / "MANIFEST.json")
    generated_manifest = read_json(generated / "MANIFEST.json")
    expected_hashes = expected_manifest.get("artifacts")
    generated_hashes = generated_manifest.get("artifacts")
    if not isinstance(expected_hashes, dict) or not isinstance(generated_hashes, dict):
        raise ValueError("publication manifest has no artifact hash mapping")
    if generated_hashes != expected_hashes:
        differing = sorted(set(expected_hashes) | set(generated_hashes))
        differing = [
            name
            for name in differing
            if expected_hashes.get(name) != generated_hashes.get(name)
        ]
        raise ValueError(f"regenerated artifact mismatch: {', '.join(differing)}")
    return {str(name): str(value) for name, value in generated_hashes.items()}


def reproduce_artifacts(temporary_root: Path) -> dict[str, int]:
    physics_output = temporary_root / "physics_residual_v12_3"
    reacher_output = temporary_root / "reacher_replication"
    run_checked(
        [
            sys.executable,
            "scripts/render_physics_residual_v12_3_artifacts.py",
            "--input-root",
            str(
                SNAPSHOT
                / "artifacts"
                / "outputs"
                / "physics_residual_v12_3_factorial_ablation"
            ),
            "--output-root",
            str(physics_output),
        ]
    )
    run_checked(
        [
            sys.executable,
            "scripts/render_reacher_replication_artifacts.py",
            "--input-root",
            str(SNAPSHOT / "artifacts" / "outputs" / "reacher_replication"),
            "--output-root",
            str(reacher_output),
        ]
    )
    physics_hashes = compare_generated_artifacts(
        physics_output, ROOT / "paper_artifacts" / "physics_residual_v12_3"
    )
    reacher_hashes = compare_generated_artifacts(
        reacher_output, ROOT / "paper_artifacts" / "reacher_replication"
    )
    return {"physics_artifacts": len(physics_hashes), "reacher_artifacts": len(reacher_hashes)}


def verify_models() -> int:
    from sb3_contrib import RecurrentPPO

    archives = (
        SNAPSHOT
        / "artifacts"
        / "outputs"
        / "reacher_replication"
        / "low_level"
        / "reacher-static-task-seed5100-steps2000k"
        / "model.zip",
        SNAPSHOT
        / "artifacts"
        / "outputs"
        / "reacher_replication"
        / "monolithic_baseline"
        / "reacher-monolithic-lifetime-seed25100-decisions100k"
        / "model.zip",
    )
    for archive in archives:
        RecurrentPPO.load(archive, device="cpu")
    return len(archives)


def main() -> None:
    args = parse_args()
    run_checked(
        [sys.executable, "scripts/audit_repository_privacy.py", "--commit", "HEAD"]
    )
    root_entries = verify_checksum_file(SNAPSHOT, SNAPSHOT / "SNAPSHOT_ROOT.sha256")
    artifact_entries = verify_checksum_file(
        SNAPSHOT, SNAPSHOT / "manifests" / "ARTIFACTS.sha256"
    )
    protocol_entries = verify_protocol_hashes()
    with tempfile.TemporaryDirectory(prefix="lifephybench-repro-") as directory:
        rendered = reproduce_artifacts(Path(directory))
    model_entries = 0 if args.skip_models else verify_models()
    if args.run_tests:
        run_checked([sys.executable, "-m", "pytest", "-q"])

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    report = {
        "status": "passed",
        "commit": commit,
        "snapshot": SNAPSHOT.name,
        "snapshot_root_entries": root_entries,
        "artifact_entries": artifact_entries,
        "protocol_entries": protocol_entries,
        "model_archives_loaded": model_entries,
        "tests_run": bool(args.run_tests),
        **rendered,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
