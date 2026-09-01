#!/usr/bin/env python3
"""Build and audit the anonymous TMLR PDF and reproducibility supplement."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from audit_repository_privacy import scan_payload


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "LifePhyBench-anonymous"
MAX_SUPPLEMENT_BYTES = 100 * 1024 * 1024
FIXED_ZIP_TIME = (2026, 9, 1, 0, 0, 0)

EXACT_FILES = {
    "README.md",
    "environment.yml",
    "pyproject.toml",
    "requirements-reproduction.txt",
}
PREFIXES = (
    "configs/",
    "manuscript/tmlr/",
    "src/",
    "tests/",
    "scripts/",
    "paper_artifacts/physics_residual_v12_3/",
    "paper_artifacts/reacher_replication/",
    "evidence/snapshots/2026-08-31_pusher_reacher_final/",
)
SUPPORTING_DOCS = {
    "docs/CLEAN_CHECKOUT_REPRODUCTION_2026-08-31.md",
    "docs/INTEGRATED_PUSHER_REACHER_AUDIT.md",
    "docs/PHYSICS_RESIDUAL_V12_FINAL_RESULTS.md",
    "docs/REACHER_REPLICATION_FINAL_RESULTS.md",
    "docs/TMLR_CLAIM_EVIDENCE_AUDIT_2026-09-01.md",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_checked(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        details = (result.stdout + "\n" + result.stderr).strip()
        raise RuntimeError(f"validation command failed: {' '.join(command)}\n{details}")
    return result.stdout.strip()


def tracked_files() -> list[Path]:
    output = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout.decode()
    return [Path(name) for name in output.split("\0") if name]


def supplement_files() -> list[Path]:
    selected = []
    for path in tracked_files():
        name = path.as_posix()
        if (
            name in EXACT_FILES
            or name in SUPPORTING_DOCS
            or any(name.startswith(prefix) for prefix in PREFIXES)
        ):
            selected.append(path)
    if not selected:
        raise RuntimeError("supplement file selection is empty")
    return sorted(selected)


def zip_entry(name: str, payload: bytes) -> tuple[zipfile.ZipInfo, bytes]:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info, payload


def audit_pdf(path: Path) -> None:
    findings = scan_payload(path.name, path.read_bytes())
    for executable, arguments in (
        ("pdfinfo", [str(path)]),
        ("pdftotext", [str(path), "-"]),
    ):
        if shutil.which(executable):
            result = subprocess.run(
                [executable, *arguments], check=True, capture_output=True
            )
            findings.extend(scan_payload(f"{path.name}:{executable}", result.stdout))
    if findings:
        raise RuntimeError(f"PDF privacy audit failed: {sorted(set(findings))}")


def build_supplement(destination: Path) -> dict[str, object]:
    files = supplement_files()
    entries: dict[str, str] = {}
    for relative in files:
        payload = (ROOT / relative).read_bytes()
        findings = scan_payload(relative.as_posix(), payload)
        if findings:
            raise RuntimeError(f"supplement source privacy audit failed: {findings}")
        entries[relative.as_posix()] = sha256_bytes(payload)

    readme = """# Anonymous TMLR supplementary material

This archive contains the code, immutable evidence snapshot, final derived
tables and figures, dependency lock, and tests supporting the submission.

From the archive root, create a Python 3.11 environment and run:

```bash
python scripts/reproduce_clean_checkout.py --run-tests --report reproduction.json
python scripts/audit_tmlr_claims.py --output claim_audit.md
```

The first command checks sealed hashes, regenerates publication artifacts,
loads retained models on CPU, and runs the test suite. The second traces the
principal manuscript claims to the immutable tables and lifetime records.
No GPU training is required to reproduce the reported analyses.
"""
    manifest = {
        "archive_root": ARCHIVE_ROOT,
        "description": "Anonymous code and evidence supplement for TMLR review",
        "file_count": len(entries),
        "files": entries,
        "reproduction_python": "3.11",
    }
    generated = {
        "README_SUPPLEMENT.md": readme.encode(),
        "SUPPLEMENT_MANIFEST.json": (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode(),
    }

    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, payload in generated.items():
            info, data = zip_entry(f"{ARCHIVE_ROOT}/{name}", payload)
            archive.writestr(info, data)
        for relative in files:
            payload = (ROOT / relative).read_bytes()
            info, data = zip_entry(
                f"{ARCHIVE_ROOT}/{relative.as_posix()}", payload
            )
            archive.writestr(info, data)

    if destination.stat().st_size > MAX_SUPPLEMENT_BYTES:
        raise RuntimeError("supplement exceeds the TMLR 100 MB limit")
    findings: list[tuple[str, str]] = []
    with zipfile.ZipFile(destination, "r") as archive:
        for entry in archive.infolist():
            findings.extend(scan_payload(entry.filename, archive.read(entry.filename)))
    if findings:
        raise RuntimeError(f"supplement privacy audit failed: {findings}")
    return {
        "bytes": destination.stat().st_size,
        "file_count": len(entries) + len(generated),
        "sha256": sha256_file(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submission")
    args = parser.parse_args()

    pdf = args.pdf.resolve()
    if not pdf.is_file():
        raise FileNotFoundError(pdf)

    run_checked([sys.executable, "scripts/validate_tmlr_manuscript.py"])
    run_checked([sys.executable, "scripts/audit_tmlr_claims.py"])
    run_checked([sys.executable, "scripts/audit_repository_privacy.py"])
    audit_pdf(pdf)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    review_pdf = output / "TMLR_REVIEW_MANUSCRIPT.pdf"
    supplement = output / "TMLR_SUPPLEMENT_ANONYMOUS.zip"
    shutil.copyfile(pdf, review_pdf)
    supplement_result = build_supplement(supplement)
    audit_pdf(review_pdf)

    report = {
        "status": "passed",
        "review_pdf": {
            "filename": review_pdf.name,
            "bytes": review_pdf.stat().st_size,
            "sha256": sha256_file(review_pdf),
        },
        "supplement": {
            "filename": supplement.name,
            **supplement_result,
            "limit_bytes": MAX_SUPPLEMENT_BYTES,
        },
    }
    report_path = output / "SUBMISSION_MANIFEST.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
