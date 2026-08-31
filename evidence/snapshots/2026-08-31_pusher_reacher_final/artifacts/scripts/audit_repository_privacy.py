#!/usr/bin/env python3
"""Fail when tracked files or the current commit expose local identifiers."""

from __future__ import annotations

import argparse
import re
import subprocess
import zipfile
from pathlib import Path


FORBIDDEN = (
    (
        "home-directory path",
        re.compile(rb"(?:" + b"/" + rb"home/|" + b"/" + rb"Users/)[^/\s]+/"),
    ),
    ("Windows user path", re.compile(rb"[A-Za-z]:\\Users\\[^\\\s]+\\")),
    ("GPU UUID", re.compile(rb"GPU-[0-9a-fA-F-]{20,}")),
    ("MAC address", re.compile(rb"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}(?![0-9A-Fa-f])")),
)
EMAIL = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ALLOWED_EMAIL_SUFFIX = b"@users.noreply.github.com"
ALLOWED_EMAILS = {b"git@github.com"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", default="HEAD")
    return parser.parse_args()


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], check=True, capture_output=True
    )
    return [Path(item) for item in result.stdout.decode().split("\0") if item]


def scan_payload(label: str, payload: bytes) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    for category, pattern in FORBIDDEN:
        if pattern.search(payload):
            findings.append((label, category))
    try:
        payload.decode("utf-8")
        text_payload = True
    except UnicodeDecodeError:
        text_payload = False
    if text_payload:
        for address in EMAIL.findall(payload):
            normalized = address.lower()
            if normalized not in ALLOWED_EMAILS and not normalized.endswith(
                ALLOWED_EMAIL_SUFFIX
            ):
                findings.append((label, "non-private email address"))
                break
    return findings


def scan_file(path: Path) -> list[tuple[str, str]]:
    findings = scan_payload(str(path), str(path).encode())
    if not path.is_file():
        return findings
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, "r") as archive:
            for entry in archive.infolist():
                findings.extend(
                    scan_payload(f"{path}!{entry.filename}", archive.read(entry.filename))
                )
        return findings
    findings.extend(scan_payload(str(path), path.read_bytes()))
    return findings


def main() -> None:
    args = parse_args()
    findings: list[tuple[str, str]] = []
    for path in tracked_files():
        findings.extend(scan_file(path))

    identity = subprocess.run(
        ["git", "show", "-s", "--format=%ae%n%ce", args.commit],
        check=True,
        capture_output=True,
    ).stdout
    findings.extend(scan_payload(f"commit {args.commit}", identity))

    if findings:
        for label, category in sorted(set(findings)):
            print(f"PRIVACY CHECK FAILED: {category}: {label}")
        raise SystemExit(1)
    print("privacy check passed")


if __name__ == "__main__":
    main()
