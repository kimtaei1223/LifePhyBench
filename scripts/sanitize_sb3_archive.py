#!/usr/bin/env python3
"""Remove a local project-root path embedded in a Stable-Baselines ZIP."""

from __future__ import annotations

import argparse
import os
import tempfile
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--project-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = args.archive.resolve()
    private_root = args.project_root.encode()
    replacement = b"${PROJECT_ROOT}"

    if not zipfile.is_zipfile(archive):
        raise SystemExit(f"not a ZIP archive: {archive}")

    with zipfile.ZipFile(archive, "r") as source:
        members = [(entry, source.read(entry.filename)) for entry in source.infolist()]

    if not any(private_root in payload for _, payload in members):
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{archive.name}.", suffix=".tmp", dir=archive.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w") as destination:
            for entry, payload in members:
                destination.writestr(entry, payload.replace(private_root, replacement))
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
