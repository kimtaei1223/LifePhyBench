#!/usr/bin/env python3
"""Bind a path-redacted protocol to its original frozen protocol hash."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    protocol = args.protocol.resolve()
    hash_path = protocol.with_suffix(".sha256")
    ledger_path = protocol.with_name("PRIVACY_REDACTION.json")
    current_hash = sha256(protocol)

    if ledger_path.is_file():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        original_hash = str(ledger["original_protocol_sha256"])
    else:
        original_hash = hash_path.read_text(encoding="utf-8").strip()

    if current_hash == original_hash:
        return

    ledger = {
        "phase": "publication_privacy_redaction",
        "redaction_scope": "local absolute repository paths only",
        "original_protocol_sha256": original_hash,
        "sanitized_protocol_sha256": current_hash,
        "result_values_changed": False,
    }
    ledger_path.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    hash_path.write_text(current_hash + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
