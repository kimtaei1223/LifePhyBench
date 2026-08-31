#!/usr/bin/env python3
"""Run a paired bootstrap over per-lifetime JSONL records."""

from __future__ import annotations

import argparse
import json

from lifephybench.lifetime_analysis import (
    paired_bootstrap,
    read_jsonl,
    result_as_dict,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", help="JSONL file containing per-lifetime records")
    parser.add_argument("--metric", required=True)
    parser.add_argument("--condition-key", default="condition")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--pairing-keys", nargs="+", default=["seed"])
    parser.add_argument(
        "--where",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="optional exact-match filters applied before pairing",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--rng-seed", type=int, default=0)
    args = parser.parse_args()
    filters = {}
    for item in args.where:
        if "=" not in item:
            raise SystemExit(f"invalid filter {item!r}; expected KEY=VALUE")
        key, value = item.split("=", maxsplit=1)
        if not key:
            raise SystemExit(f"invalid empty filter key in {item!r}")
        filters[key] = value
    records = [
        row
        for row in read_jsonl(args.records)
        if all(str(row.get(key)) == value for key, value in filters.items())
    ]
    result = paired_bootstrap(
        records,
        metric=args.metric,
        condition_key=args.condition_key,
        reference=args.reference,
        treatment=args.treatment,
        pairing_keys=args.pairing_keys,
        bootstrap_draws=args.bootstrap_draws,
        rng_seed=args.rng_seed,
    )
    output = result_as_dict(result)
    output["filters"] = filters
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
