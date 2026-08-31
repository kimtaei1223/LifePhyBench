#!/usr/bin/env python3
"""Run the CPU-only action-history versus clock shortcut audit."""

from __future__ import annotations

import json

from lifephybench.audits import run_clock_shortcut_audit
from lifephybench.toy_env import ToyWearConfig


def main() -> None:
    results = []
    for mode in ("endogenous_action", "exogenous_clock"):
        result = run_clock_shortcut_audit(
            ToyWearConfig(
                horizon=20,
                target=100.0,
                wear_rate=0.002,
                degradation_mode=mode,
                exogenous_dose_per_step=0.5,
            )
        )
        results.append(result.__dict__)
    print(json.dumps({"audit": "clock_shortcut", "results": results}, indent=2))
    if not all(row["passed"] for row in results):
        raise SystemExit("clock-shortcut audit failed")


if __name__ == "__main__":
    main()
