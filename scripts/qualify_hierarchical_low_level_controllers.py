"""Qualify a fixed low-level controller before hierarchical GPU training.

The rule is deliberately threshold based: evaluate every pre-existing static/task
controller and select the lowest seed among eligible controllers.  Rewards are not
used to rank eligible candidates, which avoids choosing the most favorable seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from run_hierarchical_thermal_gate import one_task


RUN_PATTERN = re.compile(
    r"canonical-thermal-static-task-seed(?P<seed>\d+)-steps2000k$"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_lowest_eligible(rows: list[dict]) -> dict | None:
    eligible = [row for row in rows if row["eligible"]]
    return min(eligible, key=lambda row: row["seed"]) if eligible else None


def qualify(run_directory: Path) -> dict:
    match = RUN_PATTERN.fullmatch(run_directory.name)
    if match is None:
        raise ValueError(f"unexpected candidate name: {run_directory.name}")
    seed = int(match.group("seed"))
    model = (run_directory / "model.zip").resolve()
    metadata_path = run_directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    arguments = metadata["arguments"]

    cold_low = one_task(model, 0.0, high=False)
    cold_high = one_task(model, 0.0, high=True)
    static_low = one_task(model, 0.0, high=False, static=True)
    static_high = one_task(model, 0.0, high=True, static=True)
    criteria = {
        "preexisting_canonical_static_task_model": (
            metadata.get("phase") == "fair_selective_memory_confirmatory"
            and arguments.get("degradation_mode") == "exogenous_clock"
            and arguments.get("memory_mode") == "task"
            and arguments.get("mechanism") == "thermal"
            and arguments.get("total_timesteps") == 2_000_000
            and arguments.get("eval_task_episodes") == 1_000
            and arguments.get("seed") == seed
        ),
        "cold_high_completes_full_task_without_trip": (
            not cold_high["tripped"] and cold_high["physical_steps"] == 100
        ),
        "cold_high_beats_cold_low": cold_high["reward"] > cold_low["reward"],
        "static_high_beats_static_low": static_high["reward"] > static_low["reward"],
        "low_mode_completes_full_task": (
            cold_low["physical_steps"] == 100
            and static_low["physical_steps"] == 100
        ),
    }
    return {
        "seed": seed,
        "run_directory": str(run_directory.resolve()),
        "model": str(model),
        "model_sha256": digest(model),
        "counterfactuals": {
            "cold_low": cold_low,
            "cold_high": cold_high,
            "static_low": static_low,
            "static_high": static_high,
        },
        "criteria": criteria,
        "eligible": all(criteria.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/canonical_thermal_probe"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/hierarchical_thermal_controller_qualification.json"),
    )
    args = parser.parse_args()

    candidates = sorted(
        (
            path.parent
            for path in args.input_root.glob(
                "canonical-thermal-static-task-seed*-steps2000k/metadata.json"
            )
        ),
        key=lambda path: int(RUN_PATTERN.fullmatch(path.name).group("seed")),
    )
    if len(candidates) != 10:
        raise SystemExit(f"expected 10 frozen candidates, found {len(candidates)}")

    rows = [qualify(candidate) for candidate in candidates]
    selected = select_lowest_eligible(rows)
    report = {
        "phase": "hierarchical_low_level_controller_qualification",
        "status": "design_calibration_not_learned_policy_evidence",
        "candidate_pool": "preexisting canonical static/task seeds 4000-4009",
        "selection_rule": (
            "select the lowest seed satisfying every fixed eligibility criterion; "
            "do not rank eligible candidates by reward"
        ),
        "candidates": rows,
        "eligible_seeds": [row["seed"] for row in rows if row["eligible"]],
        "selected": (
            None
            if selected is None
            else {
                "seed": selected["seed"],
                "model": selected["model"],
                "model_sha256": selected["model_sha256"],
            }
        ),
        "passed": selected is not None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if selected is None:
        raise SystemExit("no eligible low-level controller; do not launch GPU pilot")


if __name__ == "__main__":
    main()
