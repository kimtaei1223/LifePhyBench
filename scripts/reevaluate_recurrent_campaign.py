#!/usr/bin/env python3
"""Re-evaluate completed recurrent runs in comparable task-episode units."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from functools import partial
from pathlib import Path
from typing import Any

from train_recurrent_smoke import make_environment, require_recurrent_ppo

from lifephybench.recurrent_evaluation import evaluate_task_episodes, evaluation_as_dict


def evaluate_run(
    run_directory: Path,
    task_episodes: int,
    device: str,
) -> dict[str, Any]:
    metadata = json.loads((run_directory / "metadata.json").read_text(encoding="utf-8"))
    arguments = metadata["arguments"]
    modules = require_recurrent_ppo()
    factory = partial(
        make_environment,
        environment_id=arguments["environment_id"],
        mechanism=arguments["mechanism"],
        degradation_mode=arguments["degradation_mode"],
        memory_mode=arguments["memory_mode"],
        episode_steps=arguments["episode_steps"],
        episodes_per_lifetime=arguments["episodes_per_lifetime"],
        exogenous_dose_per_step=arguments.get("exogenous_dose_per_step", 0.25),
        thermal_exogenous_dose_per_step=arguments.get(
            "thermal_exogenous_dose_per_step", 0.25
        ),
        joint_aging_exogenous_dose_per_step=arguments.get(
            "joint_aging_exogenous_dose_per_step", 0.25
        ),
        monitor=modules["Monitor"],
    )
    environment = factory()
    try:
        model = modules["RecurrentPPO"].load(str(run_directory / "model"), device=device)
        evaluation_seed = int(arguments["seed"]) + 10_000_019
        evaluation = evaluate_task_episodes(
            model, environment, task_episodes, seed=evaluation_seed
        )
    finally:
        environment.close()
    result = {
        "phase": "recurrent_task_episode_reevaluation_not_final_result",
        "run_name": arguments["run_name"],
        "memory_mode": arguments["memory_mode"],
        "seed": arguments["seed"],
        "device": device,
        "evaluation_seed": evaluation_seed,
        "task_episode_evaluation": evaluation_as_dict(evaluation),
    }
    (run_directory / "task_episode_evaluation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/recurrent_campaign")
    parser.add_argument("--task-episodes", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of completed runs to evaluate concurrently.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate runs even when a matching result already exists.",
    )
    args = parser.parse_args()
    if args.jobs <= 0:
        raise SystemExit("jobs must be positive")
    root = Path(args.output_root)
    runs = sorted(path.parent for path in root.glob("*/metadata.json"))
    if not runs:
        raise SystemExit(f"no recurrent metadata files found under {root}")
    results = []
    pending: list[tuple[int, Path]] = []
    for index, run in enumerate(runs, start=1):
        result_path = run / "task_episode_evaluation.json"
        if result_path.exists() and not args.force:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            completed = existing.get("task_episode_evaluation", {}).get(
                "task_episodes"
            )
            if completed == args.task_episodes:
                print(
                    f"[SKIP {index}/{len(runs)}] {run.name} "
                    f"({completed} task episodes)",
                    flush=True,
                )
                results.append(existing)
                continue
        pending.append((index, run))

    if args.jobs == 1:
        for index, run in pending:
            print(f"[START {index}/{len(runs)}] {run.name}", flush=True)
            result = evaluate_run(run, args.task_episodes, args.device)
            results.append(result)
            print(f"[DONE {index}/{len(runs)}] {run.name}", flush=True)
    elif pending:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(args.jobs, len(pending))
        ) as executor:
            futures = {}
            for index, run in pending:
                print(f"[START {index}/{len(runs)}] {run.name}", flush=True)
                future = executor.submit(
                    evaluate_run, run, args.task_episodes, args.device
                )
                futures[future] = (index, run)
            for future in concurrent.futures.as_completed(futures):
                index, run = futures[future]
                results.append(future.result())
                print(f"[DONE {index}/{len(runs)}] {run.name}", flush=True)
    print(json.dumps({"results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
