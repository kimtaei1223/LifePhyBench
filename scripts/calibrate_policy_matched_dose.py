#!/usr/bin/env python3
"""Measure endogenous policy dose for a matched exogenous control.

The exogenous control should use the action-dose distribution of a fixed
reference policy rather than an arbitrary constant.  This script evaluates
completed endogenous recurrent policies and writes one reproducible target per
policy/memory condition.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from train_recurrent_smoke import make_environment, require_recurrent_ppo


def measure_run(run_directory: Path, task_episodes: int, device: str) -> dict[str, Any]:
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
    base_environment = environment.env.env
    model = modules["RecurrentPPO"].load(str(run_directory / "model"), device=device)
    evaluation_seed = int(arguments["seed"]) + 20_000_019
    observation, _info = environment.reset(seed=evaluation_seed)
    recurrent_state = None
    episode_start = np.array([True])
    task_count = 0
    action_doses: list[float] = []
    try:
        while task_count < task_episodes:
            action, recurrent_state = model.predict(
                observation,
                state=recurrent_state,
                episode_start=episode_start,
                deterministic=True,
            )
            action_doses.append(base_environment._normalized_action_dose(action, 2.0))
            observation, _reward, terminated, truncated, info = environment.step(action)
            task_boundary = bool(
                info.get("lifephy/inner_task_boundary", terminated or truncated)
            )
            if task_boundary:
                task_count += 1
            gym_boundary = bool(terminated or truncated)
            if gym_boundary and task_count < task_episodes:
                observation, _info = environment.reset()
            episode_start = np.array([gym_boundary])
    finally:
        environment.close()

    return {
        "run_name": arguments["run_name"],
        "seed": arguments["seed"],
        "memory_mode": arguments["memory_mode"],
        "degradation_mode": arguments["degradation_mode"],
        "task_episodes": task_count,
        "transitions": len(action_doses),
        "mean_action_dose": statistics.mean(action_doses),
        "std_action_dose": statistics.stdev(action_doses)
        if len(action_doses) > 1
        else 0.0,
        "recommended_exogenous_dose_per_step": statistics.mean(action_doses),
        "evaluation_seed": evaluation_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root", default="outputs/thermal_campaign"
    )
    parser.add_argument(
        "--pattern", default="thermal-endogenous_action-*-seed*-steps1000k"
    )
    parser.add_argument("--task-episodes", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output", default="outputs/thermal_policy_matched_dose.json"
    )
    args = parser.parse_args()
    runs = sorted(
        Path(path).parent
        for path in glob.glob(str(Path(args.input_root) / args.pattern / "metadata.json"))
    )
    if not runs:
        raise SystemExit(f"no matching runs under {args.input_root}")
    results = []
    for index, run in enumerate(runs, start=1):
        print(f"[START {index}/{len(runs)}] {run.name}", flush=True)
        result = measure_run(run, args.task_episodes, args.device)
        results.append(result)
        print(
            f"[DONE {index}/{len(runs)}] {run.name} "
            f"dose={result['recommended_exogenous_dose_per_step']:.9f}",
            flush=True,
        )
    grouped: dict[str, list[float]] = {}
    for result in results:
        grouped.setdefault(result["memory_mode"], []).append(
            result["recommended_exogenous_dose_per_step"]
        )
    summary = {
        memory: {
            "n": len(values),
            "mean_reference_dose": statistics.mean(values),
            "sd_reference_dose": statistics.stdev(values)
            if len(values) > 1
            else 0.0,
        }
        for memory, values in grouped.items()
    }
    output = {
        "phase": "policy_matched_dose_calibration",
        "input_root": args.input_root,
        "task_episodes": args.task_episodes,
        "results": results,
        "summary": summary,
    }
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
