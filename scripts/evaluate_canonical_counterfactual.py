#!/usr/bin/env python3
"""Cross-evaluate canonical thermal policies in dynamic and static physics."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from train_fair_recurrent import make_fair_environment
from train_recurrent_smoke import require_recurrent_ppo

from lifephybench.recurrent_evaluation import (
    evaluate_task_episodes,
    evaluation_as_dict,
)


EVALUATION_CELLS = {
    "dynamic": "endogenous_action",
    "static": "exogenous_clock",
}


def training_label(run_name: str) -> str:
    if "-dynamic-" in run_name:
        return "dynamic"
    if "-static-" in run_name:
        return "static"
    raise ValueError(f"cannot infer training label from {run_name!r}")


def result_matches(path: Path, task_episodes: int, device: str) -> bool:
    if not path.exists():
        return False
    result = json.loads(path.read_text(encoding="utf-8"))
    return (
        result.get("task_episodes_requested") == task_episodes
        and result.get("device") == device
    )


def make_evaluation_environment(arguments: dict[str, Any], label: str, monitor: Any):
    return make_fair_environment(
        environment_id=arguments["environment_id"],
        mechanism="thermal",
        degradation_mode=EVALUATION_CELLS[label],
        episode_steps=int(arguments["episode_steps"]),
        episodes_per_lifetime=int(arguments["episodes_per_lifetime"]),
        exogenous_dose_per_step=float(arguments.get("exogenous_dose_per_step", 0.25)),
        thermal_exogenous_dose_per_step=0.0,
        joint_aging_exogenous_dose_per_step=float(
            arguments.get("joint_aging_exogenous_dose_per_step", 0.25)
        ),
        thermal_heat_rate=float(arguments["thermal_heat_rate"]),
        thermal_cooling_rate=float(arguments["thermal_cooling_rate"]),
        thermal_episode_cooling=float(arguments["thermal_episode_cooling"]),
        canonical_task_seed=int(arguments["canonical_task_seed"]),
        monitor=monitor,
    )


def evaluate_run(
    run_directory: Path,
    output_root: Path,
    task_episodes: int,
    device: str,
    modules: dict[str, Any],
) -> list[dict[str, Any]]:
    metadata = json.loads((run_directory / "metadata.json").read_text(encoding="utf-8"))
    arguments = metadata["arguments"]
    run_name = str(arguments["run_name"])
    train_label = training_label(run_name)
    model = modules["RecurrentPPO"].load(str(run_directory / "model"), device=device)
    evaluation_seed = int(metadata["evaluation_seed"])
    results = []
    for evaluation_label, degradation_mode in EVALUATION_CELLS.items():
        result_path = output_root / f"{run_name}--eval-{evaluation_label}.json"
        if result_matches(result_path, task_episodes, device):
            result = json.loads(result_path.read_text(encoding="utf-8"))
            print(f"[SKIP] {run_name} -> {evaluation_label}", flush=True)
            results.append(result)
            continue
        environment = make_evaluation_environment(
            arguments, evaluation_label, modules["Monitor"]
        )
        try:
            evaluation = evaluate_task_episodes(
                model, environment, task_episodes, seed=evaluation_seed
            )
        finally:
            environment.close()
        evaluation_document = evaluation_as_dict(evaluation)
        native_reward = float(metadata["task_episode_evaluation"]["mean_task_episode_reward"])
        is_native = train_label == evaluation_label
        result = {
            "phase": "canonical_thermal_policy_counterfactual_evaluation",
            "source_run": str(run_directory),
            "run_name": run_name,
            "training_label": train_label,
            "evaluation_label": evaluation_label,
            "memory_mode": str(arguments["memory_mode"]),
            "seed": int(arguments["seed"]),
            "device": device,
            "task_episodes_requested": task_episodes,
            "evaluation_seed": evaluation_seed,
            "is_native_evaluation": is_native,
            "native_metadata_reward": native_reward if is_native else None,
            "native_replay_absolute_error": (
                abs(float(evaluation_document["mean_task_episode_reward"]) - native_reward)
                if is_native and task_episodes == int(arguments["eval_task_episodes"])
                else None
            ),
            "evaluation_environment": {
                "degradation_mode": degradation_mode,
                "canonical_task_seed": int(arguments["canonical_task_seed"]),
                "thermal_heat_rate": float(arguments["thermal_heat_rate"]),
                "thermal_cooling_rate": float(arguments["thermal_cooling_rate"]),
                "thermal_episode_cooling": float(arguments["thermal_episode_cooling"]),
                "thermal_exogenous_dose_per_step": 0.0,
            },
            "task_episode_evaluation": evaluation_document,
        }
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"[DONE] {run_name} -> {evaluation_label}", flush=True)
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root", type=Path, default=Path("outputs/canonical_thermal_probe")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/canonical_thermal_counterfactual"),
    )
    parser.add_argument("--task-episodes", type=int, default=1_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()
    if args.task_episodes <= 0:
        raise SystemExit("task-episodes must be positive")

    args.output_root.mkdir(parents=True, exist_ok=True)
    runs = sorted(path.parent for path in args.input_root.glob("*/metadata.json"))
    if args.seeds is not None:
        selected = set(args.seeds)
        runs = [
            run
            for run in runs
            if json.loads((run / "metadata.json").read_text(encoding="utf-8"))[
                "arguments"
            ]["seed"]
            in selected
        ]
    if not runs:
        raise SystemExit(f"no matching trained policies under {args.input_root}")

    manifest = {
        "phase": "canonical_thermal_policy_counterfactual_protocol",
        "input_root": str(args.input_root),
        "task_episodes": args.task_episodes,
        "device": args.device,
        "seeds": sorted(
            {
                int(
                    json.loads((run / "metadata.json").read_text(encoding="utf-8"))[
                        "arguments"
                    ]["seed"]
                )
                for run in runs
            }
        ),
        "evaluation_cells": EVALUATION_CELLS,
        "estimand": "memory effect by training condition crossed with evaluation physics",
    }
    manifest_path = args.output_root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise SystemExit(f"manifest mismatch: {manifest_path}; use a new output root")
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    modules = require_recurrent_ppo()
    results = []
    for index, run in enumerate(runs, start=1):
        print(f"[POLICY {index}/{len(runs)}] {run.name}", flush=True)
        results.extend(
            evaluate_run(
                run,
                args.output_root,
                args.task_episodes,
                args.device,
                modules,
            )
        )
    print(f"[COUNTERFACTUAL COMPLETE] evaluations={len(results)}", flush=True)


if __name__ == "__main__":
    main()
