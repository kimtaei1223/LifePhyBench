#!/usr/bin/env python3
"""Run the Phase-0 diagnostic benchmark and emit machine-readable JSON."""

import argparse
import json

from lifephybench.evaluation import evaluate_many
from lifephybench.policies import (
    EpisodeEMAController,
    LifetimeEMAController,
    MyopicStateOracleController,
    NominalReactiveController,
)
from lifephybench.toy_env import ToyWearConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--wear-rate", type=float, default=0.012)
    parser.add_argument("--wear-exponent", type=float, default=2.0)
    parser.add_argument("--process-noise-std", type=float, default=0.005)
    parser.add_argument("--shock-probability", type=float, default=0.01)
    parser.add_argument("--shock-size", type=float, default=0.01)
    parser.add_argument(
        "--include-lifetimes",
        action="store_true",
        help="include each seed's row instead of aggregate statistics only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.seeds <= 0:
        raise SystemExit("--episodes and --seeds must be positive")

    config = ToyWearConfig(
        horizon=args.horizon,
        wear_rate=args.wear_rate,
        wear_exponent=args.wear_exponent,
        process_noise_std=args.process_noise_std,
        stochastic_shock_probability=args.shock_probability,
        stochastic_shock_size=args.shock_size,
    )
    factories = [
        NominalReactiveController,
        EpisodeEMAController,
        LifetimeEMAController,
        MyopicStateOracleController,
    ]
    results = {
        "phase": "phase0_diagnostic_not_for_paper",
        "results": [
            evaluate_many(
                factory,
                config,
                range(args.seeds),
                args.episodes,
                persistent_physics=persistent,
                include_lifetimes=args.include_lifetimes,
            )
            for persistent in (True, False)
            for factory in factories
        ],
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
