"""CPU-only semantic audits for the diagnostic lifetime environment.

These checks are intentionally simple and transparent.  They determine whether
the environment's hidden health can be predicted by elapsed time alone, or
whether a predictor needs the accumulated action dose that caused degradation.
They are a gate for benchmark semantics, not a learned-policy result.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .toy_env import ToyWearConfig, ToyWearEnv


@dataclass(frozen=True)
class ShortcutAuditResult:
    degradation_mode: str
    time_only_rmse: float
    action_history_rmse: float
    action_history_improvement: float
    passed: bool


def _fit_line(rows: list[tuple[float, float]]) -> tuple[float, float]:
    """Fit y = intercept + slope * x without a numerical dependency."""

    if len(rows) < 2:
        raise ValueError("at least two rows are required")
    mean_x = sum(x for x, _ in rows) / len(rows)
    mean_y = sum(y for _, y in rows) / len(rows)
    variance = sum((x - mean_x) ** 2 for x, _ in rows)
    if variance == 0.0:
        return mean_y, 0.0
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in rows)
    return mean_y - covariance * mean_x / variance, covariance / variance


def _rmse(rows: list[tuple[float, float]], line: tuple[float, float]) -> float:
    intercept, slope = line
    return (sum((intercept + slope * x - y) ** 2 for x, y in rows) / len(rows)) ** 0.5


def _collect_rows(
    config: ToyWearConfig,
    seeds: range,
    steps_per_lifetime: int,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Collect `(feature, wear)` rows for time and action-dose predictors."""

    time_rows: list[tuple[float, float]] = []
    action_rows: list[tuple[float, float]] = []
    for seed in seeds:
        env = ToyWearEnv(config)
        env.reset_lifetime(seed=seed, lifetime_id=seed)
        rng = random.Random(10_000 + seed)
        cumulative_dose = 0.0
        for step in range(steps_per_lifetime):
            if step > 0 and step % config.horizon == 0:
                env.reset_episode()
            action = rng.uniform(0.05, config.action_limit)
            cumulative_dose += max(
                0.0, abs(action) - config.overload_threshold
            ) ** config.wear_exponent
            result = env.step(action)
            time_rows.append((float(step + 1), result.info["wear"]))
            action_rows.append((cumulative_dose, result.info["wear"]))
    return time_rows, action_rows


def run_clock_shortcut_audit(
    config: ToyWearConfig,
    train_seeds: range = range(10, 30),
    test_seeds: range = range(30, 50),
    steps_per_lifetime: int = 100,
) -> ShortcutAuditResult:
    """Assess whether action history carries predictive information beyond time.

    In endogenous mode, a valid diagnostic should require action history and
    show a substantial held-out RMSE reduction.  In exogenous-clock mode, time
    should be sufficient, so an action-history shortcut must not win.
    """

    train_time, train_action = _collect_rows(config, train_seeds, steps_per_lifetime)
    test_time, test_action = _collect_rows(config, test_seeds, steps_per_lifetime)
    time_rmse = _rmse(test_time, _fit_line(train_time))
    action_rmse = _rmse(test_action, _fit_line(train_action))
    improvement = time_rmse - action_rmse
    if config.degradation_mode == "endogenous_action":
        passed = action_rmse < time_rmse * 0.2
    else:
        passed = action_rmse >= time_rmse * 0.95
    return ShortcutAuditResult(
        degradation_mode=config.degradation_mode,
        time_only_rmse=time_rmse,
        action_history_rmse=action_rmse,
        action_history_improvement=improvement,
        passed=passed,
    )
