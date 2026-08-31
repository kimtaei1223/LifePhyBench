"""Leakage-resistant lifetime logging and paired uncertainty estimates."""

from __future__ import annotations

import json
import random
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Paired comparison over independent lifetimes, never individual steps."""

    metric: str
    reference: str
    treatment: str
    paired_lifetimes: int
    reference_mean: float
    treatment_mean: float
    mean_difference: float
    ci_lower: float
    ci_upper: float
    bootstrap_draws: int
    rng_seed: int


def write_jsonl(records: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    """Write one independently evaluable lifetime record per JSONL line."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True))
            handle.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL lifetime records and reject blank or malformed input."""

    source = Path(path)
    records: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"line {line_number} is not a JSON object")
            records.append(value)
    if not records:
        raise ValueError("expected at least one lifetime record")
    return records


def paired_bootstrap(
    records: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    condition_key: str,
    reference: str,
    treatment: str,
    pairing_keys: Sequence[str] = ("seed",),
    bootstrap_draws: int = 10_000,
    rng_seed: int = 0,
) -> PairedBootstrapResult:
    """Compute a percentile CI from paired complete-lifetime differences.

    Each condition may contain at most one record for a pairing key. Extra
    condition-specific lifetimes are excluded rather than treated as independent
    transitions. This forces callers to surface incomplete paired runs.
    """

    if bootstrap_draws <= 0:
        raise ValueError("bootstrap_draws must be positive")
    if not pairing_keys:
        raise ValueError("at least one pairing key is required")

    def select(condition: str) -> dict[tuple[Any, ...], float]:
        selected: dict[tuple[Any, ...], float] = {}
        for record in records:
            if record.get(condition_key) != condition:
                continue
            try:
                key = tuple(record[name] for name in pairing_keys)
                value = float(record[metric])
            except KeyError as error:
                raise ValueError(f"record missing required key {error.args[0]!r}") from error
            if key in selected:
                raise ValueError(
                    f"duplicate lifetime record for condition={condition!r}, key={key!r}"
                )
            selected[key] = value
        return selected

    reference_values = select(reference)
    treatment_values = select(treatment)
    shared_keys = sorted(reference_values.keys() & treatment_values.keys())
    if len(shared_keys) < 2:
        raise ValueError("at least two paired lifetimes are required")

    reference_samples = [reference_values[key] for key in shared_keys]
    treatment_samples = [treatment_values[key] for key in shared_keys]
    differences = [treatment - reference for reference, treatment in zip(
        reference_samples, treatment_samples, strict=True
    )]
    rng = random.Random(rng_seed)
    draw_means = []
    for _ in range(bootstrap_draws):
        draw_means.append(
            sum(differences[rng.randrange(len(differences))] for _ in differences)
            / len(differences)
        )
    draw_means.sort()
    lower_index = int(0.025 * (bootstrap_draws - 1))
    upper_index = int(0.975 * (bootstrap_draws - 1))
    return PairedBootstrapResult(
        metric=metric,
        reference=reference,
        treatment=treatment,
        paired_lifetimes=len(shared_keys),
        reference_mean=statistics.mean(reference_samples),
        treatment_mean=statistics.mean(treatment_samples),
        mean_difference=statistics.mean(differences),
        ci_lower=draw_means[lower_index],
        ci_upper=draw_means[upper_index],
        bootstrap_draws=bootstrap_draws,
        rng_seed=rng_seed,
    )


def result_as_dict(result: PairedBootstrapResult) -> dict[str, Any]:
    """Serialize a result without exposing callers to dataclass details."""

    return asdict(result)
