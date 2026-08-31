#!/usr/bin/env python3
"""Measure CPU simulation throughput and verify the GPU learner stack.

This is a pre-training calibration, not a learning experiment. It intentionally
does no gradient update and reports CUDA unavailability as a structured result
so the same command is safe in CPU-only CI or a container without GPU passthrough.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from lifephybench.envs.mujoco_pusher import ActuatorWearConfig, PusherActuatorWear


@dataclass(frozen=True)
class CpuThroughput:
    workers: int
    total_steps: int
    elapsed_seconds: float
    aggregate_steps_per_second: float
    mean_worker_steps_per_second: float


def rollout_worker(environment_id: str, seed: int, steps: int) -> tuple[int, float]:
    """Run one deterministic random-action rollout in an isolated process."""

    env = PusherActuatorWear.make(
        ActuatorWearConfig(
            wear_rate=0.001,
            thermal_enabled=True,
            thermal_heat_rate=0.005,
            joint_aging_enabled=True,
            joint_aging_rate=0.001,
        ),
        environment_id=environment_id,
    )
    rng = np.random.default_rng(seed)
    try:
        _observation, _info = env.reset_lifetime(seed=seed, lifetime_id=seed)
        start = time.perf_counter()
        completed = 0
        while completed < steps:
            action = rng.uniform(env.action_space.low, env.action_space.high)
            _observation, _reward, terminated, truncated, _info = env.step(action)
            completed += 1
            if (terminated or truncated) and completed < steps:
                _observation, _info = env.reset()
        return completed, time.perf_counter() - start
    finally:
        env.close()


def measure_cpu(
    environment_id: str, workers: int, steps_per_worker: int
) -> CpuThroughput:
    start = time.perf_counter()
    if workers == 1:
        results = [rollout_worker(environment_id, 1000, steps_per_worker)]
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            futures = [
                executor.submit(rollout_worker, environment_id, 1000 + index, steps_per_worker)
                for index in range(workers)
            ]
            results = [future.result() for future in futures]
    elapsed = time.perf_counter() - start
    total_steps = sum(steps for steps, _ in results)
    worker_rates = [steps / duration for steps, duration in results]
    return CpuThroughput(
        workers=workers,
        total_steps=total_steps,
        elapsed_seconds=elapsed,
        aggregate_steps_per_second=total_steps / elapsed,
        mean_worker_steps_per_second=float(np.mean(worker_rates)),
    )


def probe_cuda(matrix_size: int, iterations: int) -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch_not_installed"}
    if not torch.cuda.is_available():
        return {
            "available": False,
            "reason": "torch_cuda_unavailable",
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
        }
    device = torch.device("cuda:0")
    try:
        torch.cuda.reset_peak_memory_stats(device)
        left = torch.randn((matrix_size, matrix_size), device=device)
        right = torch.randn((matrix_size, matrix_size), device=device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            left = left @ right
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        return {
            "available": True,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "matrix_size": matrix_size,
            "iterations": iterations,
            "elapsed_seconds": elapsed,
            "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
        }
    except RuntimeError as error:
        return {"available": False, "reason": f"cuda_runtime_error: {error}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-id", default="Pusher-v5")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--steps-per-worker", type=int, default=2_000)
    parser.add_argument("--gpu-matrix-size", type=int, default=2_048)
    parser.add_argument("--gpu-iterations", type=int, default=10)
    args = parser.parse_args()
    if any(workers <= 0 for workers in args.workers):
        raise SystemExit("--workers values must be positive")
    if args.steps_per_worker <= 0 or args.gpu_matrix_size <= 0 or args.gpu_iterations <= 0:
        raise SystemExit("step, matrix-size, and iteration values must be positive")
    cpu = [
        asdict(measure_cpu(args.environment_id, workers, args.steps_per_worker))
        for workers in args.workers
    ]
    print(
        json.dumps(
            {
                "phase": "gpu_training_preflight_not_learning",
                "configuration": vars(args),
                "cpu_throughput": cpu,
                "cuda_probe": probe_cuda(args.gpu_matrix_size, args.gpu_iterations),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
