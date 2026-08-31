import importlib.util
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_hierarchical_thermal_v11.py"


def load_script():
    spec = importlib.util.spec_from_file_location("train_hierarchical_thermal_v11", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TinyEnvironment:
    def __init__(self):
        self.task = 0

    def reset(self, seed=None):
        self.task = 0
        return np.zeros(3), {"lifephy/v11_lifetime_initial_thermal_load": 0.04}

    def step(self, action):
        task = self.task
        self.task += 1
        boundary = self.task == 2
        info = {
            "lifephy/inner_task_boundary": True,
            "lifephy/lifetime_boundary": boundary,
            "lifephy/v11_task_index_at_selection": task,
            "lifephy/thermal_mode": "high" if int(np.asarray(action).item()) else "low",
            "lifephy/thermal_trip": False,
            "lifephy/v11_sensor_load": 0.04,
            "lifephy/thermal_load_at_mode_selection": 0.04,
            "lifephy/thermal_load": 0.05,
            "lifephy/v11_lifetime_initial_thermal_load": 0.04,
            "lifephy/v11_condition": "fixed",
        }
        return np.zeros(3), float(task + 1), False, boundary, info


class AlternatingModel:
    def __init__(self):
        self.calls = 0

    def predict(self, observation, state, episode_start, deterministic):
        action = np.asarray(self.calls % 2)
        self.calls += 1
        return action, state


def test_declared_policy_specs_are_complete():
    module = load_script()
    for arm in module.POLICY_ARMS:
        algorithm, _policy, kwargs = module.model_spec(arm)
        assert algorithm in {"PPO", "RecurrentPPO"}
        assert isinstance(kwargs, dict)
    with pytest.raises(ValueError):
        module.model_spec("posthoc_best")


def test_raw_evaluation_counts_tasks_and_lifetimes():
    module = load_script()
    aggregate, rows = module.evaluate_with_raw_rows(
        AlternatingModel(), TinyEnvironment(), task_episodes=4, seed=750000
    )
    assert len(rows) == 4
    assert aggregate["task_episodes"] == 4
    assert aggregate["completed_lifetimes"] == 2
    assert aggregate["mean_task_episode_reward"] == 1.5
    assert aggregate["high_power_selection_rate"] == 0.5
    assert aggregate["both_modes_lifetime_rate"] == 1.0
    assert [row["task_index"] for row in rows] == [0, 1, 0, 1]


def test_atomic_json_refuses_no_data_loss(tmp_path):
    module = load_script()
    target = tmp_path / "result.json"
    module.atomic_json(target, {"complete": True})
    assert target.read_text(encoding="utf-8").endswith("\n")
    assert not list(tmp_path.glob("*.tmp"))


def test_confirmatory_phase_requires_protocol_digest():
    module = load_script()
    arguments = Namespace(
        workers=8,
        total_task_decisions=100_000,
        eval_task_episodes=4_000,
        seed=8_300,
        torch_threads_per_process=1,
        episode_steps=100,
        episodes_per_lifetime=20,
        evaluation_seed=123,
        training_reward_scale=0.02,
        ent_coef=0.005,
        gamma=0.99,
        gae_lambda=0.95,
        study_phase="confirmatory",
        protocol_sha256=None,
    )
    with pytest.raises(SystemExit, match="protocol SHA-256"):
        module.validate_args(arguments)
    arguments.protocol_sha256 = "a" * 64
    module.validate_args(arguments)
