import unittest
from pathlib import Path
from runpy import run_path

from lifephybench.envs.mujoco_pusher import ActuatorWearConfig, PusherActuatorWear


class TrainingCalibrationTests(unittest.TestCase):
    def test_short_calibration_rollout(self):
        functions = run_path(
            Path(__file__).resolve().parents[1] / "scripts/calibrate_training_stack.py"
        )
        rollout_worker = functions["rollout_worker"]
        steps, elapsed = rollout_worker("Reacher-v5", seed=101, steps=10)
        self.assertEqual(steps, 10)
        self.assertGreater(elapsed, 0.0)

    def test_wrapper_remains_constructible_for_preflight(self):
        env = PusherActuatorWear.make(ActuatorWearConfig(), environment_id="Pusher-v5")
        try:
            env.reset_lifetime(seed=102)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
