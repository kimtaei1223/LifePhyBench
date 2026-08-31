import unittest

import numpy as np

from lifephybench.envs.history import SelectiveFrameStack
from lifephybench.envs.lifetime import LifetimeEpisodeScheduler
from lifephybench.envs.mujoco_pusher import ActuatorWearConfig, PusherActuatorWear


class SelectiveFrameStackTests(unittest.TestCase):
    def make_environment(self, history_mode: str) -> SelectiveFrameStack:
        base = PusherActuatorWear.make(
            ActuatorWearConfig(wear_rate=0.1), max_episode_steps=1
        )
        scheduler = LifetimeEpisodeScheduler(base, episodes_per_lifetime=2)
        return SelectiveFrameStack(scheduler, stack_size=3, history_mode=history_mode)

    def test_task_history_clears_at_every_task_reset(self):
        env = self.make_environment("task")
        try:
            initial, _info = env.reset(seed=4)
            width = initial.shape[-1] // 3
            self.assertTrue(np.all(initial[:-width] == 0.0))
            env.step(np.zeros(env.action_space.shape))
            reset_observation, _info = env.reset()
            self.assertTrue(np.all(reset_observation[:-width] == 0.0))
        finally:
            env.close()

    def test_lifetime_history_survives_task_reset_but_not_lifetime_reset(self):
        env = self.make_environment("lifetime")
        try:
            initial, _info = env.reset(seed=5)
            width = initial.shape[-1] // 3
            env.step(np.zeros(env.action_space.shape))
            continued, _info = env.reset()
            self.assertFalse(np.all(continued[:-width] == 0.0))
            env.step(np.zeros(env.action_space.shape))
            new_lifetime, _info = env.reset()
            self.assertTrue(np.all(new_lifetime[:-width] == 0.0))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
