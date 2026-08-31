import unittest

import numpy as np

from lifephybench.envs.lifetime import LifetimeEpisodeScheduler, LifetimeStreamWrapper
from lifephybench.envs.mujoco_pusher import ActuatorWearConfig, PusherActuatorWear


class LifetimeSchedulerTests(unittest.TestCase):
    def test_scheduler_preserves_then_resets_physical_state(self):
        base = PusherActuatorWear.make(
            ActuatorWearConfig(wear_rate=0.1), max_episode_steps=1
        )
        env = LifetimeEpisodeScheduler(base, episodes_per_lifetime=2)
        try:
            env.reset(seed=1)
            env.step(np.full(env.action_space.shape, 2.0))
            wear_after_first_episode = base.wear
            self.assertGreater(wear_after_first_episode, 0.0)
            env.reset()
            self.assertEqual(base.wear, wear_after_first_episode)
            env.step(np.full(env.action_space.shape, 2.0))
            env.reset()
            self.assertEqual(base.wear, 0.0)
        finally:
            env.close()

    def test_stream_preserves_memory_boundary_until_lifetime_end(self):
        base = PusherActuatorWear.make(
            ActuatorWearConfig(wear_rate=0.1), max_episode_steps=1
        )
        env = LifetimeStreamWrapper(base, episodes_per_lifetime=2)
        try:
            env.reset(seed=2)
            _observation, _reward, terminated, truncated, info = env.step(
                np.full(env.action_space.shape, 2.0)
            )
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["lifephy/inner_task_boundary"])
            self.assertFalse(info["lifephy/lifetime_boundary"])
            self.assertGreater(base.wear, 0.0)
            _observation, _reward, terminated, truncated, info = env.step(
                np.full(env.action_space.shape, 2.0)
            )
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertTrue(info["lifephy/lifetime_boundary"])
            env.reset()
            self.assertEqual(base.wear, 0.0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
