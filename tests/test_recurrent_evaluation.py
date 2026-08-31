import unittest

import numpy as np

from lifephybench.envs.lifetime import LifetimeEpisodeScheduler, LifetimeStreamWrapper
from lifephybench.envs.mujoco_pusher import ActuatorWearConfig, PusherActuatorWear
from lifephybench.recurrent_evaluation import evaluate_task_episodes


class ZeroModel:
    def predict(self, observation, state, episode_start, deterministic):
        del observation, episode_start, deterministic
        return np.zeros(7, dtype=np.float64), state


class RecurrentEvaluationTests(unittest.TestCase):
    def make_base(self):
        return PusherActuatorWear.make(
            ActuatorWearConfig(wear_rate=0.01), max_episode_steps=1
        )

    def test_task_episode_metric_is_shared_across_memory_modes(self):
        episode_environment = LifetimeEpisodeScheduler(self.make_base(), 2)
        lifetime_environment = LifetimeStreamWrapper(self.make_base(), 2)
        try:
            episode = evaluate_task_episodes(ZeroModel(), episode_environment, 4, seed=3)
            lifetime = evaluate_task_episodes(ZeroModel(), lifetime_environment, 4, seed=3)
            self.assertEqual(episode.task_episodes, 4)
            self.assertEqual(lifetime.task_episodes, 4)
            self.assertEqual(episode.completed_lifetimes, 2)
            self.assertEqual(lifetime.completed_lifetimes, 2)
        finally:
            episode_environment.close()
            lifetime_environment.close()


if __name__ == "__main__":
    unittest.main()
