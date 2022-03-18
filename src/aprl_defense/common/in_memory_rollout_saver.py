from collections import defaultdict

import numpy as np


class InMemoryRolloutSaver:
    def __init__(
        self,
        multiagent=True,
    ):
        assert (
            multiagent
        ), "currently InMemoryRolloutSaver is only supported for multiagent==True"
        self._current_reward_sums = defaultdict(int)
        self._episode_rewards = defaultdict(list)
        self._total_steps = 0
        self.mean_rewards = None

    def begin_rollout(self):
        self._current_reward_sums = defaultdict(int)
        self._episodes = defaultdict(list)

    def end_rollout(self):
        mean_rewards = {}
        for key, episode_rewards in self._episode_rewards.items():
            mean_rewards[key] = np.array(episode_rewards).mean()
        self.mean_rewards = mean_rewards

    def append_step(self, obs, action, next_obs, reward, done, info):
        """Called for every step in rollout"""
        if done:
            for key, reward_sum in self._current_reward_sums.items():
                self._episode_rewards[key].append(reward_sum)
        else:
            for key, value in reward.items():
                self._current_reward_sums[key] += value

        self._total_steps += 1
