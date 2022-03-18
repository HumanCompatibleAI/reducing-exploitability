"""Based on https://github.com/justinkterry/maddpg-rllib"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gym.spaces import Discrete, Box, MultiDiscrete
from ray import rllib
from ext.envs.make_env import make_env

import numpy as np
import time


class RLlibMultiAgentParticleEnv(rllib.MultiAgentEnv):
    """Wraps OpenAI Multi-Agent Particle env to be compatible with RLLib multi-agent."""

    # Necessary so recording works in rllib
    metadata = {
        "render.modes": ["rgb_array", "human"],
    }

    def __init__(self, max_steps=None, one_hot_agents=None, **mpe_args):
        """Create a new Multi-Agent Particle env compatible with RLlib.

        Arguments:
            max_steps: Set to a number so the environment will return done==True after this many steps.
                If set to None env will not return done==True
            one_hot_agents: Set to None: All agents are expected to provide discrete actions
                Set to List(bool): one_hot_agents[i] == False: Expect discrete actions from agent i
                   one_hot_agents[i] == True: Expect one-hot, logit or probabilities as action from agent i
                   This will be converted to discrete with np.argmax before passing it to the wrapped env
            mpe_args (dict): Arguments to pass to the underlying
                make_env.make_env instance.

        Examples:
            >>> from rllib_env import RLlibMultiAgentParticleEnv
            >>> env = RLlibMultiAgentParticleEnv(scenario_name="simple_reference")
            >>> print(env.reset())
        """

        self._env = make_env(**mpe_args)
        self.num_agents = self._env.n
        self._agent_ids = list(range(self.num_agents))

        self.observation_space_dict = self._make_dict(self._env.observation_space)
        self.action_space_dict = self._make_dict(self._env.action_space)

        self.step_counter = 0
        self.max_steps = max_steps

        self.one_hot_agents = one_hot_agents

    def reset(self):
        """Resets the env and returns observations from ready agents.

        Returns:
            obs_dict: New observations for each ready agent.
        """
        obs_list = self._env.reset()
        # Quick hack to ensure observations are float32
        new_list = []
        for obs in obs_list:
            new_list.append(obs.astype(np.float32))

        obs_dict = self._make_dict(new_list)
        return obs_dict

    def step(self, action_dict):
        """Returns observations from ready agents.

        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.

        Returns:
            obs_dict:
                New observations for each ready agent.
            rew_dict:
                Reward values for each ready agent.
            done_dict:
                Done values for each ready agent.
                The special key "__all__" (required) is used to indicate env termination.
            info_dict:
                Optional info values for each agent id.
        """

        actions = list(action_dict.values())

        if self.one_hot_agents is not None:
            for i, one_hot in enumerate(self.one_hot_agents):
                if one_hot:
                    # Convert from one-hot to discrete action
                    converted_action = np.argmax(actions[i])
                    actions[i] = converted_action

        obs_list, rew_list, done_list, info_list = self._env.step(actions)

        # attacker id = 0
        # defender id = 1
        # Change the reward, such that it is zero sum
        rew_list[1] = -rew_list[0]

        self.step_counter += 1
        # If early stopping is activated and we are over the step threshold
        if self.max_steps is not None and self.step_counter >= self.max_steps:
            done_list = [True] * len(done_list)
            self.step_counter = 0

        # Quick hack to ensure observations are float32
        new_list = []
        for obs in obs_list:
            new_list.append(obs.astype(np.float32))

        obs_dict = self._make_dict(new_list)
        rew_dict = self._make_dict(rew_list)
        done_dict = self._make_dict(done_list)
        done_dict["__all__"] = all(done_list)
        # FIXME: Currently, this is the best option to transfer agent-wise termination signal without touching RLlib code hugely.
        # FIXME: Hopefully, this will be solved in the future.
        info_dict = self._make_dict([{"done": done} for done in done_list])

        return obs_dict, rew_dict, done_dict, info_dict

    def render(self, mode="human"):
        time.sleep(0.05)
        return self._env.render(mode=mode)[0]

    def _make_dict(self, values):
        return dict(zip(self._agent_ids, values))


if __name__ == "__main__":
    for scenario_name in [
        "simple",
        "simple_adversary",
        "simple_crypto",
        "simple_push",
        "simple_reference",
        "simple_speaker_listener",
        "simple_spread",
        "simple_tag",
        "simple_world_comm",
    ]:
        print("scenario_name: ", scenario_name)
        env = RLlibMultiAgentParticleEnv(scenario_name=scenario_name)
        print("obs: ", env.reset())
        print(env.observation_space_dict)
        print(env.action_space_dict)

        action_dict = {}
        for i, ac_space in env.action_space_dict.items():
            sample = ac_space.sample()
            if isinstance(ac_space, Discrete):
                action_dict[i] = np.zeros(ac_space.n)
                action_dict[i][sample] = 1.0
            elif isinstance(ac_space, Box):
                action_dict[i] = sample
            elif isinstance(ac_space, MultiDiscrete):
                print("sample: ", sample)
                print("ac_space: ", ac_space.nvec)
                action_dict[i] = np.zeros(sum(ac_space.nvec))
                start_ls = np.cumsum([0] + list(ac_space.nvec))[:-1]
                for j in list(start_ls + sample):
                    action_dict[i][j] = 1.0
            else:
                raise NotImplementedError

        print("action_dict: ", action_dict)

        for i in env.step(action_dict):
            print(i)
