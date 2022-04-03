import pickle
import random
from pathlib import Path

import gym
import ray.rllib
from mujoco_py import MjSimState
from mujoco_py.builder import MujocoException

from aprl_defense.common.base_logger import logger


class MujocoToRllibWrapper(ray.rllib.MultiAgentEnv):
    """Stable baselines (multi vec env as wrapped from Mujoco) to RLlib wrapper"""

    def __init__(self, env):
        super().__init__()
        self.env = env
        # self.action_space = self.env.action_space
        # self.observation_space = self.env.observation_space
        # Iterate over tuples to create dicts for observation and action space
        observation_space = dict_from_tuple(env.observation_space)
        self.observation_space = gym.spaces.Dict(spaces=observation_space)
        action_space = dict_from_tuple(env.action_space)
        self.action_space = gym.spaces.Dict(spaces=action_space)
        # self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.action_buffer = []
        self.obs_buffer = []
        self.reward_buffer = []

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name)
            )
        return getattr(self.env, name)

    def step(self, action):
        # Transform back from dict to sequence
        action = list(action.values())

        self.action_buffer.append(action)
        if len(self.action_buffer) >= 25:
            self.action_buffer.pop(0)

        rand = random.randint(0, 5_000_000)
        if rand == 0:
            i = 0
            while Path(f"MUJOCO_STATE_131_{i}.pkl").is_file():
                i += 1
            self._dump_mujoco_state(self.env, f"MUJOCO_STATE_131_{i}.pkl", action)
        try:
            obs, rews, done, info = self.env.step(action)
        except MujocoException as e:
            # Dump the mujoco state
            self._dump_mujoco_state(self.env, "MUJOCO_STATE.pkl", action)
            logger.warning("ACTIONS:")
            logger.warning(self.action_buffer)
            logger.warning("Observations")
            logger.warning(self.obs_buffer)
            logger.warning("Rewards")
            logger.warning(self.reward_buffer)
            raise e

        self.obs_buffer.append(obs)
        self.reward_buffer.append(rews)
        if len(self.obs_buffer) >= 25:
            self.obs_buffer.pop(0)
        if len(self.reward_buffer) >= 50:
            self.reward_buffer.pop(0)

        # I think the most accurate way to map Mujocos done to RLlibs done is by using agent_done
        # And I assume the original done represents done['__all__']
        done_dict = {}
        for key in info.keys():
            done_dict[key] = info[key]["agent_done"]
        # We set to done for everyone when at least one agent is done. Otherwise we would run into problems because RLlib doesn't sample actions for agents that
        # are already done, but mujoco expects actions from all agents
        done_dict["__all__"] = True in done_dict.values()
        return dict_from_tuple(obs), dict_from_tuple(rews), done_dict, info

    def _dump_mujoco_state(self, env, file_name, last_action):
        last_obs = self.obs_buffer[-1]
        state = {
            "state_vector": env.state_vector(),
            "last_action": last_action,
            "last_obs": last_obs,
        }
        # Handles some rare edge cases
        try:
            state["qacc"] = env.unwrapped.env_scene.data.qacc
        except AttributeError:
            pass
        try:
            state["act"] = env.unwrapped.env_scene.data.act
        except AttributeError:
            pass
        file = open(file_name, mode="wb")
        pickle.dump(state, file)
        file.close()

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        return dict_from_tuple(obs)

    def render(self, mode="human", **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self):
        return self.env.close()


class MujocoEnvFromStateWrapper(ray.rllib.MultiAgentEnv):
    """Stable baselines (multi vec env as wrapped from Mujoco) to RLlib wrapper"""

    def __init__(self, env, state):
        super().__init__()
        self.env = env
        # self.action_space = self.env.action_space
        # self.observation_space = self.env.observation_space
        # Iterate over tuples to create dicts for observation and action space
        observation_space = dict_from_tuple(env.observation_space)
        self.observation_space = gym.spaces.Dict(spaces=observation_space)
        action_space = dict_from_tuple(env.action_space)
        self.action_space = gym.spaces.Dict(spaces=action_space)
        # self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.action_buffer = []
        self.obs_buffer = []
        self.reward_buffer = []
        self.state_set = False
        self.state = state

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name)
            )
        return getattr(self.env, name)

    def step(self, action):
        if not self.state_set:
            self.state_set = True
            state = MjSimState.from_flattened(self.state, self)
            self.env.set_state(state.qpos, state.qvel)
        return self.env.step(action)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        return dict_from_tuple(obs)

    def render(self, mode="human", **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped


def dict_from_tuple(tuple):
    """Create a dict with tuple index as key from given tuple."""
    dictionary = {}
    for i, e in enumerate(tuple):
        dictionary[i] = e

    return dictionary
