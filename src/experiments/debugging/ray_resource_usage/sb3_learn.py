import gym
import time
from stable_baselines3 import PPO

from ray.rllib.env.wrappers.open_spiel import OpenSpielEnv
import pyspiel


class RLlibToGymWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.prev_state = None

    def reset(self, ):
        next_state = self.env.reset()
        return next_state[0]

    def step(self, action):
        next_state, reward, done, info = self.env.step({0: action, 1: action})
        state = next_state[0] if 0 in next_state else self.prev_state
        self.prev_state = state
        return state, reward[0], done[0], info


def create_env(_):
    return OpenSpielEnv(pyspiel.load_game("laser_tag"))


timesteps = 41_000_000

env = create_env({})
env = RLlibToGymWrapper(env)

start = time.time()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=41_000_000)

print("==================")
print(f"{timesteps} timesteps of learning took {(time.time() - start) // 60} minutes")
print("==================")
