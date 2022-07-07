import random
import time

import pyspiel
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer, ppo
from ray.rllib.env.wrappers.open_spiel import OpenSpielEnv
from ray.tune import register_env
from ray.tune.logger import pretty_print

num_policies = 11
num_workers = 10
print_config = True
local_mode = False
max_timesteps = 11_000_000
train_batch_size = 100_000
envs_per_worker = 1

print(f"Start")
ray.init(local_mode=local_mode, dashboard_port=8268)
print(f"Initialized ray!")

def create_env(_):
    return OpenSpielEnv(pyspiel.load_game("laser_tag"))


register_env("current-env", create_env)
env = create_env({})

obs_space = env.observation_space
act_space = env.action_space

policies = {str(i): (None, obs_space, act_space, {}) for i in range(num_policies)}


def policy_mapping_fn(agent_id, episode, **kwargs):
    if agent_id == 0:
        return str(0)
    else:
        return str(random.randrange(1, num_policies))


config = {
    "env": "current-env",
    "log_level": "INFO",
    "framework": "torch",
    "train_batch_size": train_batch_size,
    "num_workers": num_workers,
    "num_envs_per_worker": envs_per_worker,
    "horizon": 1000,
    "multiagent": {
        # "policy_map_capacity": 200,
        "policies": policies,
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": None,  # All
    },
}
ppo_trainer = PPOTrainer(env="current-env", config=config)

print("==================")
print(f"num_policies = {num_policies}")
print(f"num_workers {num_workers}")
if print_config:
    # Print the config
    print(pretty_print(ppo_trainer.config))

tune.run(
    ppo.PPOTrainer,
    stop={"timesteps_total": max_timesteps},
    config=config
)

print("==================")
print(f"{max_timesteps} timesteps took {(time.time() - start) // 60} minutes")
print("==================")
