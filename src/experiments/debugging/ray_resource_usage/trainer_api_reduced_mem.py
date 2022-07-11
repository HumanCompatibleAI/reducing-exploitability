import random
import time
from collections import deque

import pyspiel
import ray
from ray.rllib.agents.ppo import PPOTrainer
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
    "log_level": "INFO",
    "framework": "torch",
    "train_batch_size": train_batch_size,
    "num_workers": num_workers,
    "num_envs_per_worker": envs_per_worker,
    "horizon": 1000,
    "multiagent": {
        "policy_map_capacity": 3,
        "policies": policies,
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": None,  # All
    },
}
ppo_trainer = PPOTrainer(env="current-env", config=config)


def increase_driver_policy_capacity(worker):
    if worker.worker_index == 0:
        worker.policy_map.deque = deque(maxlen=num_policies)
        # TODO copy from original deque
        for item in worker.policy_map.cache:
            worker.policy_map.deque.append(item)


ppo_trainer.workers.foreach_worker(increase_driver_policy_capacity)

print("==================")
print(f"num_policies = {num_policies}")
print(f"num_workers {num_workers}")
if print_config:
    # Print the config
    print(pretty_print(ppo_trainer.config))

start = time.time()

timesteps_total = 0
while timesteps_total < max_timesteps:
    results = ppo_trainer.train()
    timesteps_total = results["timesteps_total"]
    print(f"Timesteps:{timesteps_total}")

print("==================")
print(f"{max_timesteps} timesteps took {(time.time() - start) // 60} minutes")
print("==================")
