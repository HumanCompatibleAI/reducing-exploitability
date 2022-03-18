import random
import time

import gym
import pyspiel
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.wrappers.open_spiel import OpenSpielEnv
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.tune import register_env
from ray.tune.logger import pretty_print

ray.init(local_mode=False)

num_policies = 20
num_workers = 10
env = "ma_cartpole"
# env = 'laser_tag'
init_all_at_once = False
print_config = True

if env == "ma_cartpole":
    # Simple environment with num_policies independent cartpole entities
    register_env("current-env", lambda _: MultiAgentCartPole({"num_agents": 2}))
elif env == "laser_tag":
    register_env("current-env", lambda _: OpenSpielEnv(pyspiel.load_game("laser_tag")))

single_dummy_env = gym.make("CartPole-v0")
obs_space = single_dummy_env.observation_space
act_space = single_dummy_env.action_space

if init_all_at_once:
    policies = {str(i): (None, obs_space, act_space, {}) for i in range(num_policies)}
else:
    policies = {"0": (None, obs_space, act_space, {})}


def policy_mapping_fn(agent_id, episode, **kwargs):
    return str(random.randrange(0, num_policies))


print("==================")
print(f"init all at once? {init_all_at_once}")
print(f"num_policies = {num_policies}")
print(f"num_workers {num_workers}")
print(f"env = {env}")
config = {
    # 'num_cpus_for_driver': 10,
    # 'tf_session_args': {
    #     'inter_op_parallelism_threads': 1,
    #     'intra_op_parallelism_threads': 1,
    # },
    # "local_tf_session_args": {
    #     # Allow a higher level of parallelism by default, but not unlimited
    #     # since that can cause crashes with many concurrent drivers.
    #     "intra_op_parallelism_threads": 1,
    #     "inter_op_parallelism_threads": 1,
    # },
    "framework": "tf",
    "num_workers": num_workers,
    "multiagent": {
        "policy_map_capacity": 200,
        "policies": policies,
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": None,  # All
    },
}
start = time.time()

ppo_trainer = PPOTrainer(env="current-env", config=config)

if not init_all_at_once:
    for i in range(1, num_policies):
        _ = ppo_trainer.add_policy(
            policy_id=str(i), policy_cls=type(ppo_trainer.get_policy("0"))
        )

    ppo_trainer.workers.sync_weights()

print("==================")
print(f"PPO init took {time.time() - start} seconds")
print(f"number of policies created: {len(ppo_trainer.get_weights())}")
print("==================")

if print_config:
    # Print the config
    print(pretty_print(ppo_trainer.config))

start = time.time()

max_timesteps = 8000

timesteps_total = 0
while timesteps_total < max_timesteps:
    results = ppo_trainer.train()
    timesteps_total = results["timesteps_total"]

print("==================")
print(f"{max_timesteps} timesteps took {time.time() - start} seconds")
print("==================")
