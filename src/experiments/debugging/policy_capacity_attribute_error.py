import random

import gym
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env import PettingZooEnv
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.tune import register_env

ray.init(local_mode=False)

num_policies = 10
policy_map_capacity = 5  # AttributeError raised when < num_policies
env = "cartpole"
# env = 'rps'
init_all_at_once = True

if env == "cartpole":
    register_env("MainEnv-v0", lambda _: MultiAgentCartPole({"num_agents": 2}))
    single_dummy_env = gym.make("CartPole-v0")
    obs_space = single_dummy_env.observation_space
    act_space = single_dummy_env.action_space
else:  # env == 'rps':
    from pettingzoo.classic import rps_v2

    def env_creator(args):
        env = rps_v2.env()
        return env

    register_env("MainEnv-v0", lambda config: PettingZooEnv(env_creator(config)))
    single_dummy_env = env_creator({})
    obs_space = single_dummy_env.observation_spaces["player_0"]
    act_space = single_dummy_env.action_spaces["player_0"]

if init_all_at_once:
    policies = {str(i): (None, obs_space, act_space, {}) for i in range(num_policies)}
else:
    policies = {"0": (None, obs_space, act_space, {})}


def policy_mapping_fn(agent_id, episode, **kwargs):
    return str(random.randrange(0, num_policies))


print("==================")
print(f"init all at once? {init_all_at_once}")
print(f"num_policies = {num_policies}")
print(f"policy_map_capacity = {policy_map_capacity}")
print(f"env = {env}")
print("==================")

config = {
    "framework": "tf",
    "num_workers": 10,
    "multiagent": {
        "policy_map_capacity": policy_map_capacity,
        "policies": policies,
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": None,  # All
    },
}

ppo_trainer = PPOTrainer(env="MainEnv-v0", config=config)

if not init_all_at_once:
    for i in range(1, num_policies):
        _ = ppo_trainer.add_policy(
            policy_id=str(i), policy_cls=type(ppo_trainer.get_policy("0"))
        )

    ppo_trainer.workers.sync_weights()

max_timesteps = 8000

timesteps_total = 0
while timesteps_total < max_timesteps:
    results = ppo_trainer.train()
    timesteps_total = results["timesteps_total"]
