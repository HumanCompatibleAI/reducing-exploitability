"""
Based on RLLIB example: https://github.com/ray-project/ray/blob/master/rllib/examples/two_step_game.py

This is a minimal example for
- MADDPG
- Not training all policies with the help of 'policies_to_train'
"""

import ray
from gym.spaces import Discrete
from ray import tune
from ray.rllib.examples.env.two_step_game import TwoStepGame

if __name__ == "__main__":
    config = {
        "env_config": {
            "actions_are_logits": True,
        },
        "multiagent": {
            "policies": {
                "pol1": (
                    None,
                    Discrete(6),
                    TwoStepGame.action_space,
                    {
                        "agent_id": 0,
                        # This fixes the problem
                        # "use_local_critic": True
                    },
                ),
                "pol2": (
                    None,
                    Discrete(6),
                    TwoStepGame.action_space,
                    {
                        "agent_id": 1,
                    },
                ),
            },
            "policy_mapping_fn": lambda x: "pol1" if x == 0 else "pol2",
            "policies_to_train": ["pol1"],  # This causes an exception
        },
        "framework": "tf",
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 0,
    }

    ray.init(num_cpus=2)

    stop = {
        "episode_reward_mean": 7,
        "timesteps_total": 50000,
        "training_iteration": 200,
    }

    config = dict(
        config,
        **{
            "env": TwoStepGame,
        }
    )

    results = tune.run(  # MADDPGTrainer,
        "contrib/MADDPG", stop=stop, config=config, verbose=1
    )

    ray.shutdown()
