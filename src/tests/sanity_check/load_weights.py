import argparse
from pathlib import Path

import ray
import ray.cloudpickle as cloudpickle

# from aprl_defense.agents.maddpg import MADDPGTrainer
from maddpg_rllib.env import MultiAgentParticleEnv
from ray.rllib.contrib.maddpg import MADDPGTrainer
from ray.tune import register_env
from ray.tune.logger import pretty_print
from tensorboardX import SummaryWriter


def main(args):
    scenario_name = "simple_push"
    local_dir = Path.home() / "ray_results" / "test"
    # Resolve local dir
    local_dir = local_dir.resolve()

    config_path = Path(args.victim_artifact).parent / ".." / "params.pkl"
    config_path = config_path.resolve()
    with open(config_path, "rb") as f:
        victim_config = cloudpickle.load(f)

    ray.init(local_mode=True)

    # Initialize environment
    def create_mpe_env(mpe_args):
        env = MultiAgentParticleEnv(**mpe_args)

        return env

    register_env("current-env", create_mpe_env)

    _ = create_mpe_env({"scenario_name": scenario_name})

    # Changes to config for the current env setup
    # victim_config.update(config)
    # victim_config['multiagent']
    # victim_config['framework'] = 'tfe'
    victim_config["env"] = "current-env"
    victim_config["horizon"] = 25
    victim_config["n_step"] = 1
    victim_config["explore"] = False

    # Restore the trained agent with this trainer
    old_maddpg_trainer = MADDPGTrainer(env="mpe-push", config=victim_config)
    old_maddpg_trainer.restore(checkpoint_path=args.victim_artifact)

    # victim_config['local_dir'] = local_dir
    finetune_policy = "policy_0"

    writer = SummaryWriter(
        logdir=str(local_dir / "tb" / "sanity-weights"),
    )

    # Create a new maddpg trainer which will be used to train the adversarial policy
    new_maddpg_trainer = MADDPGTrainer(env="mpe-push", config=victim_config)

    # Overwrite the weights for the agent that will not be trained (policy_1)
    fixed_policy = "policy_1"
    new_maddpg_trainer.set_weights(old_maddpg_trainer.get_weights(fixed_policy))
    new_maddpg_trainer.set_weights(old_maddpg_trainer.get_weights(finetune_policy))

    print("Config:")
    print(pretty_print(new_maddpg_trainer.config))
    # # Train using trainable
    max_episodes = 60000
    episodes_total = 0
    while episodes_total < max_episodes:
        results = new_maddpg_trainer.train()
        new_maddpg_trainer.get_state()
        episodes_total = results["episodes_total"]
        # print(episodes_total)

        writer.add_scalar(
            tag="policy_reward_mean/policy_0",
            scalar_value=results["policy_reward_mean"][finetune_policy],
            global_step=episodes_total,
        )
        writer.add_scalar(
            tag="policy_reward_mean/policy_1",
            scalar_value=results["policy_reward_mean"][fixed_policy],
            global_step=episodes_total,
        )
    checkpoint = new_maddpg_trainer.save(checkpoint_dir=local_dir)
    print("checkpoint saved at", checkpoint)


def parse_args():
    parser = argparse.ArgumentParser("Adversarial Policy on MADDPG")
    parser.add_argument("--victim_path", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
