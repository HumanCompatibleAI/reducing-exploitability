import argparse
from pathlib import Path

import ray
import ray.cloudpickle as cloudpickle
from ray.tune import register_env
from tqdm import tqdm

from aprl_defense.agents.maddpg import MADDPGTrainer
from maddpg_rllib.env import MultiAgentParticleEnv


def main(args):
    local_dir = Path.home() / "ray_results" / "test2"
    # Resolve local dir
    local_dir = local_dir.resolve()

    ray.init(local_mode=True)

    # Initialize environment
    # Make sure environment accepts one-hot encoded action
    def create_mpe_env(mpe_args):
        env = MultiAgentParticleEnv(**mpe_args)
        return env

    register_env("mpe-push", create_mpe_env)

    for finetune_pol, fixed_pol in [("policy_0", "policy_1"), ("policy_1", "policy_0")]:
        # Re-load the victim config
        config_path = Path(args.victim_artifact).parent / ".." / "params.pkl"
        config_path = config_path.resolve()
        with open(config_path, "rb") as f:
            victim_config = cloudpickle.load(f)

        # Changes to config for the current env setup
        victim_config["env"] = "mpe-push"

        # Restore the trained agent with this trainer
        old_maddpg_trainer = MADDPGTrainer(env="mpe-push", config=victim_config)
        old_maddpg_trainer.restore(checkpoint_path=args.victim_artifact)

        # Policy we want to finetune uses local critic -> currently this is required, due to a bug in ray
        victim_config["multiagent"]["policies"][finetune_pol][3][
            "use_local_critic"
        ] = True
        victim_config["multiagent"]["policies_to_train"] = [
            finetune_pol
        ]  # Only train the policy we want to finetune

        # Create a new maddpg trainer which will be used to train the adversarial policy
        new_maddpg_trainer = MADDPGTrainer(env="mpe-push", config=victim_config)

        # Overwrite the weights for the agent that will not be trained (fixed_pol)
        new_maddpg_trainer.set_weights(old_maddpg_trainer.get_weights(fixed_pol))

        # Save the weights before training
        buffer_finetune = new_maddpg_trainer.get_weights()[finetune_pol][
            "_state"
        ].copy()
        buffer_fixed = new_maddpg_trainer.get_weights()[fixed_pol]["_state"].copy()

        # Train using trainable
        iterations = 50
        for iteration in tqdm(range(iterations)):
            _ = new_maddpg_trainer.train()

        checkpoint = new_maddpg_trainer.save(checkpoint_dir=local_dir)
        print("checkpoint saved at", checkpoint)

        # Get the new weights
        new_weights_finetune = new_maddpg_trainer.get_weights()[finetune_pol][
            "_state"
        ].copy()
        new_weights_fixed = new_maddpg_trainer.get_weights()[fixed_pol]["_state"].copy()

        print("Sanity checking weights")
        assert (
            all((a == b).all() for a, b in zip(new_weights_finetune, buffer_finetune))
            is False
        ), "Weights DID NOT CHANGE in the policy being TRAINED"
        assert all(
            (a == b).all() for a, b in zip(new_weights_fixed, buffer_fixed)
        ), "Weights CHAGNED in the FIXED policy"
    print("Tests ran without error")


def parse_args():
    parser = argparse.ArgumentParser("Adversarial Policy on MADDPG")
    parser.add_argument(
        "--victim_path",
        type=str,
        help="Provide the path to a pre-trained MADDPG checkpoint",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
