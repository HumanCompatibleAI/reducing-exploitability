import os
import fire
from typing import Optional


def run_multi_atack(
    env_config: str,
    num_workers: int,
    sweep_name: str,
    victim_artifact: str,
    description: str = "",
    num_seeds: int = 3,
    parallel: bool = True,
    adversary_id: Optional[int] = None,
    victim_policy_name: Optional[str] = None,
    train_batch_size: Optional[int] = None,
):
    make_parallel = "&" if parallel else ""
    for i in range(num_seeds):
        command = (
            f"python -m aprl_defense.train "
            f'-f "gin/icml/attack/{env_config}" '
            f'-p "TrialSettings.num_workers = {num_workers}" '
            f"-p \"TrialSettings.wandb_group = '{sweep_name}'\" "
            f"-p \"TrialSettings.description = '{description}'\" "
            f"-p \"attack.victim_artifact = '{victim_artifact}'\" "
        )
        # If these are not set, take whatever is in the gin config
        if adversary_id is not None:
            command += f'-p "attack.adversary_id = {adversary_id}" '
        if victim_policy_name is not None:
            command += f"-p \"attack.victim_policy_name = '{victim_policy_name}'\" "
        if train_batch_size is not None:
            command += f'-p "RLSettings.train_batch_size = {train_batch_size}" '
        # Execute all seeds in parallel
        if i != num_seeds - 1:  # Don't add & to last command
            command += f"{make_parallel} "

        print(f"Starting {i} with command:")
        print(command)
        os.system(command)


if __name__ == "__main__":
    fire.Fire(run_multi_atack)
