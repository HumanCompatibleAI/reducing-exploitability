import os

import fire


def selfplay_sweep(num_workers: int, sweep_name: str, num_seeds: int, env_config: str):
    for seed in range(num_seeds):
        os.system(
            "python3 -m aprl_defense.train "
            f'-f "gin/icml/selfplay/{env_config}" '
            f'-p "TrialSettings.num_workers = {num_workers}" '
            f"-p \"TrialSettings.wandb_group = '{sweep_name}'\" "
            " &"
        )
        print(f"Started {seed}")


if __name__ == "__main__":
    fire.Fire(selfplay_sweep)
