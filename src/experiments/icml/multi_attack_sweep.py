import os
import fire
from typing import Optional, List

from experiments.icml.multi_attack import run_multi_atack


def run_multi_atack_sweep(
    env_config: str,
    num_workers: int,
    sweep_name: str,
    victim_artifacts: List[str],
    description: str = "",
    num_seeds: int = 3,
    parallel: bool = True,
    adversary_id: Optional[int] = None,
    victim_policy_name: Optional[str] = None,
    train_batch_size: Optional[int] = None,
):
    for victim_artifact in victim_artifacts:
        run_multi_atack(
            env_config,
            num_workers,
            sweep_name,
            victim_artifact,
            description,
            num_seeds,
            parallel,
            adversary_id,
            victim_policy_name,
            train_batch_size,
        )


if __name__ == "__main__":
    fire.Fire(run_multi_atack_sweep)
