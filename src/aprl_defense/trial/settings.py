from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List

import gin


@gin.configurable
@dataclass
class TrialSettings:
    mode: str = "selfplay"
    out_path: Union[Path, str] = "out/"
    checkpoint_freq_M: Optional[float] = None
    num_checkpoints: Optional[int] = None
    seed: Optional[int] = None
    num_workers: int = 4
    override: Optional[str] = None
    override_f: List[str] = field(default_factory=list)
    continue_artifact: Optional[str] = None
    override_config: bool = False
    num_cpus: Optional[int] = None
    ray_local: bool = False
    run_name: Optional[str] = None
    disable_log: bool = False
    description: str = ""
    wandb_project: str = "pbrl-defense-icml"
    wandb_group: str = "dev"
    framework: str = "tf"
    lr: float = 5.0e-05
    num_gpus: int = 1


@gin.configurable
@dataclass
class RLSettings:
    # RL
    env: str = "mpe_simple_push"
    alg: str = "ppo"
    max_timesteps: int = 1_500_000
    train_batch_size: int = 4000  # Number of rollout timesteps would be more accurate
    horizon: Optional[int] = None  # If None, episodes only end with done signal
    num_envs_per_worker: int = 10
    num_sgd_iter: int = 10
    limit_policy_cache: bool = True
    # PPO only
    sgd_minibatch_size: int = 20000
