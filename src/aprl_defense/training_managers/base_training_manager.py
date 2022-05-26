import json
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Optional, Any

import gin
import numpy as np
import ray
import wandb
from ray.tune import Trainable
from ray.tune.logger import pretty_print
from tqdm import tqdm

import aprl_defense
import aprl_defense.common.utils
import aprl_defense.configs.common
from aprl_defense.common.artifact_manager import ArtifactManager
from aprl_defense.common.base_logger import logger
from aprl_defense.common.rllib_io import save_params
from aprl_defense.common.utils import (
    get_base_train_config,
    init_env,
)
from aprl_defense.common.utils import trainer_cls_from_str, CustomMujocoMetricsCallbacks
from aprl_defense.pbt.utils import custom_eval_log
from aprl_defense.trial.settings import RLSettings, TrialSettings
from ext.aprl.training.scheduling import Scheduler


class BaseTrainingManager(ABC):
    """Base class for all training managers. train() function of manager will be called
    in main function."""

    @abstractmethod
    def train(self):
        pass


class SingleJobTrainingManager(BaseTrainingManager):
    """Base class for all training managers that run a single job."""

    def __init__(self, trial_settings: TrialSettings, rl_settings: RLSettings):
        """
        Every single job training manager should be initialized with trial_settings and rl_settings in addition to potential override scripts.

        :param trial_settings: Contains settings that apply to any trial.
        :param rl_settings: Contains RL settings.
        """
        self.trial = trial_settings
        self.rl = rl_settings

        mode = self.get_mode()

        # Determine gin bindings
        # Collect all the bindings from gin configurable params in one dictionary
        # Unfortunately I haven't found a way to do this other than manually
        # Only collects bindings that are changed from default
        binders = ["TrialSettings", "RLSettings", "selfplay", "attack", "pbt"]
        bindings = {}
        for binder in binders:
            next_bindings = gin.get_bindings(binder)
            bindings.update(next_bindings)

        # Determine run_name in wandb
        if trial_settings.run_name is None:
            run_name = f"{mode}_{rl_settings.alg}_{rl_settings.env}_{rl_settings.max_timesteps / 1_000_000}Mts"
        else:
            run_name = trial_settings.run_name

        # Whether to disable wandb logging
        wandb_mode = "online"
        if trial_settings.disable_log:
            wandb_mode = "disabled"

        out_path = trial_settings.out_path

        # Init wandb
        wandb_dir = Path(out_path).expanduser().resolve() / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb.init(
            project=trial_settings.wandb_project,
            name=run_name,
            config=bindings,
            mode=wandb_mode,
            group=trial_settings.wandb_group,
            dir=wandb_dir,  # Change wandb dir from the default
            job_type=mode,
            notes=trial_settings.description,
        )

        self.wandb_id = wandb.run.id

        # Init the artifact manager (must be done after wandb init
        # Now that wandb is initialized, update out_path to use a sub directory with the wandb id
        out_path = (
            Path(out_path).expanduser() / trial_settings.wandb_project
        ).resolve() / wandb.run.id
        trial_settings.out_path = out_path
        trial_settings.out_path.mkdir(parents=True, exist_ok=True)

        artifact_manager = ArtifactManager(
            save_remote=not trial_settings.disable_log,
            local_checkpoint_dir=Path(out_path).resolve(),
        )
        artifact_manager.init_saving_checkpoints(
            trial_settings.mode, env_name=rl_settings.env, metadata=bindings
        )
        self.artifact_manager = artifact_manager

        # Init ray
        ray.init(local_mode=trial_settings.ray_local, num_cpus=trial_settings.num_cpus)

        self.checkpoint_freq: int
        # Calculate checkpoint freq
        if (
            self.trial.checkpoint_freq_M is None
            and self.trial.num_checkpoints is not None
        ):
            self.checkpoint_freq = self.rl.max_timesteps // self.trial.num_checkpoints
        elif (
            self.trial.checkpoint_freq_M is not None
            and self.trial.num_checkpoints is None
        ):
            self.checkpoint_freq = int(self.trial.checkpoint_freq_M * 1_000_000)
        elif (
            self.trial.checkpoint_freq_M is None and self.trial.num_checkpoints is None
        ):
            num_checkpoints = 10
            logger.info(
                f"No checkpoint freq given, setting to {num_checkpoints} checkpoints overall"
            )
            self.checkpoint_freq = int(self.rl.max_timesteps / num_checkpoints)
        else:  # checkpoint_freq_M is not None and num_checkpoints is not None:
            raise ValueError(
                "Can't set both checkpoint frequency and number of checkpoints! Choose one."
            )

        # Initialize these later
        self.scheduler = None
        # I couldn't get more strict types to work, this for example was not enough for pytype:
        # Optional[Dict[str, Union[Dict[str, Union[int, None, str]], int, str, list, float]]]
        self.config: Any = None
        self.trainer: Optional[Trainable] = None

        self._init_env()

        self.trainer_class = trainer_cls_from_str(self.rl.alg)

        # Init up seeding
        # Set seed for lib random
        random.seed(a=self.trial.seed)
        # Set seed for numpy
        np.random.seed(self.trial.seed)

    def _init_env(self):
        """Init the env appropriately according to env in rl settings."""
        if self.rl.env.startswith("mpe_"):
            scenario_name = self.rl.env[4:]  # Remove the prefix
            self.env_name = "mpe"
        elif self.rl.env.startswith("gym_"):
            self.env_name = "gym"
            scenario_name = self.rl.env[4:]
        elif self.rl.env.startswith("os_"):
            self.env_name = "open_spiel"
            scenario_name = self.rl.env[3:]  # Remove the prefix
        elif self.rl.env.startswith("pz_"):
            scenario_name = self.rl.env[3:]  # Remove the prefix
            self.env_name = "pettingzoo"
        elif self.rl.env.startswith("mc_"):
            scenario_name = self.rl.env[3:]  # Remove the prefix
            self.env_name = "multicomp"
            # Use scheduler
            SchedulerActor = ray.remote(Scheduler)
            self.scheduler = SchedulerActor.remote()
        else:
            raise ValueError(f"Env {self.rl.env} not supported!")
        self.scenario_name = scenario_name

        self.env = init_env(self.env_name, self.scenario_name, self.scheduler)

    def _log_config(self, config, config_file_name):
        """Log the config to wandb as file."""
        config_as_str = pretty_print(config)
        config_file_path = Path(self.trial.out_path) / config_file_name
        ray_config_file = open(config_file_path, mode="wt")
        ray_config_file.write(config_as_str)
        wandb.save(str(config_file_path))

    def train(self):
        """Main training loop for single job training. Should not be overridden. Instead
         override the parts that are called as needed.
        Performs setup which can't be performed in __init__.
        Sets up config, logs it to wandb, creates trainer and starts training loop."""
        # Config setup
        self.set_up_config()
        self._handle_config_override()

        # Save the ray config in wandb
        Path(self.trial.out_path).mkdir(
            exist_ok=True, parents=True
        )  # Create the out folder
        config_file_name = "ray_config.txt"
        self._log_config(self.config, config_file_name)

        # Handle seeding for tf and torch
        if self.config["framework"] in ["tf", "tf2", "tfe"]:
            import tensorflow as tf

            tf.random.set_seed(self.trial.seed)
        elif self.config["framework"] == "torch":
            import torch

            torch.manual_seed(self.trial.seed)

        # Trainer setup
        self.set_up_trainer()

        # Save config to be used with rllibs's checkpoint loading (rllib expects a
        # pickled config object). Do this after set_up_trainer, sincer there are
        # additional changes that are applied to config during set-up (this should
        # probably be handled more cleanly).
        save_params(self.config, Path(self.trial.out_path))

        # Set trainer for artifact manager, as ArtifactManager needs reference to the trainable
        self.artifact_manager.trainer = self.trainer
        # The previously logged config only contains the values that are different from the default, use this config to see all config params that ray uses
        self._log_config(self.trainer.config, "ray_config_after_init.txt")

        # Start training
        logger.info(f"Starting run, output saved at {self.trial.out_path}")
        # Start the actual training
        self.start_training_loop()

        # This helps multipricessing actually finish a process, it seems without this the processes stays alive?
        wandb.run.finish()

    def set_up_config(self):
        """Config setup for a generic training run."""
        logger.info("Initializing config")
        self.config = get_base_train_config(self.rl.alg)

        # General / trial settings
        self.config["seed"] = self.trial.seed
        self.config["framework"] = self.trial.framework
        self.config["num_workers"] = self.trial.num_workers
        self.config["lr"] = self.trial.lr
        self.config["num_gpus"] = self.trial.num_gpus

        # RL settings
        self.config["train_batch_size"] = self.rl.train_batch_size
        self.config["num_envs_per_worker"] = self.rl.num_envs_per_worker
        self.config["num_sgd_iter"] = self.rl.num_sgd_iter
        if self.rl.horizon is not None:
            self.config["horizon"] = self.rl.horizon
        # These are ppo-only
        if self.rl.alg == "ppo":
            self.config["sgd_minibatch_size"] = self.rl.sgd_minibatch_size

        # horizon= self.config[horizon]
        self.config["rollout_fragment_length"] = (
            self.rl.train_batch_size // self.trial.num_workers
        )
        # MoJoCo only
        if self.env_name == "multicomp":
            # Add mujoco logging callback
            # This callback logs the reward shaping
            self.config["callbacks"] = CustomMujocoMetricsCallbacks

    @abstractmethod
    def set_up_trainer(self) -> None:
        pass

    def generic_trainer_setup(self, policies_to_train):
        """Can be used by subclasses to implement simple trainer setup. Some jobs might
        need more elaborate trainer setup."""

        self.trainer = create_ma_trainer(
            policies_to_train,
            self.config,
            self.scenario_name,
            self.env,
            self.trainer_class,
        )

    @abstractmethod
    def start_training_loop(self):
        """The actual main training loop after all setup is done."""
        pass

    def _handle_config_override(self):
        if self.trial.override is not None:
            # Convert json string into dict
            override_params = json.loads(self.trial.override)
            # Update internal config with new override vals
            self.config.update(override_params)
        for path in self.trial.override_f:
            path: str
            json_file = open(path)
            # Convert json string into dict
            override_params = json.load(json_file)
            # Update internal config with new override vals
            self.config.update(override_params)

    def _run_trainer_helper(
        self,
        trainer,
        checkpoint_freq: int,
        max_timesteps,
        artifact_manager: ArtifactManager,
        log_setup=None,
    ):
        """Simple training loop that will be sufficient for most training managers."""
        next_checkpoint = checkpoint_freq

        pbar = tqdm(total=max_timesteps)

        timesteps_total = 0
        while timesteps_total < max_timesteps:
            results = trainer.train()

            timesteps_total = results["timesteps_total"]

            if self.scheduler is not None:
                frac_remaining = (max_timesteps - timesteps_total) / max_timesteps
                _ = ray.get(
                    self.scheduler.get_val.remote("rew_shape", frac_remaining)
                )  # Update frac remaining for scheduler

            # for policy in trainer.config['multiagent']['policies']:
            if log_setup is not None:
                for policy, name in log_setup:

                    # Check that the metric for this policy was actually collected
                    if policy not in results["policy_reward_mean"]:
                        logger.info(
                            f"No result values collected. If this keeps happening the "
                            f"environment might not provide an end of episode signal "
                            f"and 'horizon' must be set."
                        )
                        pass
                    else:
                        value = results["policy_reward_mean"][policy]
                        wandb.log(
                            {f"{name}_reward": value, "timestep": timesteps_total}
                        )
                if len(results["custom_metrics"]) != 0:
                    log_dict = {"timestep": timesteps_total}
                    log_dict.update(results["custom_metrics"])
                    wandb.log(log_dict)

            custom_eval_log(results, timesteps_total, timesteps_total)

            # Save intermediate checkpoint if necessary
            if checkpoint_freq != -1 and timesteps_total > next_checkpoint:
                artifact_manager.save_new_checkpoint()

                next_checkpoint += (
                    checkpoint_freq  # We save the next checkpoint over this threshold
                )

            pbar.update(timesteps_total - pbar.n)

        pbar.close()

        # Save last checkpoint at the end
        artifact_manager.save_new_checkpoint()
        logger.info("checkpoint saved")
        return timesteps_total

    def get_mode(self):
        """Return the mode of the training manager. This is a separate function in
        order to allow this to be called in the init of the base class."""
        return self.trial.mode


def create_ma_trainer(policies_to_train, config, scenario_name, env, trainer_cls):
    """Create basic 2 agent trainer with new weights"""
    multiagent = aprl_defense.common.utils.generate_multiagent_2_policies(
        env, policies_to_train
    )
    if "env_config" not in config:
        config["env_config"] = {}
    config["env_config"]["scenario_name"] = scenario_name
    config["multiagent"] = multiagent
    # Update config with some settings that apply to all configs regardless of
    # algorithm/trainer
    # Create a new trainer based on class.
    # Use noop_logger_creator to surpress ray logs that get saved by default.
    new_trainer = trainer_cls(
        env="current-env",
        config=deepcopy(config),  # , logger_creator=noop_logger_creator
    )
    return new_trainer
