from copy import deepcopy

import gin

import aprl_defense
from aprl_defense.common.base_logger import logger
from aprl_defense.common.utils import (
    noop_logger_creator,
    load_saved_checkpoint,
)
from aprl_defense.training_managers.base_training_manager import (
    SingleJobTrainingManager,
)
from aprl_defense.trial.settings import TrialSettings, RLSettings


@gin.configurable(name_or_fn="two_policy_selfplay")
class TwoPolicySelfplayTrainingManager(SingleJobTrainingManager):
    """Training manager for selfplay with separate policies for each agent."""

    def __init__(
        self,
        trial_settings: TrialSettings,
        rl_settings: RLSettings,
        train_only_one_policy: bool = False,
    ):
        """
        :param trial_settings:
        :param rl_settings:
        :param override:
        :param override_f:
        :param train_only_one_policy: Settings this to true will disable training for the second agent. For debugging / sanity checks purposes.
        """
        super().__init__(trial_settings, rl_settings)
        self.train_only_one_policy = train_only_one_policy

    def start_training_loop(self):
        """Default training loop with policy for each agent."""
        log_setup = [("policy_0", "policy_0"), ("policy_1", "policy_1")]
        try:
            self._run_trainer_helper(
                self.trainer,
                self.checkpoint_freq,
                self.rl.max_timesteps,
                artifact_manager=self.artifact_manager,
                log_setup=log_setup,
            )
        except Exception as e:
            logger.error(
                f"Encountered Exception will try to save checkpoint before "
                "re-raising"
            )
            self.artifact_manager.save_new_checkpoint()
            raise e

    def set_up_trainer(self):
        """Set up the trainer with possible continuation from checkpoint."""
        if self.trial.continue_artifact is None:
            if self.train_only_one_policy:
                policies_to_train = ["policy_1"]
            else:
                policies_to_train = None  # None == all policies train
            self.generic_trainer_setup(policies_to_train)
        else:
            # Use saved trainer from checkpoint
            file = self.artifact_manager.get_remote_checkpoint(
                self.trial.continue_artifact
            )
            if self.trial.override_config:
                config = self.config
            else:
                config = None
            self.trainer = load_saved_checkpoint(self.trainer_class, file, config)

    def get_mode(self):
        return "two_policy_selfplay"


@gin.configurable(name_or_fn="selfplay")
class SinglePolicySelfplayTrainingManager(SingleJobTrainingManager):
    """Training manager for selfplay with separate policies for each agent."""

    def __init__(
        self,
        trial_settings: TrialSettings,
        rl_settings: RLSettings,
    ):
        """
        :param trial_settings:
        :param rl_settings:
        :param override:
        :param override_f:
        :param train_only_one_policy: Settings this to true will disable training for the second agent. For debugging / sanity checks purposes.
        """
        super().__init__(trial_settings, rl_settings)

    def start_training_loop(self):
        """Default training loop with policy for each agent."""
        log_setup = [("shared_policy", "shared_policy")]
        try:
            self._run_trainer_helper(
                self.trainer,
                self.checkpoint_freq,
                self.rl.max_timesteps,
                artifact_manager=self.artifact_manager,
                log_setup=log_setup,
            )
        except Exception as e:
            logger.error(
                f"Encountered Exception will try to save checkpoint before "
                "re-raising"
            )
            self.artifact_manager.save_new_checkpoint()
            raise e

    def set_up_trainer(self):
        """Set up the trainer with possible continuation from checkpoint."""
        if self.trial.continue_artifact is None:
            policies_to_train = None  # None == all policies train

            multiagent = aprl_defense.common.utils.generate_multiagent_shared_policy()
            if "env_config" not in self.config:
                self.config["env_config"] = {}
            self.config["env_config"]["scenario_name"] = self.scenario_name
            self.config["multiagent"] = multiagent
            # Update config with some settings that apply to all configs regardless of
            # algorithm/trainer
            # Create a new trainer based on class.
            # Use noop_logger_creator to surpress ray logs that get saved by default.
            self.trainer = self.trainer_class(
                env="current-env",
                config=deepcopy(self.config),
                # logger_creator=noop_logger_creator,
            )
        else:
            # Use saved trainer from checkpoint
            file = self.artifact_manager.get_remote_checkpoint(
                self.trial.continue_artifact
            )
            if self.trial.override_config:
                config = self.config
            else:
                config = None
            self.trainer = load_saved_checkpoint(self.trainer_class, file, config)

    def get_mode(self):
        return "selfplay"


class SingleAgentTrainingManager(SingleJobTrainingManager):
    """Training manager for single agent training. Should support all normal gym
    single-agent envs."""

    def __init__(
        self,
        trial_settings: TrialSettings,
        rl_settings: RLSettings,
    ):
        """
        :param trial_settings:
        :param rl_settings:
        :param override:
        :param override_f:
        """
        super().__init__(trial_settings, rl_settings)

    def start_training_loop(self):
        """Normal training loop with only one agent."""
        log_setup = [("policy_0", "policy_0")]
        self._run_trainer_helper(
            self.trainer,
            self.checkpoint_freq,
            self.rl.max_timesteps,
            artifact_manager=self.artifact_manager,
            log_setup=log_setup,
        )

    def set_up_trainer(self):
        """Set up a trainer without multi-agent."""
        if self.trial.continue_artifact is None:
            # This general setup only allows one type of algorithm
            if len(self.trainer_class) > 1:
                raise ValueError("Only one algorithm allowed.")
            self.config["env_config"]["scenario_name"] = self.scenario_name
            # Create a new maddpg trainer which will be used to train the adversarial policy
            self.trainer = self.trainer_class(
                env="current-env",
                config=deepcopy(self.config),
                logger_creator=noop_logger_creator,
            )
        else:
            # Use saved trainer from checkpoint
            file = self.artifact_manager.get_remote_checkpoint(
                self.trial.continue_artifact
            )
            if self.trial.override_config:
                config = self.config
            else:
                config = None
            self.trainer = load_saved_checkpoint(self.trainer_class, file, config)

    def get_mode(self):
        return "single-agent"
