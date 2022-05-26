from typing import Optional

import gin
from copy import deepcopy

import aprl_defense
from aprl_defense.common.base_logger import logger
from aprl_defense.common.utils import load_saved_weights, load_saved_checkpoint
from aprl_defense.training_managers.base_training_manager import (
    SingleJobTrainingManager,
)
from aprl_defense.trial.settings import TrialSettings, RLSettings


@gin.configurable(name_or_fn="attack")
class AttackManager(SingleJobTrainingManager):
    """Training manager for adversarial policy attack."""

    def __init__(
        self,
        trial_settings: TrialSettings,
        rl_settings: RLSettings,
        victim_artifact: str,
        adversary_id: int = 1,
        victim_policy_name: Optional[str] = None,
        victim_config_setting: str = "use_saved",
    ):
        """
        Create an attack manager against a specific victim.

        :param trial_settings:
        :param rl_settings:
        :param override:
        :param override_f:
        :param victim_artifact: wandb artifact id with version of saved victim policy.
        :param adversary_id: Agent id to use for training adversary. Victim used is the other agent.
        :param victim_policy_name: In case the name of the policy saved in the checkpoint is not simply 'policy_<victim_id>', provide the name of the victim
            policy here.
        """
        super().__init__(trial_settings, rl_settings)

        self.victim_artifact = victim_artifact
        self.adversary_id = adversary_id

        self.victim_id = 1 - adversary_id

        self.victim_config_setting = victim_config_setting

        if victim_policy_name is None:
            self.victim_policy_name = f"policy_{self.victim_id}"
            logger.info(
                f"Using {self.victim_policy_name} as automatic fallback because no victim policy name was provided"
            )
        else:
            self.victim_policy_name = victim_policy_name

    def start_training_loop(self):
        """Load victim and start normal training against it."""
        trainer = self.trainer
        adv_name = f"policy_{self.adversary_id}"
        new_victim_name = f"policy_{self.victim_id}"

        file = self.artifact_manager.get_remote_checkpoint(self.victim_artifact)

        # This all is a very ad-hoc workaround, which is necessary because some saved
        # configs were not saved correctly in old versions of the code. This allows
        # weights from these old versions to still be loaded. In future using solely
        # 'use_saved' should be sufficient. These other options could be removed at
        # that point.
        if self.victim_config_setting == "use_parent":
            # Using self.config here is not appropriate: The victim would have been
            # trained with a different config than the attacker. However, since we are
            # only interested in loading the victim's weights, the config doesn't
            # matter. RLlib simply requires one, however it does not affect the weights.
            # Using the victims config instead (by omitting the config argument) would
            # be preferable, however due to a bug in my code, saved pickled configs were
            # not saved correctly.
            config = self.config
        elif self.victim_config_setting == "use_saved":
            # This will load the config that was saved with the victim's checkpoint.
            # This is preferable, assuming the saved pkl config isn't corrupted.
            config = None
        elif self.victim_config_setting == "use_single_policy":
            config = deepcopy(self.config)
            multiagent = aprl_defense.common.utils.generate_multiagent_shared_policy()
            if "env_config" not in config:
                config["env_config"] = {}
            config["env_config"]["scenario_name"] = self.scenario_name
            config["multiagent"] = multiagent
        else:
            raise ValueError(f"Unknown config setting {self.victim_config_setting}")
        victim_weights = load_saved_weights(
            file,
            self.scenario_name,
            self.victim_policy_name,
            self.trainer_class,
            config,
        )

        if self.victim_policy_name not in victim_weights:
            raise ValueError(
                f"No policy named {self.victim_policy_name} among victim weights. Please provide the correct name of the victim policy."
            )

        # Overwrite the weights for the agent that will not be trained, which is the victim.
        trainer.set_weights({new_victim_name: victim_weights[self.victim_policy_name]})

        trainer.workers.sync_weights()  # IMPORTANT!!1!!1

        log_setup = [(adv_name, "adversary")]
        self._run_trainer_helper(
            trainer,
            self.checkpoint_freq,
            self.rl.max_timesteps,
            artifact_manager=self.artifact_manager,
            log_setup=log_setup,
        )

    def set_up_trainer(self):
        """Generic trainer setup where only attacker is trained."""
        if self.trial.continue_artifact is None:
            # Create a new trainer
            policies_to_train = [f"policy_{self.adversary_id}"]
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
        return "attack"
