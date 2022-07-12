import os
import random
import time

import gin
from absl import app
from absl import flags
from dotenv import load_dotenv

from aprl_defense.common.base_logger import logger
from aprl_defense.training_managers.base_training_manager import (
    BaseTrainingManager,
)
from aprl_defense.training_managers.pbt_manager import PBTManager
from aprl_defense.training_managers.pbt_train_and_attack_manager import (
    PBTTrainAndAttackManager,
)
from aprl_defense.training_managers.simple_training_manager import (
    TwoPolicySelfplayTrainingManager,
    SingleAgentTrainingManager,
    SinglePolicySelfplayTrainingManager,
)
from aprl_defense.training_managers.attack_manager import AttackManager
from aprl_defense.trial.settings import TrialSettings, RLSettings

# The supported flags:
# -f to provide a gin config
# -p to overwrite gin parameters
flags.DEFINE_multi_string("f", None, "List of paths to the config files.")
flags.DEFINE_multi_string(
    "p", None, "Newline separated list of Gin parameter bindings."
)

FLAGS = flags.FLAGS


def main(argv):
    gin.parse_config_files_and_bindings(FLAGS.f, FLAGS.p)

    # Determine out_path and update trial_settings if necessary
    # Default is overwritten by cli param, is overwritten by environment variable
    # Env var is overwritten by dotenv file
    load_dotenv()
    out_path = os.environ.get("POLICY_DEFENSE_OUT", None)
    if out_path is None:
        trial_settings = TrialSettings()
    else:
        trial_settings = TrialSettings(
            out_path=out_path
        )  # Explicitly passing out_path overwrites both gin and default path
    rl_settings = RLSettings()

    if trial_settings.framework == 'tf2' or trial_settings.framework == 'tfe':
        import tensorflow as tf
        tf.compat.v1.enable_eager_execution()

    # Determine seed
    if trial_settings.seed is None:
        trial_settings.seed = random.randrange(
            2**32 - 1
        )  # Numpy has this as e max number for seed

    mode = trial_settings.mode

    training_manager: BaseTrainingManager
    if mode == "selfplay":
        training_manager = SinglePolicySelfplayTrainingManager(
            trial_settings,
            rl_settings,
        )
    elif mode == "two_policy_selfplay":
        training_manager = TwoPolicySelfplayTrainingManager(
            trial_settings,
            rl_settings,
        )
    elif mode == "single-agent":
        training_manager = SingleAgentTrainingManager(trial_settings, rl_settings)
    elif mode == "attack":
        training_manager = AttackManager(trial_settings, rl_settings)
    elif mode == "pbt":
        training_manager = PBTManager(trial_settings, rl_settings)
    elif mode == "pbt+attack":
        training_manager = PBTTrainAndAttackManager(
            trial_settings,
            rl_settings,
            num_ops_list=gin.REQUIRED,
            num_training=gin.REQUIRED,
            num_attacks=gin.REQUIRED,
            num_processes=gin.REQUIRED,
        )
    else:  # Anything else is unsupported
        raise ValueError(f"Illegal argument for mode: {mode}")

    logger.info(f"Training with mode {mode}")
    time_start = time.time()
    training_manager.train()
    logger.info(f"Training of took {(time.time() - time_start) / 60} minutes")


if __name__ == "__main__":
    app.run(main)
