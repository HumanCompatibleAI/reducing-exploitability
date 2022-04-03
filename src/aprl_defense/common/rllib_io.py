from pathlib import Path

from ray import cloudpickle

from aprl_defense.common.io import get_saved_config
from aprl_defense.common.utils import create_trainer


def save_params(config, log_dir: Path):
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    # Save params file
    with (checkpoint_dir / "params.pkl").open(mode="wb") as file:
        cloudpickle.dump(config, file=file)
    return checkpoint_dir


def restore_trainer_from_path(
    path: Path, scenario_name: str, trainer_cls, victim_config=None, config_update=None
):
    if (
        victim_config is None
    ):  # Load victim config if not provided, otherwise the provided one is used
        victim_config = get_saved_config(path)

    # Changes to victim config for finetuning
    victim_config["env"] = "current-env"
    victim_config["env_config"]["scenario_name"] = scenario_name

    if config_update is not None:
        victim_config.update(config_update)

    # Restore the trained agent with this trainer
    trainer = create_trainer(trainer_cls, victim_config)
    trainer.restore(checkpoint_path=str(path))
    return trainer
