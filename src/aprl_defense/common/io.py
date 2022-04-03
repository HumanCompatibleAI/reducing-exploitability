from pathlib import Path
from typing import Union

from ray.cloudpickle import cloudpickle


def get_saved_config(path):
    config_path = Path(path).parent / ".." / "params.pkl"
    config_path = config_path.resolve()
    with open(config_path, "rb") as f:
        loaded_config = cloudpickle.load(f)
    return loaded_config


def get_checkpoint_file(path: Union[str, Path], specific_folder=False) -> Path:
    """For a path to a checkpoint directory return the path the sub-file of the newest checkpoint"""
    if specific_folder:  # A specific folder with checkpoint files is provided
        folder = Path(path)
    else:  # Get the newest subfolder in the checkpoints folder
        checkpoint_folders = Path(path) / "checkpoints"
        checkpoints = [
            child.name for child in checkpoint_folders.iterdir() if child.is_dir()
        ]
        checkpoints = sorted(
            checkpoints
        )  # If we sort the checkpoints by name, they are in training order
        checkpoint_file = None
        # Get newest checkpoint
        newest = checkpoints[-1]

        folder = checkpoint_folders / newest

    for file in folder.iterdir():
        if not file.suffix == ".tune_metadata" and not file.name == ".is_checkpoint":
            checkpoint_file = file
    # Some sanity checks
    if checkpoint_file is None or not checkpoint_file.is_file():
        raise ValueError("Something is wrong with the checkpoint structure")
    return checkpoint_file
