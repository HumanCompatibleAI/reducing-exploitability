from pathlib import Path
from typing import Optional

import wandb
from ray.tune import Trainable

from aprl_defense.common.io import get_checkpoint_file


class ArtifactManager:
    """ArtifactManager manages receiving and saving all data such as models.
    The manager saves checkpoints using RLlib's Trainable.save() and optionally logs to wandb.
    Can also be used to load artifacts from wandb."""

    def __init__(
        self,
        save_remote: bool,
        local_checkpoint_dir: Path,
        save_locally: bool = True,
    ):
        """
        If not only loading artifacts, use init_saving_checkpoints() before training to set up the saving of checkpoints.

        :param save_remote: Whether to also log checkpoints to wandb
        :param local_checkpoint_dir: Local dir for temporary checkpoints
        :param save_locally: Whether to also save the checkpoints locally
        """

        # Arguments
        self.save_remote: bool = save_remote
        self.type: Optional[str] = None
        self.name: str = wandb.run.id
        self.metadata: Optional[dict] = None
        self.local_checkpoint_dir: Path = local_checkpoint_dir
        self.save_locally = save_locally

        # These will be set later
        self.trainer: Optional[Trainable] = None

        self.saving_checkpoints_possible = False

    def init_saving_checkpoints(self, mode: str, env_name: str, metadata: dict) -> None:
        """
        Initialize saving checkpoints in wandb. Call this before training if saving checkpoints is necessary.
        :param mode: Name of the training mode. Used to distinguish type of artifact in wandb.
        :param env_name: Name of the environment. Used to distinguish type of artifact in wandb.
        :param metadata: Metadata to attach to the checkpoint.
        :return:
        """
        self.type = f"{mode}_{env_name}"
        self.metadata: dict = metadata
        self.saving_checkpoints_possible = True

    def save_new_checkpoint(self) -> None:
        """
        Save a new checkpoint file according to the params of this manager. If self.save_remote, also log the resulting checkpoint files as an Artifact in wandb
        Attention: self.trainer and self.local_checkpoint_dir must be set, otherwise an exception is thrown.
        """
        # Check edge cases
        if self.trainer is None:
            raise ValueError(
                "self.trainer must be set to appropriate trainable before logging checkpoints!"
            )
        if not self.saving_checkpoints_possible:
            raise ValueError(
                "Can't save checkpoints! init_saving_checkpoints must be used first!"
            )

        # We can only disable saving locally, if neither local nor remote saves are wished, as remote saves also require the local files to be generated
        if self.save_locally or self.save_remote:
            # Log checkpoint locally
            current_checkpoint_file = self.trainer.save(
                checkpoint_dir=str(self.local_checkpoint_dir)
            )

            if self.save_remote:
                # Determine folder of new checkpoint
                current_checkpoint_dir = Path(current_checkpoint_file).parent

                # Create the artifact that will be logged
                # On subsequent calls with the same artifact name and type, wandb will log them as new versions of the artifact
                artifact = wandb.Artifact(
                    name=self.name, type=self.type, metadata=self.metadata
                )
                # Every artifact also need to contain the params, such that it can be loaded
                artifact.add_file(
                    str(self.local_checkpoint_dir / "checkpoints" / "params.pkl")
                )
                # This contains the actual checkpoint files
                artifact.add_dir(
                    str(current_checkpoint_dir), name=current_checkpoint_dir.name
                )

                wandb.log_artifact(artifact)

    def get_remote_checkpoint(self, artifact_identifier) -> Path:
        """
        Download the remote checkpoint given by the identifer and return Path to the checkpoint file.
        :param artifact_identifier: Identifier of the artifact to download.
        :return: Path to the downloaded checkpoint folder containing checkpoint files.
        """
        artifact: wandb.Artifact = wandb.use_artifact(artifact_identifier)

        # Download the artifact's contents
        artifact_dir = Path(
            artifact.download(
                root=str(self.local_checkpoint_dir / "artifacts" / "checkpoints")
            )
        )

        path = get_checkpoint_file(artifact_dir.parent, specific_folder=False)

        return path
