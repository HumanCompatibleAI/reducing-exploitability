import os
from pathlib import Path

# path = "/home/pavel/code/out/chai/models/normal/01/"
import wandb

from aprl_defense.common.artifact_manager import ArtifactManager

artifact = "1cvttgnf:latest"
vid_dir = "/home/pavel/vids/new/"
# vid_dir = None
local_mode = False
project = "pbrl-defense"

artifact_dir = Path("/home/pavel/out/code/chai/tmp")

wandb.init(project=project, group="eval", dir=artifact_dir)

artifact_manager = ArtifactManager(
    save_remote=False,
    local_checkpoint_dir=artifact_dir.resolve() / wandb.run.id,
)

file = artifact_manager.get_remote_checkpoint(artifact)

vid_command = "--render" if vid_dir is None else f"--video-dir {vid_dir}"

os.system(
    f"python -m aprl_defense.evaluate {file} --run PPO --env mpe {vid_command}" +
    # num_workers: If this is not set low enough, ray will complain that there are not enough resources, even though the resources for
    # training are never actually used
    ' --config {\\"num_workers\\":0,\\"evaluation_num_workers\\":0,\\"custom_eval_function\\":null,\\"evaluation_duration\\":1} '
    "--episodes 0 --steps 25"
)
