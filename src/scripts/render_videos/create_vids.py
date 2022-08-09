import os
from pathlib import Path

# path = "/home/pavel/code/out/chai/models/normal/01/"
path = "/home/pavel/out/code/chai/tmp/24ir4s89/artifacts"
vid_dir = "/home/pavel/vids/new/"

checkpoint_folders = Path(path) / "checkpoints"
checkpoints = [child for child in checkpoint_folders.iterdir() if child.is_dir()]
checkpoints = sorted(
    checkpoints
)  # If we sort the checkpoints by name, they are in training order

i = 0
for checkpoint in checkpoints:
    for checkpoint_full in checkpoint.iterdir():
        if (
            not checkpoint_full.suffix == ".tune_metadata"
            and not checkpoint_full.name == ".is_checkpoint"
        ):
            os.system(
                f"python -m aprl_defense.evaluate {checkpoint_full} --run PPO --env mpe --video-dir {vid_dir}{i:03d}"
                +
                # num_workers: If this is not set low enough, ray will complain that there are not enough resources, even though the resources for
                # training are never actually used
                ' --config {\\"num_workers\\":0,\\"evaluation_num_workers\\":0,\\"custom_eval_function\\":null,\\"evaluation_duration\\":1} '
                "--episodes 0 --steps 25"
            )
            i += 1

#
