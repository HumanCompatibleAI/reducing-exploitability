import os

num_ops_list = [1, 2, 4, 8]  # train 16 ops separately
# env_name, timestep_M
envs = [("mpe_simple_push_comm_50",)]
num_seeds = 5
num_parallel = 5
num_workers = 10
group = "icml-pbt-push-v1"
description = "aws3"
num_gpus = 0

counter = 0
for seed in range(num_seeds):
    for num_ops in num_ops_list:
        for (env,) in envs:
            for main in [0, 1]:
                if num_ops < num_workers:
                    num_workers = num_ops

                if counter == num_parallel - 1:
                    parallel = ""
                    counter = 0
                else:
                    parallel = "&"

                train_batch_size = 5000 * num_ops

                os.system(
                    "python -m aprl_defense.train "
                    '-f "gin/icml/pbt/simple_push_50.gin" '
                    f"-p \"TrialSettings.wandb_group = '{group}'\" "
                    f"-p \"RLSettings.env = '{env}'\" "
                    f'-p "pbt.main_id = {main}" '
                    f'-p "pbt.num_ops = {num_ops}" '
                    f'-p "TrialSettings.num_workers = {num_workers}" '
                    f'-p "TrialSettings.num_gpus = {num_gpus}" '
                    f"-p \"TrialSettings.description = '{description}'\" "
                    f'-p "RLSettings.train_batch_size = {train_batch_size}" '
                    f"{parallel}"
                )

                counter += 1
