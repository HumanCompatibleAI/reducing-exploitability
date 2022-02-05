#!/bin/bash
#SBATCH --job-name=reproduce-systematic
#SBATCH --account=fc_chai
#SBATCH --partition=savio2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=22
#SBATCH --time=16:00:00
## Command(s) to run:
module load python
# module load ml

source /global/home/users/${USER}/.bashrc
# source activate /global/home/users/${USER}/.conda/envs/defense/
source activate /global/scratch/users/${USER}/conda_envs/ray-source-2

export CUDA_VISIBLE_DEVICES=-1
# When using savio this helps keep the home directory small. The cache is where e.g. remote artifacts are downloaded to
export WANDB_CACHE_DIR=/global/scratch/users/${USER}/wandb_cache

python -V
python -m aprl_defense.train --alg ppo --wandb-group $SLURM_JOB_NAME --policy-cache /global/scratch/users/${USER}/policy_cache/$SLURM_JOBID --mode single-trainer-pbt --num-workers 15 --checkpoint-freq-M 2 --max-timesteps 1000000 --out-path /global/scratch/users/${USER}/out/rllib --env os_laser_tag_zs --harden-id 0 --num-ops 75 --override "{\"framework\":\"tf\"}" --description "commit 8 a0dcd6: main + fix"
