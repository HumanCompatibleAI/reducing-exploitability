#!/bin/bash
#SBATCH --job-name=mujoco-normal-training
#SBATCH --account=fc_chai
#SBATCH --partition=savio2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=22
#SBATCH --time=72:00:00
## Command(s) to run:
module load python
# module load ml

source /global/home/users/${USER}/.bashrc
source activate /global/home/users/${USER}/.conda/envs/defense/

source activate /global/scratch/users/${USER}/conda_envs/ray-source-2

export CUDA_VISIBLE_DEVICES=-1
export WANDB_CACHE_DIR=/global/scratch/users/${USER}/wandb_cache
#export MUJOCO_PY_MJPRO_PATH="/global/scratch/users/${USER}/mjpro131"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/home/users/pavelczempin/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

python -V
python -m aprl_defense.train --alg ppo --wandb-group $SLURM_JOB_NAME --mode normal --num-workers 40 --num-cpus 50 --checkpoint-freq-M 5 --eval-freq-M 10000 --max-timesteps 10000000000 --out-path /global/scratch/users/pavelczempin/out/rllib --env mc_SumoAnts-v0 --override "{\"framework\":\"tf\"}"
