#!/usr/bin/env bash

#SBATCH --job-name run_all_seeds
#SBATCH --array 0-19
#SBATCH --time 06:00
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 4096
#SBATCH --gres gpu
#SBATCH -o /home/fd15hava/logs/%A_%a-out.txt
#SBATCH -e /home/fd15hava/logs/%A_%a-err.txt

set -o errexit
set -o nounset

echo "Starting Job $SLURM_JOB_ID; Array Job $SLURM_ARRAY_JOB_ID Index $SLURM_ARRAY_TASK_ID."

model_name="$1"
dataset_name="$2"
dataset_split_index="$SLURM_ARRAY_TASK_ID"

echo "Training model $model_name on $dataset_name dataset with test/train split $dataset_split_index."

working_dir="$HOME/rfsf"
cd "$working_dir"
source venv/bin/activate

cmd="python src/experiments/experiment.py with $model_name dataset.name=$dataset_name dataset.split_index=$dataset_split_index double_precision=True"
echo "Running '$cmd'."
PYTHONPATH="$working_dir/src" NO_WANDB=y $cmd
