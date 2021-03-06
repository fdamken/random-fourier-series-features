#!/usr/bin/env bash

#SBATCH --job-name rfsf
#SBATCH --array 0-19
#SBATCH --time 06:00:00
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 4096
#SBATCH --gres gpu:a100
#SBATCH -o /home/fd15hava/logs/%A_%a-out.txt
#SBATCH -e /home/fd15hava/logs/%A_%a-err.txt
#SBATCH --mail-type ALL

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

export TMPDIR=$HPC_SCRATCH/tmp
export PYTHONPATH="$working_dir/src"
export NO_WANDB=y
cmd="python src/experiments/experiment.py with $model_name dataset.name=$dataset_name dataset.split_index=$dataset_split_index dataset.double_precision=True"
echo "Running '$cmd' with 'TMPDIR=$TMPDIR', 'PYTHONPATH=$PYTHONPATH', and 'NO_WANDB=$NO_WANDB'."
$cmd
