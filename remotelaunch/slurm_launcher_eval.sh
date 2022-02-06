#!/usr/bin/env bash

#SBATCH --job-name rfsf_eval
#SBATCH --array 1-945
#SBATCH --time 10:00
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

run_index="$SLURM_ARRAY_JOB_ID"

working_dir="$HOME/rfsf"
cd "$working_dir"
source venv/bin/activate

run_id="$(ls -1 data/temp/results/slurm | head -n $run_index | tail -n 1)"

echo "Evaluating run $run_id."

export TMPDIR=$HPC_SCRATCH/tmp
export PYTHONPATH="$working_dir/src"
export NO_WANDB=y
cmd="python src/scripts/evaluation/eval_rmse_and_avg_ll.py -d slurm/$run_id/1"
echo "Running '$cmd' with 'TMPDIR=$TMPDIR', 'PYTHONPATH=$PYTHONPATH', and 'NO_WANDB=$NO_WANDB'."
$cmd
