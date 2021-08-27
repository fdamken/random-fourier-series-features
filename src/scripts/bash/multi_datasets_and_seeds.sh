#!/bin/bash

if [[ ! -d src ]]; then
    echo "E: Not executing from the repository root." >&2
    exit 1
fi

datasets=$1
shift
num_eval=$1
shift

echo "Activating virtualenv."
source venv/bin/activate

echo "Starting training."
for train_test_split_seed in $(seq 1 $num_eval); do
    for dataset in $datasets; do
        echo "Evaluating dataset $dataset, split seed $train_test_split_seed"
        PYTHONPATH=src python src/experiments/experiment.py with dataset.name=$dataset dataset.train_test_split_seed=$train_test_split_seed $*
    done
done
