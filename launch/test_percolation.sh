#!/bin/bash
IDS=("baseline" "incremental" "bidirectional" "dfs" "reverse")
P_VALUES=(0)

for ID in "${IDS[@]}"; do
    echo "Processing ID: ${ID}"
    MODEL_PATH="outputs/training/training-${ID}"

    for P in "${P_VALUES[@]}"; do
        RUN_ID="${ID}-p-${P}"
        echo "Running testing with run_id=${RUN_ID}"
        CUDA_VISIBLE_DEVICES=0 python test_model.py problem.model.model_path=${MODEL_PATH} \
            name=testing-${ID}-percolation plot_outputs=true problem.test_data=9 \
            problem.hyp.test_mode=stable problem.model.test_iterations.high=200 \
            problem.dataset=maze-dataset problem.percolation=${P} problem.deadend_start=False +run_id=${RUN_ID}
    done
done