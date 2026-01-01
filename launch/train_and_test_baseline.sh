#!/bin/bash

MODE="baseline"
SIZES=(9 13 19 25 33 59)
RUN=1
DEVICE=1

echo "=== TRAINING PHASE ==="

TRAIN_RUN_ID="${MODE}-${RUN}"
echo "Training run ${RUN} ${TRAIN_RUN_ID}"
CUDA_VISIBLE_DEVICES=${DEVICE} python train_model.py \
    problem=mazes \
    problem/model=dt_net_recall_2d \
    problem.train_data=9 problem.test_data=9 \
    problem.hyp.train_mode=default \
    name=training \
    problem.hyp.mazesolver_mode=${MODE} \
    problem.hyp.step=3 \
    +run_id=${TRAIN_RUN_ID}


echo "=== TESTING PHASE ==="

MODEL_PATH="outputs/training/training-${MODE}-${RUN}"

if [ ! -f "${MODEL_PATH}/model_best.pth" ]; then
    echo "Model not found: ${MODEL_PATH}/model_best.pth, skipping run ${RUN}"
    continue
fi

echo "Testing model ${RUN}/${NUM_RUNS}: ${MODEL_PATH}"

for SIZE in "${SIZES[@]}"; do
    TEST_RUN_ID="${MODE}-${SIZE}x${SIZE}-${RUN}"
    echo "  Testing on size ${SIZE}x${SIZE}, run_id=${TEST_RUN_ID}"
    
    CUDA_VISIBLE_DEVICES=${DEVICE} python test_model.py \
        problem.model.model_path=${MODEL_PATH} \
        name=testing-${MODE} \
        plot_outputs=false \
        problem.test_data=${SIZE} \
        problem.hyp.test_mode=stable \
        problem.model.test_iterations.high=1000 \
        problem.dataset=easy-to-hard-data \
        +run_id=${TEST_RUN_ID}
done

echo "Training and testing complete"