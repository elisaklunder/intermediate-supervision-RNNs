MODEL_PATH="outputs/training/training-baseline"
ID="baseline-test"
SIZES=(9)

for SIZE in "${SIZES[@]}"; do
  RUN_ID="${ID}-${SIZE}x${SIZE}"
  echo "Running testing with run_id=${RUN_ID}"
  CUDA_VISIBLE_DEVICES=0 python test_model.py problem.model.model_path=${MODEL_PATH} \
      name=testing-${ID} plot_outputs=true problem.test_data=${SIZE} \
      problem.hyp.test_mode=stable problem.model.test_iterations.high=1000 \
      problem.dataset=easy-to-hard-data +run_id=${RUN_ID}
done
