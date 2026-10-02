#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
COLLECT_SCRIPT="${REPO_ROOT}/src/pisr/collect_and_train_from_api.jl"

RUNS="${RUNS:-10}"
MASTER_SEED="${MASTER_SEED:-4242}"
SERVICE_URL="${SERVICE_URL:-${PF_BASE_URL:-http://127.0.0.1:8012}}"

TRAIN_SAMPLES="${TRAIN_SAMPLES:-50}"
TEST_SAMPLES="${TEST_SAMPLES:-30}"
POLL_MS="${POLL_MS:-1000}"
SETTLE_MS="${SETTLE_MS:-150}"
SLACK_BUS_ID="${SLACK_BUS_ID:-66}"

INITIAL_ITER="${INITIAL_ITER:-20}"
STEP_ITER="${STEP_ITER:-100}"
MAX_TOTAL_ITER="${MAX_TOTAL_ITER:-1020}"
THRESHOLD="${THRESHOLD:-0.001}"
MAX_SIZE="${MAX_SIZE:-30}"
NP_FACTOR="${NP_FACTOR:-3}"
MULTIPROCESS="${MULTIPROCESS:-false}"
TRAIN_DP_MODE="${TRAIN_DP_MODE:-all}"
TRAIN_DQ_MODE="${TRAIN_DQ_MODE:-all}"

TRAIN_DP_MAX_LO="${TRAIN_DP_MAX_LO:-0.015}"
TRAIN_DP_MAX_HI="${TRAIN_DP_MAX_HI:-0.080}"
TRAIN_DQ_MAX_LO="${TRAIN_DQ_MAX_LO:-0.015}"
TRAIN_DQ_MAX_HI="${TRAIN_DQ_MAX_HI:-0.080}"
TEST_DP_MAX_LO="${TEST_DP_MAX_LO:-0.015}"
TEST_DP_MAX_HI="${TEST_DP_MAX_HI:-0.080}"
TEST_DQ_MAX_LO="${TEST_DQ_MAX_LO:-0.015}"
TEST_DQ_MAX_HI="${TEST_DQ_MAX_HI:-0.080}"

TRAIN_DP_SAMPLES_LO="${TRAIN_DP_SAMPLES_LO:-10}"
TRAIN_DP_SAMPLES_HI="${TRAIN_DP_SAMPLES_HI:-${TRAIN_SAMPLES}}"
TRAIN_DQ_SAMPLES_LO="${TRAIN_DQ_SAMPLES_LO:-10}"
TRAIN_DQ_SAMPLES_HI="${TRAIN_DQ_SAMPLES_HI:-${TRAIN_SAMPLES}}"

RANDOMIZE_START_ROW="${RANDOMIZE_START_ROW:-true}"
START_ROW_BASE="${START_ROW_BASE:-0}"
START_ROW_MODE="${START_ROW_MODE:-set_and_release}"   # set_and_release | pin | off
START_ROW_SETTLE_MS="${START_ROW_SETTLE_MS:-200}"

BATCH_NAME="${BATCH_NAME:-multi_collect_train}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MAIN_DIR="${MAIN_DIR:-${REPO_ROOT}/results/V-B_training_robustness_fig8/${BATCH_NAME}_${TIMESTAMP}}"
RUNS_DIR="${MAIN_DIR}/runs"
DATASETS_DIR="${MAIN_DIR}/datasets"
MODELS_DIR="${MAIN_DIR}/models"
MANIFEST_CSV="${MAIN_DIR}/manifest.csv"

if [[ ! -f "$COLLECT_SCRIPT" ]]; then
  echo "Missing collect script: $COLLECT_SCRIPT" >&2
  exit 1
fi
if ! command -v julia >/dev/null 2>&1; then
  echo "Julia not found in PATH" >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "curl not found in PATH" >&2
  exit 1
fi

mkdir -p "$MAIN_DIR" "$RUNS_DIR" "$DATASETS_DIR" "$MODELS_DIR"

rand_float() {
  local seed="$1" lo="$2" hi="$3"
  awk -v seed="$seed" -v lo="$lo" -v hi="$hi" 'BEGIN {
    srand(seed);
    v = lo + rand() * (hi - lo);
    printf "%.6f\n", v;
  }'
}

rand_int() {
  local seed="$1" lo="$2" hi="$3"
  awk -v seed="$seed" -v lo="$lo" -v hi="$hi" 'BEGIN {
    srand(seed);
    if (hi < lo) { tmp = hi; hi = lo; lo = tmp; }
    span = hi - lo + 1;
    v = lo + int(rand() * span);
    if (v > hi) v = hi;
    printf "%d\n", v;
  }'
}

find_final_model() {
  local run_dir="$1"
  find "${run_dir}/training" -maxdepth 1 -type f -name 'final_model_iter*.jls' | sort | tail -n 1
}

api_get_scenario_n_rows() {
  local body
  body="$(curl -fsS "${SERVICE_URL}/scenario")" || return 1
  sed -n 's/.*"n_rows"[[:space:]]*:[[:space:]]*\([0-9]\+\).*/\1/p' <<<"$body" | head -n 1
}

api_set_scenario_frozen() {
  local frozen="$1"
  curl -fsS -X POST "${SERVICE_URL}/scenario/freeze?frozen=${frozen}" >/dev/null
}

api_set_benchmark_row() {
  local row_json="$1"
  curl -fsS -X POST "${SERVICE_URL}/benchmark/row" -H "Content-Type: application/json" -d "$row_json" >/dev/null
}

write_run_config() {
  local cfg_path="$1"
  shift
  : > "$cfg_path"
  while (($#)); do
    printf '%s\n' "$1" >> "$cfg_path"
    shift
  done
}

cat <<EOF
================================================================
  Batch Collect + Train From API
================================================================
  Repo:             $REPO_ROOT
  Service:          $SERVICE_URL
  Runs:             $RUNS
  Train samples:    $TRAIN_SAMPLES
  Test samples:     $TEST_SAMPLES
  Initial iter:     $INITIAL_ITER
  Step iter:        $STEP_ITER
  Max total iter:   $MAX_TOTAL_ITER
  Output root:      $MAIN_DIR
================================================================
EOF

{
  printf '%s\n' "run_id,run_seed,train_seed,test_seed,start_row,train_dp_max_mw,train_dq_max_mvar,test_dp_max_mw,test_dq_max_mvar,train_dp_samples,train_dq_samples,run_dir,train_csv,test_csv,model_path,fit_dir"
} > "$MANIFEST_CSV"

scenario_n_rows=""
if [[ "${RANDOMIZE_START_ROW}" == "true" && "${START_ROW_MODE}" != "off" ]]; then
  scenario_n_rows="$(api_get_scenario_n_rows || true)"
  if [[ -z "$scenario_n_rows" || "$scenario_n_rows" -lt 1 ]]; then
    echo "Failed to query scenario row count from ${SERVICE_URL}/scenario" >&2
    exit 1
  fi
  echo "  Scenario rows:   ${scenario_n_rows}"
fi

for ((run_idx=1; run_idx<=RUNS; run_idx++)); do
  run_id="$(printf 'run_%02d' "$run_idx")"
  run_dir="${RUNS_DIR}/${run_id}"
  run_log="${run_dir}/collect_and_train.log"
  run_seed=$((MASTER_SEED + run_idx * 1000))
  train_seed=$((run_seed + 11))
  test_seed=$((run_seed + 29))

  train_dp_max="$(rand_float $((run_seed + 101)) "$TRAIN_DP_MAX_LO" "$TRAIN_DP_MAX_HI")"
  train_dq_max="$(rand_float $((run_seed + 102)) "$TRAIN_DQ_MAX_LO" "$TRAIN_DQ_MAX_HI")"
  test_dp_max="$(rand_float $((run_seed + 103)) "$TEST_DP_MAX_LO" "$TEST_DP_MAX_HI")"
  test_dq_max="$(rand_float $((run_seed + 104)) "$TEST_DQ_MAX_LO" "$TEST_DQ_MAX_HI")"
  train_dp_samples="$(rand_int $((run_seed + 201)) "$TRAIN_DP_SAMPLES_LO" "$TRAIN_DP_SAMPLES_HI")"
  train_dq_samples="$(rand_int $((run_seed + 202)) "$TRAIN_DQ_SAMPLES_LO" "$TRAIN_DQ_SAMPLES_HI")"
  start_row=""
  if [[ "${RANDOMIZE_START_ROW}" == "true" && "${START_ROW_MODE}" != "off" ]]; then
    row_offset="$(rand_int $((run_seed + 301)) 0 $((scenario_n_rows - 1)))"
    start_row=$(( (START_ROW_BASE + row_offset) % scenario_n_rows ))
  fi

  mkdir -p "$run_dir"

  echo ""
  echo "── ${run_id} ───────────────────────────────────────────────"
  echo "  train_seed=${train_seed} test_seed=${test_seed}"
  echo "  train_dp_max=${train_dp_max} train_dq_max=${train_dq_max}"
  echo "  test_dp_max=${test_dp_max} test_dq_max=${test_dq_max}"
  echo "  train_dp_samples=${train_dp_samples} train_dq_samples=${train_dq_samples}"
  if [[ -n "$start_row" ]]; then
    echo "  start_row=${start_row} (mode=${START_ROW_MODE})"
  fi

  write_run_config "${run_dir}/run_config.env" \
    "RUN_ID=${run_id}" \
    "RUN_SEED=${run_seed}" \
    "TRAIN_SEED=${train_seed}" \
    "TEST_SEED=${test_seed}" \
    "START_ROW=${start_row}" \
    "SERVICE_URL=${SERVICE_URL}" \
    "TRAIN_SAMPLES=${TRAIN_SAMPLES}" \
    "TEST_SAMPLES=${TEST_SAMPLES}" \
    "POLL_MS=${POLL_MS}" \
    "SETTLE_MS=${SETTLE_MS}" \
    "SLACK_BUS_ID=${SLACK_BUS_ID}" \
    "TRAIN_DP_MAX_MW=${train_dp_max}" \
    "TRAIN_DQ_MAX_MVAR=${train_dq_max}" \
    "TEST_DP_MAX_MW=${test_dp_max}" \
    "TEST_DQ_MAX_MVAR=${test_dq_max}" \
    "TRAIN_DP_SAMPLES=${train_dp_samples}" \
    "TRAIN_DQ_SAMPLES=${train_dq_samples}" \
    "TRAIN_DP_MODE=${TRAIN_DP_MODE}" \
    "TRAIN_DQ_MODE=${TRAIN_DQ_MODE}" \
    "INITIAL_ITER=${INITIAL_ITER}" \
    "STEP_ITER=${STEP_ITER}" \
    "MAX_TOTAL_ITER=${MAX_TOTAL_ITER}" \
    "THRESHOLD=${THRESHOLD}" \
    "MAX_SIZE=${MAX_SIZE}" \
    "NP_FACTOR=${NP_FACTOR}" \
    "MULTIPROCESS=${MULTIPROCESS}"

  if [[ -n "$start_row" ]]; then
    if [[ "${START_ROW_MODE}" == "pin" ]]; then
      api_set_benchmark_row "{\"row\": ${start_row}}"
    elif [[ "${START_ROW_MODE}" == "set_and_release" ]]; then
      api_set_scenario_frozen "false"
      api_set_benchmark_row "{\"row\": ${start_row}}"
      sleep "$(awk -v ms="$START_ROW_SETTLE_MS" 'BEGIN{ printf "%.3f", (ms < 0 ? 0 : ms)/1000.0 }')"
      api_set_benchmark_row '{"row": null}'
    else
      echo "Unsupported START_ROW_MODE=${START_ROW_MODE}; use set_and_release|pin|off" >&2
      exit 1
    fi
  fi

  julia --threads auto --project="${REPO_ROOT}" "$COLLECT_SCRIPT" \
    SERVICE_URL="$SERVICE_URL" \
    POLL_MS="$POLL_MS" \
    SETTLE_MS="$SETTLE_MS" \
    SLACK_BUS_ID="$SLACK_BUS_ID" \
    TRAIN_SAMPLES="$TRAIN_SAMPLES" \
    TRAIN_DP_MAX_MW="$train_dp_max" \
    TRAIN_DP_SAMPLES="$train_dp_samples" \
    TRAIN_DP_MODE="$TRAIN_DP_MODE" \
    TRAIN_DQ_MAX_MVAR="$train_dq_max" \
    TRAIN_DQ_SAMPLES="$train_dq_samples" \
    TRAIN_DQ_MODE="$TRAIN_DQ_MODE" \
    TRAIN_SEED="$train_seed" \
    TEST_SAMPLES="$TEST_SAMPLES" \
    TEST_DP_MAX_MW="$test_dp_max" \
    TEST_DQ_MAX_MVAR="$test_dq_max" \
    TEST_SEED="$test_seed" \
    INITIAL_ITER="$INITIAL_ITER" \
    STEP_ITER="$STEP_ITER" \
    MAX_TOTAL_ITER="$MAX_TOTAL_ITER" \
    THRESHOLD="$THRESHOLD" \
    MAX_SIZE="$MAX_SIZE" \
    NP_FACTOR="$NP_FACTOR" \
    MULTIPROCESS="$MULTIPROCESS" \
    OUTPUT_DIR="$run_dir" \
    2>&1 | tee "$run_log"

  train_csv="${run_dir}/train_data_complex.csv"
  test_csv="${run_dir}/test_data_complex_blockrand.csv"
  model_path="$(find_final_model "$run_dir")"
  fit_dir="${run_dir}/training/fit_plots"

  if [[ ! -f "$train_csv" ]]; then
    echo "Missing train CSV for ${run_id}: ${train_csv}" >&2
    exit 1
  fi
  if [[ ! -f "$test_csv" ]]; then
    echo "Missing test CSV for ${run_id}: ${test_csv}" >&2
    exit 1
  fi
  if [[ -z "$model_path" || ! -f "$model_path" ]]; then
    echo "Missing final model for ${run_id}" >&2
    exit 1
  fi

  ln -sfn "../runs/${run_id}/train_data_complex.csv" "${DATASETS_DIR}/${run_id}_train_data_complex.csv"
  ln -sfn "../runs/${run_id}/test_data_complex_blockrand.csv" "${DATASETS_DIR}/${run_id}_test_data_complex_blockrand.csv"
  ln -sfn "../runs/${run_id}/training/$(basename "$model_path")" "${MODELS_DIR}/${run_id}_$(basename "$model_path")"

  printf '%s\n' "${run_id},${run_seed},${train_seed},${test_seed},${start_row},${train_dp_max},${train_dq_max},${test_dp_max},${test_dq_max},${train_dp_samples},${train_dq_samples},${run_dir},${train_csv},${test_csv},${model_path},${fit_dir}" >> "$MANIFEST_CSV"
done

PY="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
"$PY" "${SCRIPT_DIR}/summarize_collect_train_runs.py" --manifest "$MANIFEST_CSV" --cross-eval

echo ""
echo "================================================================"
echo "  Done."
echo "  Main directory: ${MAIN_DIR}"
echo "  Manifest:       ${MANIFEST_CSV}"
echo "  Datasets:       ${DATASETS_DIR}"
echo "  Models:         ${MODELS_DIR}"
echo "  Run summary:    ${MAIN_DIR}/summary_boxplots (Fig. 8: cross_eval/cross_eval_heatmap_{mag,ang}_mae.png)"
echo "================================================================"
