#!/usr/bin/env bash
# Impedance-increase topology change (ring-identified models, breaker Q1/K4 opened).
#   experiments/V-D_online_resilience_fig10/run_impedance_increase.sh [run_06|run_09]   (Table 4, Fig. 12; ~15 min per model)
# Starts the routed stack with the ring-topology model of the given Fig. 8 run (default run_06, the
# model with the lowest off-diagonal magnitude MAE) and the sensitivity benchmark identified from the
# same training data, then records the sequence of sequence_impedance_increase.py.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

RUN_ID="${1:-run_06}"
EXP=experiments/V-D_online_resilience_fig10
OUT="${OUT:-results/V-D_impedance_increase/$RUN_ID}"
PORT="${PORT:-8012}"
mkdir -p "$OUT"

export MODEL_PATH="$ROOT_DIR/models/cross_eval/runs/$RUN_ID/training/final_model_iter1020.jls"
export TRAIN_COMPLEX_CSV="$ROOT_DIR/models/cross_eval/runs/$RUN_ID/train_data_complex.csv"

echo "[impedance-increase] starting routed stack with $RUN_ID"
PORT="$PORT" start_bg "$OUT/stack.log" bash "$EXP/run_routed_stack.sh"
STACK_PID="$!"
t=0
until grep -q "Stack running" "$OUT/stack.log" 2>/dev/null; do
  kill -0 "$STACK_PID" 2>/dev/null || { echo "stack exited, see $OUT/stack.log" >&2; exit 1; }
  sleep 2; t=$((t + 2))
  (( t < 1200 )) || { echo "timeout waiting for warm-up, see $OUT/stack.log" >&2; exit 1; }
done

cd "$EXP"
# Table 4 / Fig. 12 settings: 20 s step (the droop needs 3*tau = 10 s), 12 s settling after each breaker operation.
# For model 09 the paper uses MODES="Uncontrolled,Robust Sensitivity,PISR".
"$PY" sequence_impedance_increase.py --base-url "http://127.0.0.1:${PORT}" --out "$ROOT_DIR/$OUT" --label "$RUN_ID" \
  --repeats "${REPEATS:-3}" --hold "${HOLD:-20}" --settle "${SETTLE:-12}" \
  --modes "${MODES:-Uncontrolled,Local Q(U) droop,Robust Sensitivity,PISR}"
echo "[impedance-increase] done: $OUT"
