#!/usr/bin/env bash
# Section V-D, online resilience (Fig. 10): Uncontrolled vs. Robust Sensitivity vs. PISR.
#
#   experiments/V-D_online_resilience_fig10/run.sh          # ~5 min incl. Julia compilation
#
# 1. run_routed_stack.sh starts the routed digital twin (port 8012, 20 Hz PF, 1 Hz scenarios)
#    plus the PISR controller (models/cil_longrun) and the sensitivity benchmark
#    (identified from models/cil_longrun/train_data_complex.csv), and warms both up.
# 2. sequence_breaker_noise.py switches /control_router to none -> sensitivity -> pisr and, for
#    each, records three 5 s phases with the dP=-50 kW, dQ=-50 kvar step on bus 61 (t=1..3 s):
#    baseline (radial), breaker K4/Q1 closed (ring), radial again + measurement noise.
# Output: results/V-D_online_resilience_fig10/sequence_breaker_noise_comparison.{eps,png}
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

EXP=experiments/V-D_online_resilience_fig10
OUT=results/V-D_online_resilience_fig10
PORT="${PORT:-8012}"
mkdir -p "$OUT"

echo "[V-D/Fig10] starting routed stack"
PORT="$PORT" start_bg "$OUT/stack.log" bash "$EXP/run_routed_stack.sh"
STACK_PID="$!"

t=0
until grep -q "Stack running" "$OUT/stack.log" 2>/dev/null; do
  kill -0 "$STACK_PID" 2>/dev/null || { echo "stack exited, see $OUT/stack.log" >&2; exit 1; }
  sleep 2; t=$((t + 2))
  (( t < 1200 )) || { echo "timeout waiting for warm-up, see $OUT/stack.log" >&2; exit 1; }
done

"$PY" "$EXP/sequence_breaker_noise.py" --base-url "http://127.0.0.1:${PORT}" --out "$OUT"
echo "[V-D/Fig10] done: $OUT/sequence_breaker_noise_comparison.{eps,png}"
