#!/usr/bin/env bash
# Section V-C: offline control efficacy (Fig. 9) and computational speed.
#
#   experiments/V-C_offline_control_fig9/run.sh            # PISR optimization on the test set + Fig. 9 (~2 min)
#   experiments/V-C_offline_control_fig9/run.sh speed      # same optimization with pandapower runpp as inner model
#
# pisr:  src/controllers/mpc_noslack.jl, model models/offline_highvar (high-variability training set),
#        cost (1) with w_P = 30, penalty 1e4, |dQ| <= 0.1 Mvar, 0 <= dP <= 0.1 MW, BlackBoxOptim DE,
#        population 20, <= 200 evaluations, 0.1 s time budget per sample.
#        Prints the per-sample optimization time and the single-input PISR inference time.
# speed: src/services/powerflow_solver_api.py (pandapower runpp: NR, flat start, tol 1e-7 MVA,
#        max 30 iterations, voltage angles, Q limits, numba) serves POST /solve; src/controllers/mpc_powerflow.jl
#        runs the identical optimization with that solver as the inner voltage model.
#        Compare the "[timing]" lines (per-evaluation cost) of both runs.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
EXP=experiments/V-C_offline_control_fig9
OUT=results/V-C_offline_control_fig9

case "${1:-pisr}" in
  pisr)
    mkdir -p "$OUT/pisr"
    (cd "$OUT/pisr" && "${JL[@]}" "$ROOT_DIR/src/controllers/mpc_noslack.jl") 2>&1 | tee "$OUT/pisr/mpc_noslack.log"
    "$PY" "$EXP/plot_fig9.py" --controlled "$OUT/pisr/updated_powers_controlled_complex.csv" --out "$OUT"
    grep '\[timing\]' "$OUT/pisr/mpc_noslack.log" ;;
  speed)
    mkdir -p "$OUT/powerflow"
    PORT="${PORT:-8001}"
    start_bg "$OUT/powerflow/solver_api.log" "$PY" src/services/powerflow_solver_api.py --host 127.0.0.1 --port "$PORT"
    wait_http "http://127.0.0.1:${PORT}/info" 120
    (cd "$OUT/powerflow" && SOLVER_URL="http://127.0.0.1:${PORT}/solve" "${JL[@]}" "$ROOT_DIR/src/controllers/mpc_powerflow.jl") \
      2>&1 | tee "$OUT/powerflow/mpc_powerflow.log"
    grep '\[timing\]' "$OUT/pisr/mpc_noslack.log" "$OUT/powerflow/mpc_powerflow.log" || true ;;
  *) echo "usage: $0 {pisr|speed}" >&2; exit 2 ;;
esac
