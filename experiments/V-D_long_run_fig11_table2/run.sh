#!/usr/bin/env bash
# Section V-D, long randomized run: Table 2 and Fig. 11.
#
#   experiments/V-D_long_run_fig11_table2/run.sh table2   # ~17 min  -> Table 2
#   experiments/V-D_long_run_fig11_table2/run.sh fig11    # ~2 h     -> Fig. 11
#   experiments/V-D_long_run_fig11_table2/run.sh plot     # re-plot existing recordings only
#
# Setup reproduced from the paper recordings:
#   * digital twin: src/services/realtime_powerflow_service_with_noise_row_api.py
#     (scenario rows advance at 1 Hz -> one random P/Q event per second at the uncontrolled nodes)
#   * fixed -40 kW offset on uncontrolled bus 61, measurement noise off, breaker open (radial)
#   * PISR controller: src/controllers/rt_controller_showerror.jl with models/cil_longrun
#   * Table 2: PISR only, scenario rows 799..1799 (1000 s, ~19 948 samples at 20 Hz)
#   * Fig. 11: controller OFF, then PISR ON, replaying the same rows from row 1000 for 3600 s
# Override durations with T2_DURATION / FIG11_DURATION (seconds) for a quick smoke test.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

EXP=experiments/V-D_long_run_fig11_table2
OUT=results/V-D_long_run_fig11_table2
PORT="${PORT:-8000}"
BASE="http://127.0.0.1:${PORT}"
export PF_BASE_URL="$BASE"
T2_DURATION="${T2_DURATION:-1000}"
T2_START_ROW="${T2_START_ROW:-799}"
FIG11_DURATION="${FIG11_DURATION:-3600}"
FIG11_START_ROW="${FIG11_START_ROW:-1000}"
mkdir -p "$OUT"

start_service() {
  echo "[V-D] starting digital twin on :$PORT"
  start_bg "$OUT/service.log" "$PY" src/services/realtime_powerflow_service_with_noise_row_api.py \
    --host 127.0.0.1 --port "$PORT" --pf-rate 20 --scenario-rate 1
  SERVICE_PID="$!"
  wait_http "$BASE/status" 120
  reset_operating_point
  # wait for the first completed power flow (voltages published)
  until curl -sf "$BASE/state" | "$PY" -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("voltages") else 1)'; do sleep 1; done
}

reset_operating_point() {
  api DELETE "$BASE/mods" >/dev/null
  api POST "$BASE/mods" '{"items":[{"bus_ext":61,"dP_mw":-0.04,"dQ_mvar":0.0}]}' >/dev/null
  api POST "$BASE/noise" '{"enabled":false}' >/dev/null
  api POST "$BASE/breaker?closed=false" >/dev/null || true
}

start_controller() {
  echo "[V-D] starting PISR controller (first start compiles, this can take a few minutes)"
  local since; since="$(date +%s)"
  start_bg "$OUT/pisr_controller_$1.log" "${JL[@]}" src/controllers/rt_controller_showerror.jl
  CTRL_PID="$!"
  WAIT_PID="$CTRL_PID" wait_cost_source "$BASE" 900 "$since" julia pisr
}

stop_controller() {
  stop_pid "$CTRL_PID"
  reset_operating_point   # remove the last setpoints the controller left behind
}

run_table2() {
  start_controller table2
  api POST "$BASE/scenario/row" "{\"row\":${T2_START_ROW},\"frozen\":false}" >/dev/null
  "$PY" "$EXP/record_random_samples.py" --base-url "$BASE" --duration "$T2_DURATION" --out "$OUT/random_samples.csv"
  stop_controller
}

run_fig11() {
  "$PY" "$EXP/record_controller_comparison.py" --base-url "$BASE" --phase off \
    --duration "$FIG11_DURATION" --start-row "$FIG11_START_ROW" --out "$OUT/controller_comparison"
  start_controller fig11
  "$PY" "$EXP/record_controller_comparison.py" --base-url "$BASE" --phase on --out "$OUT/controller_comparison"
  stop_controller
}

plot_all() {
  [[ -f "$OUT/random_samples.csv" || -f "$OUT/random_samples.csv.gz" ]] && "$PY" "$EXP/plot_random_samples.py" --csv "$OUT/random_samples.csv" --no-show
  [[ -f "$OUT/controller_comparison/on.csv" || -f "$OUT/controller_comparison/on.csv.gz" ]] && "$PY" "$EXP/plot_controller_comparison.py" --no-show
  return 0
}

case "${1:-all}" in
  table2) start_service; run_table2; plot_all ;;
  fig11)  start_service; run_fig11;  plot_all ;;
  all)    start_service; run_table2; run_fig11; plot_all ;;
  plot)   plot_all ;;
  *) echo "usage: $0 {table2|fig11|all|plot}" >&2; exit 2 ;;
esac
echo "[V-D] Table 2 values: runs.PISR in $OUT/random_samples.stats.json; Fig. 11: $OUT/controller_comparison/comparison.{eps,png}"
