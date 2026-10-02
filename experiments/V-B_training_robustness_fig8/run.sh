#!/usr/bin/env bash
# Section V-B (Fig. 8): robustness of the training procedure, 10 models x 10 datasets.
#
#   experiments/V-B_training_robustness_fig8/run.sh plot        # heatmaps from the shipped cross_eval_metrics.csv (seconds)
#   experiments/V-B_training_robustness_fig8/run.sh crosseval   # re-evaluate the 10 shipped models on the 10 datasets (minutes)
#   experiments/V-B_training_robustness_fig8/run.sh collect     # collect 10 new datasets over the REST API and train 10 models (hours)
#
# collect: starts the routed digital twin (src/services/realtime_powerflow_service_routed_apicontrolled.py,
# 15 Hz PF as in the paper batch, default measurement noise) and runs batch_collect_and_train.sh:
# per run a seed (4242 + 1000 i), perturbation magnitudes U(15, 80) kW/kvar, a random scenario start row,
# 50 training / 30 test samples, then PISR training (src/pisr/incremental_rt.jl, 1020 iterations),
# followed by the cross-evaluation of every model on every dataset.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
EXP=experiments/V-B_training_robustness_fig8
OUT=results/V-B_training_robustness_fig8
PORT="${PORT:-8012}"
mkdir -p "$OUT"

case "${1:-plot}" in
  plot)
    "$PY" "$EXP/plot_fig8.py" ;;
  crosseval)
    "$PY" "$EXP/summarize_collect_train_runs.py" --manifest models/cross_eval/manifest.csv \
      --output-dir "$OUT/shipped_models" --cross-eval
    "$PY" "$EXP/plot_fig8.py" --metrics "$OUT/shipped_models/cross_eval/cross_eval_metrics.csv" --out "$OUT/shipped_models" ;;
  collect)
    start_bg "$OUT/service.log" "$PY" src/services/realtime_powerflow_service_routed_apicontrolled.py \
      --host 127.0.0.1 --port "$PORT" --pf-rate "${PF_RATE:-15}" --scenario-rate 1
    wait_http "http://127.0.0.1:${PORT}/status" 120
    SERVICE_URL="http://127.0.0.1:${PORT}" bash "$EXP/batch_collect_and_train.sh" ;;
  *) echo "usage: $0 {plot|crosseval|collect}" >&2; exit 2 ;;
esac
