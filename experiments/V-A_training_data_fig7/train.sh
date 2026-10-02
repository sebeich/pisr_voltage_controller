#!/usr/bin/env bash
# Section V-A (Fig. 7): train the two offline PISR models and plot their training progress.
#
#   experiments/V-A_training_data_fig7/train.sh          # retrain both models (long: SR with 1020 iterations each)
#   experiments/V-A_training_data_fig7/train.sh plot     # only plot, from the shipped paper models
#
# Both models: src/pisr/incremental_noslack.jl, Algorithm 1 settings (maxsize 30, 20 + 10x100 = 1020
# iterations, npopulations = 3 x (CPU threads - 2)), 100 randomly drawn rows (TRAIN_SEED=0) of the
# 500-sample training set, evaluated on data/offline/test_data_complex_blockrand.csv.
# The high-variability model (models/offline_highvar) is the one used for the offline control (Fig. 9).
# SymbolicRegression is stochastic, so retrained models differ slightly from the shipped ones.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
EXP=experiments/V-A_training_data_fig7
OUT=results/V-A_training_data_fig7

if [[ "${1:-train}" == "train" ]]; then
  mkdir -p "$OUT"
  "${JL[@]}" src/pisr/incremental_noslack.jl TRAIN_CSV=data/offline/train_data_complex.csv \
    OUTPUT_DIR="$ROOT_DIR/$OUT/correlated" 2>&1 | tee "$OUT/train_correlated.log"
  "${JL[@]}" src/pisr/incremental_noslack.jl TRAIN_CSV=data/offline/train_data_complex_var.csv \
    OUTPUT_DIR="$ROOT_DIR/$OUT/highvar" 2>&1 | tee "$OUT/train_highvar.log"
  "$PY" "$EXP/plot_training_progress.py" --corr "$OUT/correlated/metrics_progress.csv" \
    --var "$OUT/highvar/metrics_progress.csv" --outdir "$OUT/retrained"
fi
"$PY" "$EXP/plot_training_progress.py"
