#!/usr/bin/env bash
# Copy the paper figures (EPS) from results/ into paper_figures/ under their original file names,
# so the folder can be copied as is into the LaTeX project.
#
#   experiments/collect_paper_figures.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT=paper_figures
mkdir -p "$OUT"
FIGS=(
  results/IV-A_datasets_fig3/training_dataset_powers_S_real.eps                        # Fig. 3
  results/V-A_training_data_fig7/training_progress_compare.eps                         # Fig. 7
  results/V-B_training_robustness_fig8/cross_eval_heatmap_mag_mae.eps                  # Fig. 8 (left)
  results/V-B_training_robustness_fig8/cross_eval_heatmap_ang_mae.eps                  # Fig. 8 (right)
  results/V-C_offline_control_fig9/Controlled_PF_V57_comparison.eps                    # Fig. 9
  results/V-D_online_resilience_fig10/sequence_breaker_noise_comparison.eps            # Fig. 10
  results/V-D_long_run_fig11_table2/controller_comparison/comparison.eps               # Fig. 11
  results/V-E_phil_lab_fig12_fig13/CoSES_PHiL_Validation_PISR_Controller_Operation.eps        # Fig. 12
  results/V-E_phil_lab_fig12_fig13/CoSES_PHiL_Validation_PISR_Controller_Operation_Short.eps  # Fig. 13
)
for src in "${FIGS[@]}"; do
  if [[ -f "$src" ]]; then cp "$src" "$OUT/"; echo "  $OUT/$(basename "$src")"
  else echo "  MISSING $src (run its experiment first)" >&2; fi
done
