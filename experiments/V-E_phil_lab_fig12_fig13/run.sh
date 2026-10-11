#!/usr/bin/env bash
# Section V-E (Figs. 12, 13): PHiL validation in the CoSES laboratory.
#
#   experiments/V-E_phil_lab_fig12_fig13/run.sh           # Figs. 12 and 13 from the lab recordings (data/lab)
#
# The live experiment itself needs the laboratory (see README, Sec. V-E).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
EXP=experiments/V-E_phil_lab_fig12_fig13
OUT=results/V-E_phil_lab_fig12_fig13
mkdir -p "$OUT"

case "${1:-plot}" in
  plot)
    "$PY" "$EXP/plot_fig12_fig13.py" --out "$OUT" ;;
  *) echo "usage: $0 {plot}" >&2; exit 2 ;;
esac
