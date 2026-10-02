#!/usr/bin/env python3
"""Fig. 7: PISR training progress (mean |V| MAE on the test set) for correlated vs high-variability data.

    .venv/bin/python experiments/V-A_training_data_fig7/plot_training_progress.py
    .venv/bin/python experiments/V-A_training_data_fig7/plot_training_progress.py \
        --corr results/V-A_training_data_fig7/correlated/metrics_progress.csv \
        --var  results/V-A_training_data_fig7/highvar/metrics_progress.csv

Defaults read the metrics shipped with the paper models (models/offline_*/metrics_progress_*.csv).
"""
from __future__ import annotations
import os
import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def load_metrics(p: str) -> pd.DataFrame:
    df = pd.read_csv(p)
    # ensure numeric columns
    for c in ['iteration', 'mean_mag_mae', 'worst_mag_max', 'mean_mag_mae', 'worst_ang_max']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def approx_rmse_from_mae(mae: pd.Series) -> pd.Series:
    # For approximately Gaussian zero-mean errors: E|X| = sigma * sqrt(2/pi)
    # so sigma (RMSE) ≈ mae * sqrt(pi/2)
    factor = math.sqrt(math.pi / 2.0)
    return mae * factor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corr', default=os.path.join(ROOT, 'models', 'offline_correlated', 'metrics_progress_corr.csv'))
    parser.add_argument('--var', default=os.path.join(ROOT, 'models', 'offline_highvar', 'metrics_progress_var.csv'))
    parser.add_argument('--outdir', default=os.path.join(ROOT, 'results', 'V-A_training_data_fig7'))
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df_corr = load_metrics(args.corr)
    df_var = load_metrics(args.var)


    # We'll plot mean magnitude MAE (average across buses) directly
    it_corr = df_corr['iteration'] if 'iteration' in df_corr.columns else df_corr.index
    it_var = df_var['iteration'] if 'iteration' in df_var.columns else df_var.index

    # Prepare plot
    fig, axes = plt.subplots(1, 1, figsize=(5, 2), sharex=True)

    ax1 = axes
    # Worst-bus MAE (peak) comparison
    if 'worst_mag_max' in df_corr.columns:
        ax1.plot(it_corr, df_corr['mean_mag_mae'], marker='o', linestyle='-', label='Correlated mean MAE')
    if 'worst_mag_max' in df_var.columns:
        ax1.plot(it_var, df_var['mean_mag_mae'], marker='s', linestyle='--', label='High-variability mean MAE')
    ax1.set_ylabel('Mean MAE (p.u.)')
    ax1.set_xlabel('Training iterations')
    ax1.grid(True)
    ax1.legend(fontsize='small')

    plt.tight_layout()

    out_png = os.path.join(args.outdir, 'training_progress_compare.png')
    out_pdf = os.path.join(args.outdir, 'training_progress_compare.eps')
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    print('Saved', out_png, out_pdf)


if __name__ == '__main__':
    main()
