#!/usr/bin/env python3
"""Fig. 8: 10x10 cross-evaluation heatmaps (magnitude MAE, angle MAE) from a cross_eval_metrics.csv.

    .venv/bin/python experiments/V-B_training_robustness_fig8/plot_fig8.py [--metrics CSV] [--out DIR]

Also prints the per-model off-diagonal mean MAE ranges quoted in Section V-B.
"""
import os
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # reproducible EPS files (no creation timestamp)
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


LABELS = {"mag_mae": "Voltage magnitude MAE in p.u.", "ang_mae": "Voltage angle MAE in rad"}


def plot_heatmap(df: pd.DataFrame, metric: str, out_dir: Path):
    piv = df.pivot(index="model_run_id", columns="test_run_id", values=metric)
    datasets = [r.replace("run_", "") for r in piv.columns]
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(piv.values, aspect="auto")
    ax.set_xticks(range(len(datasets)), datasets)
    ax.set_yticks(range(len(piv.index)), [r.replace("run_", "") for r in piv.index])
    ax.set_xlabel("Evaluation dataset")
    ax.set_ylabel("Model (trained on dataset)")
    fig.colorbar(im, ax=ax).set_label(LABELS[metric])
    fig.savefig(out_dir / f"cross_eval_heatmap_{metric}.png", dpi=180)
    fig.savefig(out_dir / f"cross_eval_heatmap_{metric}.svg")
    fig.savefig(out_dir / f"cross_eval_heatmap_{metric}.eps")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, default=ROOT / "models" / "cross_eval" / "cross_eval_metrics.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "V-B_training_robustness_fig8")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.metrics)
    for metric in ("mag_mae", "ang_mae"):
        plot_heatmap(df, metric, args.out)
    off = df[df["model_run_id"] != df["test_run_id"]]
    per_model = off.groupby("model_run_id")[["mag_mae", "ang_mae"]].mean()
    print(per_model.to_string(float_format=lambda v: f"{v:.3e}"))
    print(f"off-diagonal magnitude MAE: {per_model.mag_mae.min():.2e} .. {per_model.mag_mae.max():.2e} p.u.")
    print(f"off-diagonal angle MAE:     {per_model.ang_mae.min():.2e} .. {per_model.ang_mae.max():.2e} rad")
    print(f"Saved {args.out}/cross_eval_heatmap_{{mag,ang}}_mae.{{png,svg}}")


if __name__ == "__main__":
    main()
