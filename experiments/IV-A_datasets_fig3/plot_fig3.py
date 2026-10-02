#!/usr/bin/env python3
"""Fig. 3: first 100 active-power samples of the high-variability and the correlated training set.

    .venv/bin/python experiments/IV-A_datasets_fig3/plot_fig3.py [--data data/offline]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "grid"))
from coses_net import read_complex_csv  # noqa: E402


def plot_S_real(df, ax):
    for c in [c for c in df.columns if str(c).startswith("S")]:
        ax.plot(df.index, np.real(df[c].values.astype(complex)), label=str(c), linewidth=0.9, alpha=0.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "offline")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "IV-A_datasets_fig3")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    var = read_complex_csv(args.data / "train_data_complex_var.csv")
    corr = read_complex_csv(args.data / "train_data_complex.csv")

    fig, axes = plt.subplots(2, 1, figsize=(5, 4), sharex=True)
    plot_S_real(var[:100], axes[0])
    axes[0].set_title("high-variability training dataset")
    axes[0].set_ylabel("P in MW")
    axes[0].grid(True)
    plot_S_real(corr[:100], axes[1])
    axes[1].set_title("correlated training dataset")
    axes[1].set_ylabel("P in MW")
    axes[1].set_xlabel("Sample number")
    axes[1].grid(True)
    fig.tight_layout()
    for ext in ("eps", "png"):
        fig.savefig(args.out / f"training_dataset_powers_S_real.{ext}", dpi=300)
    print(f"Saved {args.out}/training_dataset_powers_S_real.{{eps,png}}")


if __name__ == "__main__":
    main()
