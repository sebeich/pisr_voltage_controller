#!/usr/bin/env python3
"""Figs. 12 and 13: PHiL validation of the PISR controller in the CoSES laboratory grid.

Input: the NI VeriStand logger exports of the PHiL test on 2025-10-07 (data/lab/*.csv, measurement channels only).

    .venv/bin/python experiments/V-E_phil_lab_fig12_fig13/plot_fig12_fig13.py [--slow CSV] [--fast CSV] [--out DIR]

  --slow  log at 100 Hz with the controller repeatedly enabled/disabled  -> Fig. 12
  --fast  log at 1 kHz around one controller activation                -> Fig. 13 (samples 1800-2100)

Voltages are phase-peak values (S_EU_Mag_A / (sqrt(2) * 230 V)); amplifier powers are per phase (x3).
"""
import os
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # reproducible EPS files (no creation timestamp)
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# Sample indices (100 Hz log) at which the controller was enabled / disabled, read from the controller log
ENABLES_FIG12 = [1355, 2430, 4050]
DISABLES_FIG12 = [900, 1780, 2890, 4960]


def plot_test_data(df, out_stem: Path, figsize, sr, voltage_limit=None, enables=(), disables=(), xlim=None):
    vcols = [c for c in df.columns if "S_EU_Mag_A" in c]
    pcols = [c for c in df.columns if c.endswith("Egston.P_A")]
    qcols = [c for c in df.columns if c.endswith("Egston.Q_A")]
    p_pv = [c for c in df.columns if c.endswith("S_EP")]
    q_pv = [c for c in df.columns if c.endswith("S_EQ")]
    if not (vcols or pcols or qcols):
        raise ValueError("No matching columns found for voltage (S_EU_Mag_A), Egston.P_A or Egston.Q_A.")

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    for x in enables:
        for ax in axes:
            ax.axvline(x, color="green", linestyle="--", linewidth=2)
    for x in disables:
        for ax in axes:
            ax.axvline(x, color="blue", linestyle="--", linewidth=2)

    for c in vcols:
        axes[0].plot(df.index, df[c] / (np.sqrt(2) * 230), label=c)
    if voltage_limit is not None:
        axes[0].axhline(y=voltage_limit, color="r", linestyle="--", label="Voltage Limit")
    axes[0].set_ylabel("Max voltage in p.u.")

    for c in pcols:
        axes[1].plot(df.index, df[c] * 3 / 1000, label=c, alpha=0.9)
    for c in p_pv:
        axes[1].plot(df.index, df[c] / 1000, label=c, alpha=0.9)
    axes[1].set_ylabel("Active power in kW")

    for c in qcols:
        axes[2].plot(df.index, df[c] * 3 / 1000, label=c)
    for c in q_pv:
        axes[2].plot(df.index, df[c] / 1000, label=c)
    axes[2].set_ylabel("Reactive power in kvar")

    for ax in axes:
        ax.legend(loc="upper left", fontsize="small")
        ax.grid(True)
    axes[0].set_xlim(*(xlim if xlim else (0, len(df))))
    for ax in axes:  # sample index -> seconds
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels([f"{x / sr}" for x in ax.get_xticks()])
    axes[0].set_xlim(*(xlim if xlim else (0, len(df))))
    axes[-1].set_xlabel("Time in s")
    fig.tight_layout()
    for ext in ("png", "eps"):
        fig.savefig(out_stem.with_suffix(f".{ext}"))
    plt.close(fig)
    print(f"Saved {out_stem}.{{png,eps}}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slow", type=Path, default=ROOT / "data" / "lab" / "105setpoint.csv", help="Fig. 12 log")
    ap.add_argument("--fast", type=Path, default=ROOT / "data" / "lab" / "105setpoint_high_speed.csv", help="Fig. 13 log")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "V-E_phil_lab_fig12_fig13")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    slow = pd.read_csv(args.slow)
    plot_test_data(slow, args.out / "CoSES_PHiL_Validation_PISR_Controller_Operation", figsize=(12, 6), sr=100.0,
                   voltage_limit=1.05, enables=ENABLES_FIG12, disables=DISABLES_FIG12)

    fast = pd.read_csv(args.fast).iloc[1800:2100]
    plot_test_data(fast, args.out / "CoSES_PHiL_Validation_PISR_Controller_Operation_Short", figsize=(6, 6),
                   sr=1000.0, xlim=(1800, 2100))


if __name__ == "__main__":
    main()
