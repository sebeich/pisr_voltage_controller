#!/usr/bin/env python3
"""Fig. 9: offline optimization on the test set.

Takes the setpoints chosen by the PISR-based optimizer (src/controllers/mpc_noslack.jl output
updated_powers_controlled_complex.csv: optimized S columns + PISR-predicted V columns), re-runs the
AC power flow of the digital twin with these setpoints to get the "actual" controlled voltages, and
plots them against the uncontrolled test set and the PISR prediction.

    .venv/bin/python experiments/V-C_offline_control_fig9/plot_fig9.py [--controlled CSV] [--out DIR]

Plotted is the maximum over the prosumer buses (bus 57 in every test sample).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "grid"))
from coses_net import CosesGrid, DESIRED_SCENARIO_BUS_ORDER, parse_complex, read_complex_csv  # noqa: E402

OUT_DEFAULT = ROOT / "results" / "V-C_offline_control_fig9"


def replay_controlled(grid: CosesGrid, controlled_csv: Path) -> pd.DataFrame:
    ctrl = pd.read_csv(controlled_csv)
    for c in ctrl.columns:
        if str(c).endswith("_complex"):
            ctrl[c] = ctrl[c].map(parse_complex)
    v_cols = [c for c in ctrl.columns if str(c).startswith("V") and str(c).endswith("_complex")]
    pred = ctrl[v_cols].rename(columns={c: c.replace("_complex", "_pred_complex") for c in v_cols})
    rows = [grid.solve_row({ext: ctrl.at[i, f"S{ext}_complex"] if f"S{ext}_complex" in ctrl else 0j
                            for ext in DESIRED_SCENARIO_BUS_ORDER}) for i in range(len(ctrl))]
    pf = pd.DataFrame(rows, columns=grid.columns(last_v="V65_complex"))
    return pd.concat([pf.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controlled", type=Path, default=OUT_DEFAULT / "pisr" / "updated_powers_controlled_complex.csv")
    ap.add_argument("--test", type=Path, default=ROOT / "data" / "offline" / "test_data_complex_blockrand.csv")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    test = read_complex_csv(args.test)
    controlled = replay_controlled(CosesGrid(), args.controlled)
    controlled.to_csv(args.out / "updated_powers_controlled_pf_results.csv", index=False)

    buses = [c for c in controlled.columns if c.endswith("_pred_complex")]
    buses = [c.replace("_pred_complex", "_complex") for c in buses]          # prosumer buses predicted by PISR
    v_unc = test[buses].abs().max(axis=1)
    v_ctl = controlled[buses].abs().max(axis=1)
    v_pred = controlled[[b.replace("_complex", "_pred_complex") for b in buses]].abs().max(axis=1)
    print("bus with the maximum voltage:", test[buses].abs().idxmax(axis=1).value_counts().to_dict())
    print(f"uncontrolled max |V| = {v_unc.max():.4f}, controlled (PF) max = {v_ctl.max():.4f}, "
          f"samples > 1.05: {(v_unc > 1.05).sum()} -> {(v_ctl > 1.05).sum()}, "
          f"max PISR prediction error after control = {(v_ctl - v_pred).abs().max():.4f} p.u.")

    plt.figure(figsize=(5, 4))
    plt.hlines(xmin=0.0, xmax=100, y=1.05, color="r", linestyle="--", label="Voltage Limit")
    plt.plot(v_unc, label="Maximum Uncontrolled Test Dataset Voltage")
    plt.plot(v_ctl, label="Maximum Controlled Test Dataset Voltage")
    plt.plot(v_pred, label="Maximum Controlled Voltage predicted by PISR")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 100)
    plt.xlabel("Test sample")
    plt.ylabel("Max voltage in p.u.")
    plt.tight_layout()
    for ext in ("eps", "png"):
        plt.savefig(args.out / f"Controlled_PF_V57_comparison.{ext}", dpi=600)
    print(f"Saved {args.out}/Controlled_PF_V57_comparison.{{eps,png}}")


if __name__ == "__main__":
    main()
