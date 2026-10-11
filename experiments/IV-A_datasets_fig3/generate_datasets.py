#!/usr/bin/env python3
"""Section IV-A: offline training/test datasets from the AR(1) scenarios.

One AC power flow (pandapower, NR, flat start) per 1 s scenario sample:
  train_data_complex.csv            correlated training set   (scenarios_correlated_pv.csv, rows 1-500)
  train_data_complex_var.csv        high-variability training set (scenarios_random_high_var.csv, rows 1-500)
  test_data_complex_blockrand.csv   test set: correlated rows 501-600; in block k (k=1..5) the P/Q of
                                    prosumer k is replaced by uniform random values (Fig. 7 test set, Fig. 9)

    .venv/bin/python experiments/IV-A_datasets_fig3/generate_datasets.py [--seed 0] [--out DIR]

The paper datasets (data/offline/) were produced without a fixed seed for the block randomization;
the training sets are deterministic given the scenario CSVs. Default output is results/, so the
shipped data is not overwritten.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "grid"))
from coses_net import CosesGrid, DESIRED_SCENARIO_BUS_ORDER  # noqa: E402

N_TRAIN, N_TEST = 500, 100


def randomize_blocks(scenarios: pd.DataFrame, rng: np.random.Generator, n_loads: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    block = N_TEST // (n_loads + 1)
    test = scenarios.iloc[N_TRAIN:N_TRAIN + N_TEST].copy()
    doc = []
    for i in range(n_loads):
        start, end = (i + 1) * block, min((i + 2) * block, N_TEST)
        if start >= N_TEST:
            break
        p, q = f"P_node{i+1}", f"Q_node{i+1}"
        lo_p, hi_p = test[p].min() - 5, test[p].max() + 5
        lo_q, hi_q = test[q].min(), test[q].max()
        test.loc[test.index[start:end], p] = rng.uniform(lo_p, hi_p, end - start)
        test.loc[test.index[start:end], q] = rng.uniform(lo_q, hi_q, end - start)
        doc.append({"load": i + 1, "p_col": p, "q_col": q, "start_idx": start, "end_idx": end - 1})
    return test.reset_index(drop=True), pd.DataFrame(doc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", type=Path, default=ROOT / "data" / "scenarios")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "IV-A_datasets_fig3" / "datasets")
    ap.add_argument("--seed", type=int, default=None, help="seed for the test-set block randomization")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    grid = CosesGrid()

    corr = pd.read_csv(args.scenarios / "scenarios_correlated_pv.csv")
    var = pd.read_csv(args.scenarios / "scenarios_random_high_var.csv")
    grid.run_pf_and_collect_complex(corr, 1, N_TRAIN).to_csv(args.out / "train_data_complex.csv", index=False)
    grid.run_pf_and_collect_complex(var, 1, N_TRAIN).to_csv(args.out / "train_data_complex_var.csv", index=False)

    test_scen, doc = randomize_blocks(corr, np.random.default_rng(args.seed), len(DESIRED_SCENARIO_BUS_ORDER))
    grid.run_pf_and_collect_complex(test_scen, 1, N_TEST).to_csv(args.out / "test_data_complex_blockrand.csv", index=False)
    doc.to_csv(args.out / "test_data_blockrand_cx_randomization_doc.csv", index=False)
    print(f"Datasets written to {args.out}")


if __name__ == "__main__":
    main()
