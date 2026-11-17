#!/usr/bin/env python3
"""Table 4 (Sec. V-D): summarize the impedance-increase recordings.

    python experiments/V-D_online_resilience_fig10/summarize_table4.py [results/V-D_impedance_increase]

Inputs (written by run_impedance_increase.sh / sequence_impedance_increase.py and offline_optimum.py):
    <dir>/run_06/impedance_increase_summary.csv, impedance_increase_recording.csv[.gz]
    <dir>/run_09/...
    <dir>/optimum/optimum.json
Outputs in <dir>:
    table4_means.csv              steady-state max |V| (v), controller dQ (q, kvar), dP (p, kW) per mode and phase,
                                  mean over the repetitions; vr = spread of v over the repetitions
    table4_settling_times.csv     time after the step until max |V| stays within 1e-3 p.u. of its final value
    optimum/table4_optimum_comparison.csv
                                  cost (1) of each controller's steady-state setpoints vs. the offline optimum
"""
import ast, json, os, sys
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
W_P = 30.0
N_STEADY = 60          # last 3 s of the step at 20 Hz
TOL = 1e-3
LABEL = {"PISR": "PISR (model {m})", "Robust Sensitivity": "Robust sensitivity", "Local Q(U) droop": "Local Q(U) droop"}


def read_rec(d):
    p = os.path.join(d, "impedance_increase_recording.csv")
    return pd.read_csv(p if os.path.exists(p) else p + ".gz")


def step_part(g):
    """samples while the disturbance step is applied (total minus controller dP below -40 kW)"""
    return g[(g.total_dP_mw - g.ctrl_dP_mw) < -0.04]


def cost(mods):
    return sum(W_P * mods[b]["dP_mw"] ** 2 + mods[b]["dQ_mvar"] ** 2 for b in ("55", "57", "64"))


def main(base):
    means, settle, comp = [], [], []
    opt = json.load(open(os.path.join(base, "optimum", "optimum.json")))["phases"]
    for m in ("06", "09"):
        d = os.path.join(base, f"run_{m}")
        if not os.path.isdir(d):
            continue
        sm = pd.read_csv(os.path.join(d, "impedance_increase_summary.csv"))
        g = sm.groupby(["mode", "phase"]).agg(v=("vmax_steady_mean_pu", "mean"), q=("ctrl_dQ_steady_kvar", "mean"),
                                              p=("ctrl_dP_steady_kw", "mean"),
                                              vr=("vmax_steady_mean_pu", lambda x: x.max() - x.min())).reset_index()
        means.append(g.assign(model=m))
        rec = read_rec(d)
        for run, r in rec.groupby("run", sort=False):
            r = r.reset_index(drop=True); s = step_part(r)
            if s.empty:
                continue
            v_end = s.max_voltage.iloc[-N_STEADY:].mean()
            bad = s.index[(s.max_voltage - v_end).abs() > TOL]
            t = (r.t_phase[bad[-1] + 1] - r.t_phase[s.index[0]]) if len(bad) else 0.0
            settle.append(dict(model=m, mode=r["mode"][0], phase=r.phase[0], rep=int(r.rep[0]), settle_s=round(float(t), 2)))
            if r["mode"][0] == "Uncontrolled" or (m == "09" and r["mode"][0] != "PISR"):
                continue
            w = s.iloc[-N_STEADY:]
            comp.append(dict(phase=r.phase[0], controller=LABEL[r["mode"][0]].format(m=m), rep=int(r.rep[0]),
                             vmax=w.max_voltage.mean(), Q_kvar=w.ctrl_dQ_mvar.mean() * 1e3, P_kW=w.ctrl_dP_mw.mean() * 1e3,
                             J=np.mean([cost(ast.literal_eval(x)) for x in w.mods])))
    pd.concat(means).to_csv(os.path.join(base, "table4_means.csv"), index=False)
    pd.DataFrame(settle).to_csv(os.path.join(base, "table4_settling_times.csv"), index=False)
    c = pd.DataFrame(comp).groupby(["phase", "controller"], sort=False)[["vmax", "Q_kvar", "P_kW", "J"]].mean().reset_index()
    o = pd.DataFrame([dict(phase=ph, controller="Optimum (offline AC power flow, full model)", vmax=v["vmax_pu"],
                           Q_kvar=v["Q_kvar"], P_kW=v["P_kW"], J=v["J"]) for ph, v in opt.items()])
    out = pd.concat([o, c], ignore_index=True)
    out["J_rel"] = out.J / out.phase.map({ph: v["J"] for ph, v in opt.items()})
    out.to_csv(os.path.join(base, "optimum", "table4_optimum_comparison.csv"), index=False)
    print(pd.concat(means).round(4).to_string(index=False)); print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(ROOT, "results", "V-D_impedance_increase"))
