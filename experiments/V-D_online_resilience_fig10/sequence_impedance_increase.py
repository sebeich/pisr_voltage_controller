#!/usr/bin/env python3
"""Section V-D supplement: unmodelled topology change that INCREASES the grid impedance.

Counterpart of Fig. 10, phase 2. There, the models are identified on the radial grid and breaker
Q1 (K4 in the code) is CLOSED, which lowers the impedance. Here, the PISR surrogate and the
sensitivity benchmark are identified on RING-topology data (K4 closed) and the breaker is then
OPENED, so the controllers face a higher impedance than the one they were identified on.

Per controller (Uncontrolled / Robust Sensitivity / PISR) and per repetition, two 5 s phases with the
same dP = -50 kW, dQ = -50 kvar step on bus 61 at t = 1..3 s that Fig. 10 uses:
  1. ring    : K4 closed  (topology of the training data)
  2. radial  : K4 open    (unmodelled impedance increase)
Measurement noise is off. Sampling, disturbance and recording are imported unchanged from
sequence_breaker_noise.py (the Fig. 10 driver).

The routed stack has to be started with ring-identified models, e.g.
  MODEL_PATH=models/cross_eval/runs/run_06/training/final_model_iter1020.jls \
  TRAIN_COMPLEX_CSV=models/cross_eval/runs/run_06/train_data_complex.csv \
  bash experiments/V-D_online_resilience_fig10/run_routed_stack.sh
(see run_impedance_increase.sh, which does this and then calls this script).

Optional fourth controller "Local Q(U) droop": decentralised reactive-power droop of every controllable
inverter on its own bus voltage, standard Q(U) characteristic of VDE-AR-N 4105 (dead band 0.97..1.03 U/Un,
full reactive power at 0.93 / 1.07 U/Un, first-order response with 3*tau = 10 s). It needs no model and no
communication. Each inverter gets the same reactive capability as the bound of the optimisation-based
controllers (QLIMIT, default 100 kvar). Because of the 10 s settling time the droop is compared with a long
disturbance hold (--hold 20) and a settling time after every breaker operation (--settle 12).
"""
from __future__ import annotations
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
import argparse
import ast
import json
import threading
import time
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt

import requests
import sequence_breaker_noise as sbn

V_LIMIT = 1.05
DIST_BUS = "61"
PHASES = [("ring", True), ("radial", False)]          # (label, breaker K4 closed?)
MODES = ["Uncontrolled", "Robust Sensitivity", "PISR"]
DROOP = "Local Q(U) droop"
sbn.ROUTER_MODE[DROOP] = "none"          # the droop writes the inverter setpoints itself (see QUDroop)
DROOP_BUSES = [55, 57, 64]               # the controllable inverters of the other controllers
DROOP_QMAX_MVAR = float(os.environ.get("QLIMIT", "0.1"))
DROOP_KNEES = (0.93, 0.97, 1.03, 1.07)   # U/Un corner points of the VDE-AR-N 4105 standard Q(U) curve
DROOP_3TAU_S = 10.0                      # settling time (3 tau) of the first-order response
DROOP_PERIOD_S = 0.1                     # same 10 Hz update rate as the other controllers


def qu_characteristic(u: float) -> float:
    """Q/Qmax in the sign convention of the twin (positive = reactive power absorbed, lowers the voltage)."""
    u1, u2, u3, u4 = DROOP_KNEES
    if u >= u3:
        return min(1.0, (u - u3) / (u4 - u3))
    if u <= u2:
        return -min(1.0, (u2 - u) / (u2 - u1))
    return 0.0


class QUDroop(threading.Thread):
    """Local Q(U) droop of the controllable inverters, posted through the manual /mods channel."""

    def __init__(self, base_url: str):
        super().__init__(daemon=True)
        self.base_url = base_url; self.active = threading.Event(); self.stop = threading.Event()
        self.q = {b: 0.0 for b in DROOP_BUSES}; self.http = requests.Session()

    def reset(self):
        self.q = {b: 0.0 for b in DROOP_BUSES}

    def run(self):
        alpha = DROOP_PERIOD_S / (DROOP_3TAU_S / 3.0)
        while not self.stop.is_set():
            t0 = time.perf_counter()
            if self.active.is_set():
                try:
                    volt = self.http.get(f"{self.base_url}/state", timeout=2.0).json().get("voltages", {})
                    items = []
                    for b in DROOP_BUSES:
                        u = abs(sbn.parse_complex(volt.get(f"V{b}_complex")))
                        self.q[b] += alpha * (DROOP_QMAX_MVAR * qu_characteristic(u) - self.q[b])
                        items.append({"bus_ext": b, "dP_mw": 0.0, "dQ_mvar": self.q[b]})
                    self.http.post(f"{self.base_url}/mods", json={"items": items}, timeout=2.0)
                except Exception as exc:
                    print("droop step failed:", exc)
            time.sleep(max(0.0, DROOP_PERIOD_S - (time.perf_counter() - t0)))
STEADY = (2.3, 2.95)   # s within a phase: disturbance active (1..3 s); the first 1.3 s are discarded so that
                       # the occasional double application of an incremental update (see diag_noise_repeats.py) has decayed
PRE = (0.2, 0.95)      # s within a phase: before the disturbance


def split_mods(mods):
    """Controller part (all buses except the disturbance bus) of the applied modifications."""
    if isinstance(mods, str):
        try:
            mods = ast.literal_eval(mods)
        except Exception:
            mods = {}
    p = q = 0.0
    for bus, v in (mods or {}).items():
        if str(bus) == DIST_BUS:
            continue
        p += float(v.get("dP_mw", 0.0)); q += float(v.get("dQ_mvar", 0.0))
    return p, q


def record(rate: float, repeats: int, modes: List[str], settle_s: float = 0.2) -> pd.DataFrame:
    phase_len = max(ev["start"] + ev["duration"] for ev in sbn.SEQUENCE) + 2.0
    parts = []
    droop = QUDroop(sbn.BASE_URL) if DROOP in modes else None
    if droop is not None:
        droop.start()
    for rep in range(1, repeats + 1):
        for mode in modes:
            print(f"\n=== repetition {rep}, mode {mode} ===")
            sbn.set_noise(enabled=False, v_noise_std=0.0, s_noise_std=0.0)
            sbn.del_mods()
            sbn.set_breaker(True)                      # start every mode on the trained (ring) topology
            sbn.set_router(sbn.ROUTER_MODE[mode])
            if droop is not None:
                droop.reset()
                droop.active.set() if mode == DROOP else droop.active.clear()
            for k, (label, closed) in enumerate(PHASES):
                resp = sbn.set_breaker(closed)
                time.sleep(settle_s)
                print(f" phase {label}: breaker closed={closed} -> {resp}")
                df = sbn.run_sequence(f"{mode}_{label}_rep{rep}", rate=rate)
                df["t_phase"] = df["i"] / float(rate)
                df["t_rel"] = df["t_phase"] + k * phase_len
                df["phase"] = label; df["mode"] = mode; df["rep"] = rep
                parts.append(df)
            if droop is not None:
                droop.active.clear(); time.sleep(2 * DROOP_PERIOD_S)
            sbn.del_mods()
    if droop is not None:
        droop.stop.set()
    sbn.set_router("none", settle_s=0.0)
    sbn.set_breaker(True)
    full = pd.concat(parts, ignore_index=True)
    ctrl = full["mods"].apply(split_mods)
    full["ctrl_dP_mw"] = ctrl.apply(lambda x: x[0]); full["ctrl_dQ_mvar"] = ctrl.apply(lambda x: x[1])
    return full


def steady_window(hold_s: float):
    """Evaluation window within a phase (disturbance starts at 1 s and lasts hold_s)."""
    if hold_s >= 10.0:
        return (1.0 + hold_s - 3.0, 1.0 + hold_s - 0.05)     # last 3 s of a long hold
    return STEADY


def summarise(full: pd.DataFrame, STEADY=STEADY) -> pd.DataFrame:
    rows = []
    for (mode, phase, rep), g in full.groupby(["mode", "phase", "rep"], sort=False):
        s = g[(g.t_phase >= STEADY[0]) & (g.t_phase <= STEADY[1])]
        p = g[(g.t_phase >= PRE[0]) & (g.t_phase <= PRE[1])]
        rows.append({
            "mode": mode, "phase": phase, "rep": rep, "n_steady": len(s),
            "vmax_pre_pu": p.max_voltage.mean(),
            "vmax_steady_mean_pu": s.max_voltage.mean(),
            "vmax_steady_max_pu": s.max_voltage.max(),
            "vmax_peak_pu": g.max_voltage.max(),
            "excess_steady_mean_pu": s.max_voltage.mean() - V_LIMIT,
            "ctrl_dQ_steady_kvar": 1e3 * s.ctrl_dQ_mvar.mean(),
            "ctrl_dP_steady_kw": 1e3 * s.ctrl_dP_mw.mean(),
            "ctrl_dQ_steady_std_kvar": 1e3 * s.ctrl_dQ_mvar.std(),
        })
    return pd.DataFrame(rows)


def plot(full: pd.DataFrame, out: str, rep: int = 1):
    """Fig. style of Fig. 10 (sequence_breaker_noise.py); one repetition, both phases stitched."""
    phase_len = max(ev["start"] + ev["duration"] for ev in sbn.SEQUENCE) + 2.0
    d = full[full.rep == rep]
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    axes[0].axhline(V_LIMIT, color="r", linestyle="--", label="Voltage Limit")
    colors = {"Uncontrolled": "C0", "Robust Sensitivity": "C1", "PISR": "C2", DROOP: "C3"}
    for mode in [m for m in ["Uncontrolled", DROOP, "Robust Sensitivity", "PISR"] if m in set(d["mode"])]:
        g = d[d["mode"] == mode]
        axes[0].plot(g.t_rel, g.max_voltage, label=mode, color=colors[mode])
        axes[1].plot(g.t_rel, 1e3 * g.total_dP_mw, label=mode, color=colors[mode])
        axes[2].plot(g.t_rel, 1e3 * g.total_dQ_mvar, label=mode, color=colors[mode])
    for ax in axes:
        ax.axvline(phase_len, color="blue", linestyle="--", linewidth=1)
        ax.grid(True)
    ymid = sum(axes[1].get_ylim()) / 2.0
    axes[1].text(0.5 * phase_len, ymid, "Q1 closed (identified topology)", ha="center", va="center", fontsize="small")
    axes[1].text(1.5 * phase_len, ymid, "Q1 open (impedance increase)", ha="center", va="center", fontsize="small")
    axes[0].set_ylabel("Max voltage in p.u."); axes[1].set_ylabel("Total \u0394P in kW"); axes[2].set_ylabel("Total \u0394Q in kvar")
    axes[2].set_xlabel("Time in s"); axes[0].set_xlim(0, 2 * phase_len)
    axes[0].legend(fontsize="small", loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False)
    fig.tight_layout()
    for ext in ("png", "eps"):
        fig.savefig(os.path.join(out, f"impedance_increase_comparison.{ext}"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8012")
    ap.add_argument("--rate", type=float, default=20.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--label", default="", help="free-text tag stored in the summary (e.g. model id)")
    ap.add_argument("--out", default=os.path.join(sbn.BASE_DIR, "results", "V-D_impedance_increase"))
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--hold", type=float, default=2.0, help="duration of the disturbance step in s (Fig. 10: 2 s)")
    ap.add_argument("--settle", type=float, default=0.2, help="waiting time after every breaker operation in s")
    args = ap.parse_args()
    sbn.SEQUENCE[0]["duration"] = float(args.hold)
    window = steady_window(args.hold)
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "impedance_increase_recording.csv")
    if args.replot:
        full = pd.read_csv(csv_path)
    else:
        sbn.BASE_URL = args.base_url; sbn.RATE = args.rate
        # The routed twin holds the scenario operating point by default (GET /scenario -> frozen: true),
        # so every controller and repetition sees the same pre-disturbance state.
        try:
            scen = sbn.session.get(f"{sbn.BASE_URL}/scenario", timeout=5.0).json()
        except Exception:
            scen = {}
        print("scenario state:", scen)
        full = record(args.rate, args.repeats, [m.strip() for m in args.modes.split(",") if m.strip()], settle_s=args.settle)
        keep = [c for c in full.columns if not c.endswith("_complex")] + [c for c in full.columns if c.startswith("S") and c.endswith("_complex")]
        full[keep].to_csv(csv_path, index=False)
    summ = summarise(full, window)
    summ.to_csv(os.path.join(args.out, "impedance_increase_summary.csv"), index=False)
    agg = (summ.groupby(["mode", "phase"], sort=False)
               [["vmax_steady_mean_pu", "excess_steady_mean_pu", "vmax_peak_pu", "ctrl_dQ_steady_kvar", "ctrl_dP_steady_kw"]]
               .agg(["mean", "min", "max"]))
    with open(os.path.join(args.out, "impedance_increase_summary.json"), "w") as fh:
        json.dump({"label": args.label, "v_limit_pu": V_LIMIT, "steady_window_s": window, "hold_s": args.hold, "settle_s": args.settle,  "repeats": int(summ.rep.max()),
                   "per_run": summ.to_dict(orient="records")}, fh, indent=1)
    print(agg.round(4).to_string())
    plot(full, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
