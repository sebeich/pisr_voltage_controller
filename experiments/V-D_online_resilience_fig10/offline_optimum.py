"""Offline optimum of the controller cost (1) for the recorded operating points of the
impedance-increase experiment (Table 4, row "Optimum (full model)").
Uses the digital-twin network of the routed service, the recorded uncontrolled steady state
(including the -50 kW/-50 kvar step at bus 61), cost 30*sum(dP^2) + sum(dQ^2) [MW, Mvar],
bounds |dQ| <= 0.1 Mvar, 0 <= dP <= min(0.3 MW, feed-in), and |V| <= 1.05 p.u. at every energized
low-voltage bus as hard constraint (SLSQP on the full AC power flow). In all recorded operating points
the grid maximum is at bus 61, so constraining only the four prosumer buses gives the same optimum.
    python offline_optimum.py <impedance_increase_recording.csv[.gz]> [out.json]
Writes the result to out.json (default: results/V-D_impedance_increase/optimum/optimum.json).
"""
import ast, copy, json, os, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "services"))
import realtime_powerflow_service_routed_apicontrolled as svc

BUSES = [64, 57, 55, 61]; CTRL = [64, 57, 55]; WP = 30.0; QLIM = 0.1; PLIM = 0.3; VLIM = 1.05

def main(rec_csv, out_json=None):
    rec_csv = os.path.abspath(rec_csv)
    out_json = os.path.abspath(out_json) if out_json else os.path.join(ROOT, "results", "V-D_impedance_increase", "optimum", "optimum.json")
    result = {}
    os.chdir(ROOT)
    rtp = svc.RealTimePowerflow(os.path.join("data", "grid", "coses_matpower_source.xlsx"),
                                os.path.join("data", "scenarios", "scenarios_correlated_pv_rt.csv"))
    rec = pd.read_csv(rec_csv)
    for phase, closed in (("ring", True), ("radial", False)):
        g = rec[(rec["mode"] == "Uncontrolled") & (rec.phase == phase) & (rec.rep == 1)]
        g = g[(g.total_dP_mw - g.ctrl_dP_mw) < -0.04]
        S = {b: complex(str(g[f"S{b}_complex"].iloc[-1]).replace(" ", "")) for b in BUSES}
        net = copy.deepcopy(rtp.net_template); net.switch.at[rtp.k4_sw_idx, "closed"] = closed
        for b in BUSES:
            li = rtp.ext_to_load_idx[b]; net.load.at[li, "p_mw"] = S[b].real; net.load.at[li, "q_mvar"] = S[b].imag
        rtp._run_powerflow(net)
        sup = [b for b in net.bus.index[net.bus.vn_kv < 1.0] if np.isfinite(net.res_bus.vm_pu.get(b, np.nan))]
        def volts(x):
            for i, b in enumerate(CTRL):
                li = rtp.ext_to_load_idx[b]
                net.load.at[li, "p_mw"] = S[b].real + x[2 * i]; net.load.at[li, "q_mvar"] = S[b].imag + x[2 * i + 1]
            rtp._run_powerflow(net); return net.res_bus.vm_pu.loc[sup].values
        bnds = sum([[(0.0, min(PLIM, max(0.0, -S[b].real))), (-QLIM, QLIM)] for b in CTRL], [])
        f = lambda x: WP * np.sum(x[0::2] ** 2) + np.sum(x[1::2] ** 2)
        res = minimize(f, np.zeros(6), method="SLSQP", bounds=bnds,
                       constraints=[{"type": "ineq", "fun": lambda x: VLIM - volts(x)}], options={"ftol": 1e-12, "maxiter": 500})
        print(f"{phase}: J={res.fun:.6f} sum dQ={res.x[1::2].sum()*1e3:.1f} kvar sum dP={res.x[0::2].sum()*1e3:.2f} kW "
              f"max V={volts(res.x).max():.4f} p.u. ({len(sup)} LV buses constrained) success={res.success}")
        result[phase] = {"J": float(res.fun), "Q_kvar": float(res.x[1::2].sum() * 1e3), "P_kW": float(res.x[0::2].sum() * 1e3),
                         "vmax_pu": float(volts(res.x).max()), "success": bool(res.success),
                         "dP_mw": {str(b): float(res.x[2 * i]) for i, b in enumerate(CTRL)},
                         "dQ_mvar": {str(b): float(res.x[2 * i + 1]) for i, b in enumerate(CTRL)}}
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump({"source": os.path.relpath(rec_csv, ROOT), "w_P": WP, "q_limit_mvar": QLIM, "p_limit_mw": PLIM, "v_limit_pu": VLIM, "phases": result}, fh, indent=2)
    print("saved", out_json)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
