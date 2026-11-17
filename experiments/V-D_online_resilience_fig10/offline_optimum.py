"""Offline optimum of the controller cost (1) for the recorded operating points of the
impedance-increase experiment (Table 4, row "Optimum (full model)").
Uses the digital-twin network of the routed service, the recorded uncontrolled steady state
(including the -50 kW/-50 kvar step at bus 61), cost 30*sum(dP^2) + sum(dQ^2) [MW, Mvar],
bounds |dQ| <= 0.1 Mvar, 0 <= dP <= min(0.3 MW, feed-in), and |V| <= 1.05 p.u. at buses 64, 57, 55, 61
as hard constraint (SLSQP on the full AC power flow).
    python offline_optimum.py <impedance_increase_recording.csv>
"""
import ast, copy, os, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "services"))
import realtime_powerflow_service_routed_apicontrolled as svc

BUSES = [64, 57, 55, 61]; CTRL = [64, 57, 55]; WP = 30.0; QLIM = 0.1; PLIM = 0.3; VLIM = 1.05

def main(rec_csv):
    os.chdir(ROOT)
    rtp = svc.RealTimePowerflow(os.path.join("data", "grid", "coses_matpower_source.xlsx"),
                                os.path.join("data", "scenarios", "scenarios_correlated_pv_rt.csv"))
    rec = pd.read_csv(rec_csv)
    sup = [rtp.nodecount_to_busidx[b] for b in BUSES]
    for phase, closed in (("ring", True), ("radial", False)):
        g = rec[(rec["mode"] == "Uncontrolled") & (rec.phase == phase) & (rec.rep == 1)]
        g = g[(g.total_dP_mw - g.ctrl_dP_mw) < -0.04]
        S = {b: complex(str(g[f"S{b}_complex"].iloc[-1]).replace(" ", "")) for b in BUSES}
        net = copy.deepcopy(rtp.net_template); net.switch.at[rtp.k4_sw_idx, "closed"] = closed
        for b in BUSES:
            li = rtp.ext_to_load_idx[b]; net.load.at[li, "p_mw"] = S[b].real; net.load.at[li, "q_mvar"] = S[b].imag
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
              f"max V={volts(res.x).max():.4f} p.u. success={res.success}")

if __name__ == "__main__":
    main(sys.argv[1])
