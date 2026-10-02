#!/usr/bin/env python3
"""Fair speed benchmark, power-flow side (Sec. V-C).

    .venv/bin/python experiments/V-C_offline_control_fig9/benchmark_powerflow.py [--reps 20]

Solves the AC power flow in-process (no HTTP, no network copy inside the timed region) for the same
100 test inputs that benchmark_pisr.jl evaluates. The network is the one used by the speed baseline
(src/services/powerflow_solver_api.py). The solver settings are those stated in the paper: pandapower
runpp, Newton-Raphson, flat start on every solve, tolerance 1e-7 MVA, max. 30 iterations, voltage
angles, reactive-power limits, numba. The timed call is pp.runpp(net, ...) only, so pandapower's
model building and result writing are included, as they are in every optimizer evaluation.
"""
import argparse
import cmath
import copy
import re
import timeit
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandapower as pp
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "services"))
sys.path.insert(0, str(ROOT / "src" / "grid"))
import powerflow_solver_api as api  # noqa: E402
from coses_net import read_complex_csv  # noqa: E402

RUNPP = dict(calculate_voltage_angles=True, init="flat", algorithm="nr", enforce_q_lims=True,
             max_iteration=30, tolerance_mva=1e-7, numba=True)
PROSUMERS = [64, 57, 55, 53, 61]


def environment():
    """Hardware/software description stored with the results."""
    import platform
    import numba, pandapower, scipy
    cpu = platform.processor()
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        pass
    try:
        mem_gb = round(int(next(l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemTotal"))) / 2**20, 1)
    except Exception:
        mem_gb = None
    try:
        governor = open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read().strip()
    except Exception:
        governor = None
    import os
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "memory_gb": mem_gb, "cpu_governor": governor,
            "os": platform.platform(), "python": platform.python_version(), "pandapower": pandapower.__version__,
            "numba": numba.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
            "timed_process": "single Python process, single thread, no other benchmark running in parallel"}


def compile_pisr(p):
    """Turn the exported SR equations into one plain-Python function (complex arithmetic only)."""
    def to_py(eq):
        return re.sub(r"(\d)im\b", r"\1j", eq)
    env = {"line_current": lambda v, s: (s / v).conjugate(), "voltage_drop": lambda i, z: i * z, "cmath": cmath}
    body = ", ".join(f"abs({to_py(e)})" for e in p["equations"])
    return eval(f"lambda {', '.join(p['inputs'])}: max({body})", env)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=20, help="timed repetitions per input")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "V-C_offline_control_fig9" / "benchmark_powerflow.json")
    args = ap.parse_args()

    net_template, ext_to_bus = api.create_net_from_excel(str(ROOT / "data" / "grid" / "coses_matpower_source.xlsx"))
    net = copy.deepcopy(net_template)
    bus_to_ext = {b: e for e, b in ext_to_bus.items()}
    loads = {}
    for li in net.load.index:
        if bool(net.load.at[li, "in_service"]):
            loads.setdefault(bus_to_ext.get(int(net.load.at[li, "bus"])), []).append(li)
    test = read_complex_csv(ROOT / "data" / "offline" / "test_data_complex_blockrand.csv")
    prosumer_buses = [ext_to_bus[e] for e in PROSUMERS]

    per_input, vmax = [], []
    for i in range(len(test)):
        for ext in PROSUMERS:
            s = test.at[i, f"S{ext}_complex"]
            for li in loads.get(ext, []):
                net.load.at[li, "p_mw"] = float(s.real)
                net.load.at[li, "q_mvar"] = float(s.imag)
        pp.runpp(net, **RUNPP)  # warm-up (numba JIT on the first call)
        times = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            pp.runpp(net, **RUNPP)
            times.append(time.perf_counter() - t0)
        per_input.append(statistics.median(times))
        vmax.append(float(net.res_bus.loc[prosumer_buses, "vm_pu"].max()))

    t = np.array(per_input)
    res = {"environment": environment(), "n_samples": len(t), "reps": args.reps, "settings": RUNPP,
           "network": "CoSES LV grid, src/services/powerflow_solver_api.py:create_net_from_excel",
           "warm_start": "none: init='flat' on every solve, no result/Ybus recycling",
           "runpp_median_s": float(np.median(t)), "runpp_p05_s": float(np.quantile(t, 0.05)),
           "runpp_p95_s": float(np.quantile(t, 0.95)), "vmax_pf": vmax}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"pandapower runpp (in-process, one input): median {1e3*np.median(t):.2f} ms "
          f"(p5 {1e3*np.quantile(t, 0.05):.2f}, p95 {1e3*np.quantile(t, 0.95):.2f})")

    pisr_json = args.out.with_name("benchmark_pisr.json")
    if pisr_json.exists():
        p = json.loads(pisr_json.read_text())
        ratio = res["runpp_median_s"] / p["lowlevel_median_s"]
        # PISR in the same runtime as pandapower: the exported equations evaluated in plain Python
        f = compile_pisr(p)
        x = [[test.at[i, c] for c in p["inputs"]] for i in range(len(test))]
        v_py = [f(*xi) for xi in x]
        # median per input of 50 timings, each averaging 100 calls (same statistic as the Julia side)
        t_py = [statistics.median(timeit.repeat(lambda xi=xi: f(*xi), number=100, repeat=50)) / 100 for xi in x]
        assert np.allclose(v_py, p["vmax_pred"], atol=1e-9), "Python evaluation differs from Julia"
        err = np.abs(np.array(vmax) - np.array(p["vmax_pred"]))
        summary = {"pisr_julia_median_us": 1e6 * p["lowlevel_median_s"],
                   "pisr_python_median_us": 1e6 * float(np.median(t_py)),
                   "pisr_julia_mlj_predict_median_us": 1e6 * p["mlj_predict_median_s"],
                   "runpp_median_ms": 1e3 * res["runpp_median_s"],
                   "speedup_python_vs_python": res["runpp_median_s"] / float(np.median(t_py)),
                   "speedup_julia_lowlevel": ratio,
                   "speedup_julia_mlj_predict": res["runpp_median_s"] / p["mlj_predict_median_s"],
                   "max_abs_vmax_error_pu": float(err.max()), "mean_abs_vmax_error_pu": float(err.mean()),
                   "environment": res["environment"], "pisr_environment": p.get("environment")}
        args.out.with_name("benchmark_summary.json").write_text(json.dumps(summary, indent=1))
        print(f"PISR in Python (same runtime as pandapower): median {summary['pisr_python_median_us']:.1f} us")
        print(f"speed-up per evaluation: {summary['speedup_python_vs_python']:.0f}x Python vs Python, "
              f"{ratio:.0f}x Julia low-level PISR vs pandapower, "
              f"{summary['speedup_julia_mlj_predict']:.0f}x Julia MLJ.predict vs pandapower")
        print(f"|Vmax| PISR vs PF on the test set: mean {err.mean():.1e}, max {err.max():.1e} p.u.")


if __name__ == "__main__":
    main()
