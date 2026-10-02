#!/usr/bin/env python3
"""
Solver-only powerflow service.

- Loads network once from an Excel file (same sheet layout as original).
- Exposes POST /solve which accepts:
    {
      "loads": [ {"bus_ext": 57, "p_mw": -0.123, "q_mvar": 0.01}, ... ],
      "breaker_closed": true   # optional
    }
  and returns JSON with complex voltages and magnitudes:
    { "voltages": {"V57_complex": "+0.999000+0.001000j", ...},
      "voltages_mag": {"V57_mag": 0.999, ...},
      "ok": true }

- No scenario loop, no internal scenario data. Intended to be called from external code (e.g. Julia).
"""
from __future__ import annotations
import argparse
import copy
import math
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pandapower as pp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Defaults (re-used from original)
REF_BUS_EXT: int = 33
KW_TO_MW = 1000.0

# Keep order used by notebook/realtime for scenario mapping so the solver API
# builds an identical network (only the scenario loads active).
DESIRED_SCENARIO_BUS_ORDER = [64, 57, 55, 53, 61]

STD_TYPES = {
    "cfg70":  {"c_nf_per_km": 0, "r_ohm_per_km": 0.268, "x_ohm_per_km": 0.0804, "max_i_ka": 0.142, "type": "cs", "q_mm2": 50, "alpha": 4.03e-3},
    "cfg95":  {"c_nf_per_km": 0, "r_ohm_per_km": 0.193, "x_ohm_per_km": 0.082309728, "max_i_ka": 0.142, "type": "cs", "q_mm2": 50, "alpha": 4.03e-3},
    "cfg16":  {"c_nf_per_km": 0, "r_ohm_per_km": 0.15,  "x_ohm_per_km": 0.092484,    "max_i_ka": 0.142, "type": "cs", "q_mm2": 50, "alpha": 4.03e-3},
    "cfg150": {"c_nf_per_km": 0, "r_ohm_per_km": 0.127, "x_ohm_per_km": 0.080,       "max_i_ka": 0.142, "type": "cs", "q_mm2": 50, "alpha": 4.03e-3},
    "cfg50":  {"c_nf_per_km": 0, "r_ohm_per_km": 0.39,  "x_ohm_per_km": 0.084915,    "max_i_ka": 0.142, "type": "cs", "q_mm2": 50, "alpha": 4.03e-3},
    "cfgx1":  {"c_nf_per_km": 0, "r_ohm_per_km": 0.127, "x_ohm_per_km": 0.08,        "max_i_ka": 0.142, "type": "cs", "q_mm2": 50, "alpha": 4.03e-3},
}

# Request models
class LoadItem(BaseModel):
    bus_ext: int
    p_mw: float
    q_mvar: float

class SolveRequest(BaseModel):
    loads: List[LoadItem]
    breaker_closed: Optional[bool] = None

# Network builder (compact adaptation of your create_net_from_excel)
def create_net_from_excel(excel_path: str) -> Tuple[pp.pandapowerNet, Dict[int,int]]:
    lines_df = pd.read_excel(excel_path, sheet_name="coses_matpower_source")
    buses_df = pd.read_excel(excel_path, sheet_name="buses")
    loads_df = pd.read_excel(excel_path, sheet_name="Loads")
    gens_df = pd.read_excel(excel_path, sheet_name="Generator")

    net = pp.create_empty_network()
    # register std types
    for name, params in STD_TYPES.items():
        if name not in net.std_types["line"]:
            pp.create_std_type(net, params, name=name, element='line')

    nodecount_to_busidx: Dict[int,int] = {}
    for idx in range(len(buses_df["Node_count"])):
        ext_id = int(buses_df["Node_count"][idx])
        name = buses_df["Node_names"][idx]
        btype = buses_df["type"][idx]
        bus_index = pp.create_bus(net, name=name, vn_kv=0.40, type=btype)
        nodecount_to_busidx[ext_id] = bus_index

    # base loads
    for idx in range(len(loads_df["bus"])):
        bus_ext = int(loads_df["bus"][idx])
        if bus_ext not in nodecount_to_busidx:
            continue
        p_mw = float(loads_df["pd"][idx]) / KW_TO_MW
        q_mvar = float(loads_df["qd"][idx]) / KW_TO_MW
        pp.create_load(net, bus=nodecount_to_busidx[bus_ext], p_mw=p_mw, q_mvar=q_mvar, name=f"base_load{idx}")

    # thevenin source (minimal reproduction)
    thevenin_anchor_ext = int(gens_df["bus"].iloc[0])
    thevenin_anchor_busidx = nodecount_to_busidx[thevenin_anchor_ext]
    # small helper
    def insert_thevenin_source(net: pp.pandapowerNet, anchor_bus: int, s_sc_target_mva: float = 30.0, x_r_ratio: float = 3.0):
        vn_lv_kv = float(net.bus.at[anchor_bus, "vn_kv"])
        vn_hv_kv = 20.0
        hv_bus = pp.create_bus(net, name="Thevenin HV Bus", vn_kv=vn_hv_kv, type="b")
        z_th = (vn_hv_kv ** 2) / s_sc_target_mva
        r_th = z_th / math.sqrt(1.0 + x_r_ratio ** 2)
        x_th = r_th * x_r_ratio
        ext_bus = pp.create_bus(net, name="External Grid Bus", vn_kv=vn_hv_kv, type="b")
        pp.create_ext_grid(net, bus=ext_bus, vm_pu=1.0, va_degree=0.0,
                          s_sc_max_mva=1e9, s_sc_min_mva=1e9, rx_min=0.0, rx_max=0.0, name="Thevenin External Grid 20kV")
        pp.create_line_from_parameters(net, name="Thevenin Line 20kV", from_bus=ext_bus, to_bus=hv_bus,
                                       length_km=1.0, r_ohm_per_km=r_th, x_ohm_per_km=x_th, c_nf_per_km=0.0, max_i_ka=10.0, type="ol")
        pp.create_transformer_from_parameters(net, hv_bus=hv_bus, lv_bus=anchor_bus, sn_mva=0.25, vn_hv_kv=vn_hv_kv, vn_lv_kv=vn_lv_kv,
                                              vk_percent=4.0, vkr_percent=0.5, pfe_kw=0.5, i0_percent=0.2, shift_degree=0, vector_group="Dyn", name="Thevenin Trafo 250kVA 4%")
    insert_thevenin_source(net, thevenin_anchor_busidx, 30.0, 3.0)

    # lines (safe-guard) -- limit to same range as notebook/realtime
    for idx in range(min(len(lines_df), 61)):
        f_ext = int(lines_df["f_bus_id"][idx])
        t_ext = int(lines_df["t_bus_id"][idx])
        if f_ext not in nodecount_to_busidx or t_ext not in nodecount_to_busidx:
            continue
        length_km = float(lines_df["length_km"][idx])
        cfg = str(lines_df["config"][idx])
        if cfg not in net.std_types["line"]:
            cfg = "cfg70"
        pp.create_line(net, name=f"line_{idx}", from_bus=nodecount_to_busidx[f_ext], to_bus=nodecount_to_busidx[t_ext], length_km=length_km, std_type=cfg)

    return net, nodecount_to_busidx

# Map external bus -> first load index on that bus
def resolve_load_indices(net: pp.pandapowerNet) -> Dict[int,int]:
    busidx_to_ext = {v:k for k,v in net._bus_name_index.items()} if hasattr(net, '_bus_name_index') else {}
    # safer approach: build mapping from load rows
    mapping: Dict[int,int] = {}
    # first find ext id mapping by checking 'name' or store created mapping externally
    # We'll rely on the net.load.bus -> bus index, then user will pass ext ids matching net.bus.index names (we keep reverse from create)
    for load_idx in list(net.load.index):
        bus_idx = int(net.load.at[load_idx, 'bus'])
        # try to use net.bus.name or 'name' column is not guaranteed --> instead expecting caller to use the same ext ids as used to build network
        # We'll build busidx->ext mapping below in server initialization (from create_net_from_excel return)
        mapping.setdefault(bus_idx, load_idx)
    return mapping

# Helper to format complex as string "+a.bbb+ c.dddj"
def c2s(z: complex) -> str:
    return f"{z.real:+.9f}{z.imag:+.9f}j"

def build_app(net_template: pp.pandapowerNet, nodecount_to_busidx: Dict[int,int], k4_sw_idx: Optional[int]):
    app = FastAPI(title="Powerflow Solver", version="0.1")

    # reverse mapping
    busidx_to_ext = {v:k for k,v in nodecount_to_busidx.items()}

    # working net used per-request (deepcopy inside request)
    @app.post("/solve")
    def solve(req: SolveRequest):
        # shallow validation
        if not isinstance(req.loads, list):
            raise HTTPException(status_code=400, detail="Missing loads list")
        # create working copy
        net = copy.deepcopy(net_template)
        # if breaker provided, apply to working net if switch exists
        if req.breaker_closed is not None and k4_sw_idx is not None:
            try:
                if 'switch' in net and k4_sw_idx in net.switch.index:
                    net.switch.at[k4_sw_idx, 'closed'] = bool(req.breaker_closed)
            except Exception:
                pass

        # Build mapping ext->list of load indices on that bus (only loads in_service)
        ext_to_load_idxs: Dict[int, List[int]] = {}
        for li in list(net.load.index):
            try:
                in_service = bool(net.load.at[li, 'in_service'])
            except Exception:
                in_service = True
            if not in_service:
                continue
            bus_idx = int(net.load.at[li, 'bus'])
            ext = busidx_to_ext.get(bus_idx)
            if ext is not None:
                ext_to_load_idxs.setdefault(ext, []).append(li)

        # Apply provided loads (replace p_mw/q_mvar for all loads on that ext)
        for it in req.loads:
            ext = int(it.bus_ext)
            if ext not in nodecount_to_busidx:
                raise HTTPException(status_code=400, detail=f"Unknown bus_ext: {ext}")
            load_idxs = ext_to_load_idxs.get(ext, [])
            if not load_idxs:
                # if no base load exists, create one at that bus
                bus_idx = nodecount_to_busidx[ext]
                new_idx = pp.create_load(net, bus=bus_idx, p_mw=float(it.p_mw), q_mvar=float(it.q_mvar), name=f"dynamic_load_{ext}")
                load_idxs = [new_idx]
            for li in load_idxs:
                net.load.at[li, 'p_mw'] = float(it.p_mw)
                net.load.at[li, 'q_mvar'] = float(it.q_mvar)

        # run pf
        try:
            # Match realtime notebook defaults: faster iterations and tight tolerance when using numba
            pp.runpp(net, calculate_voltage_angles=True, init="flat", algorithm="nr", enforce_q_lims=True, max_iteration=30, tolerance_mva=1e-7, numba=True)
        except Exception:
            try:
                # Fallback to looser settings without numba
                pp.runpp(net, calculate_voltage_angles=True, init="dc", algorithm="nr", enforce_q_lims=False, max_iteration=50, tolerance_mva=1e-6, numba=False)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"powerflow failed: {e}")

        # collect voltages relative to REF bus angle
        ref_rad = 0.0
        if REF_BUS_EXT in nodecount_to_busidx:
            ref_idx = nodecount_to_busidx[REF_BUS_EXT]
            if ref_idx in net.res_bus.index:
                ref_deg = float(net.res_bus.at[ref_idx, 'va_degree'])
                ref_rad = math.radians(ref_deg)

        voltages = {}
        voltages_mag = {}
        for ext, bus_idx in nodecount_to_busidx.items():
            if bus_idx in net.res_bus.index:
                vm = float(net.res_bus.at[bus_idx, 'vm_pu'])
                va_deg = float(net.res_bus.at[bus_idx, 'va_degree'])
                if not (np.isfinite(vm) and np.isfinite(va_deg)):
                    continue
                va_rad = math.radians(va_deg) - ref_rad
                v_complex = vm * (math.cos(va_rad) + 1j * math.sin(va_rad))
                voltages[f"V{ext}_complex"] = c2s(v_complex)
                voltages_mag[f"V{ext}_mag"] = float(abs(v_complex))

        return {"ok": True, "voltages": voltages, "voltages_mag": voltages_mag}

    @app.get("/info")
    def info():
        return {"buses": sorted(list(nodecount_to_busidx.keys())), "ref_bus_ext": REF_BUS_EXT, "breaker_present": k4_sw_idx is not None}

    return app

def find_k4_switch_idx(net: pp.pandapowerNet, nodecount_to_busidx: Dict[int,int]) -> Optional[int]:
    # try to find line between ext buses 7 and 8 as in original and create a switch in template net if present
    try:
        bus7 = nodecount_to_busidx.get(7)
        bus8 = nodecount_to_busidx.get(8)
        if bus7 is None or bus8 is None or len(net.line.index) == 0:
            return None
        for li in list(net.line.index):
            fb = int(net.line.at[li, 'from_bus'])
            tb = int(net.line.at[li, 'to_bus'])
            if (fb == bus7 and tb == bus8) or (fb == bus8 and tb == bus7):
                # create a switch on bus7 side
                return pp.create_switch(net, bus=bus7, element=int(li), et='l', type='CB', closed=True, name='K4 breaker')
    except Exception:
        return None
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", default=os.path.join("data", "grid", "coses_matpower_source.xlsx"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    if not os.path.isfile(args.excel):
        raise FileNotFoundError(f"Excel not found: {args.excel}")

    net_template, nodecount_to_busidx = create_net_from_excel(args.excel)
    # find/create K4 switch index (template)
    k4_sw_idx = find_k4_switch_idx(net_template, nodecount_to_busidx)

    app = build_app(net_template, nodecount_to_busidx, k4_sw_idx)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()