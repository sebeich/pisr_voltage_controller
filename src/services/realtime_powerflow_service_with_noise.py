#!/usr/bin/env python3
"""
Real-time powerflow service with optional Gaussian noise on voltages and powers.

Based on realtime_powerflow_service_breaker.py, with added:
- Noise config (GET/POST /noise) controlling additive Gaussian noise applied to reported voltages and powers
- Scenario freeze toggle (keeps row index fixed like the old service)
- Reused richer Plotly UI with a working Pause button and an extra Noise section

Voltage noise (v_noise_std) applies to real and imag parts (p.u.).
Power noise (s_noise_std            <div style="margin-top:10px; display:flex; gap:8px; flex-wrap: wrap;">
                <button id="apply" class="primary">Apply Mod</button>
                <button id="clearOne">Set 0 for Selected</button>
                <button id="clearAll">Clear All Mods</button>
            </div>

            <div class="fieldset">
                <h4 style="margin:6px 0 4px;">PISR Controller</h4>
                <div class="row" style="gap:8px; flex-wrap:wrap;">
                    <button id="pisrToggleBtn">Toggle PISR</button>
                    <span class="small">PISR Status: <strong id="pisrStatus">unknown</strong></span>
                </div>
            </div>

            <div class="fieldset grid2">
                <div>
                    <div class="small">Row: <span id="rowidx">-</span></div>
                    <div class="small">PF Rate: <span id="pf">-</span> Hz</div>to P (MW) and Q (MVAr) independently.
"""
from __future__ import annotations
import argparse
import copy
import math
import os
import threading
import time
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pandapower as pp
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# -------------------- Config defaults --------------------
DESIRED_SCENARIO_BUS_ORDER: List[int] = [64, 57, 55, 61]
JULIA_CONTROLLED_BUSES: List[int] = [55, 57, 64]
REF_BUS_EXT: int = 33
KW_TO_MW = 1000.0

STD_TYPES = {
    "cfg70":  {"c_nf_per_km": 0, "r_ohm_per_km": 0.268, "x_ohm_per_km": 0.0804,      "max_i_ka": 0.142, "type": "cs", "q_mm2": 50, "alpha": 4.03e-3},
    "cfg95":  {"c_nf_per_km": 0, "r_ohm_per_km": 0.193, "x_ohm_per_km": 0.082309728, "max_i_ka": 0.142, "type": "cs", "q_mm2": 50, "alpha": 4.03e-3},
    "cfg16":  {"c_nf_per_km": 0, "r_ohm_per_km": 0.15,  "x_ohm_per_km": 0.092484,    "max_i_ka": 0.142, "type": "cs", "q_mm2": 50, "alpha": 4.03e-3},
    "cfg150": {"c_nf_per_km": 0, "r_ohm_per_km": 0.127, "x_ohm_per_km": 0.080,       "max_i_ka": 0.142, "type": "cs", "q_mm2": 50, "alpha": 4.03e-3},
    "cfg50":  {"c_nf_per_km": 0, "r_ohm_per_km": 0.39,  "x_ohm_per_km": 0.084915,    "max_i_ka": 0.142, "type": "cs", "q_mm2": 50, "alpha": 4.03e-3},
    "cfgx1":  {"c_nf_per_km": 0, "r_ohm_per_km": 0.127, "x_ohm_per_km": 0.08,        "max_i_ka": 0.142, "type": "cs", "q_mm2": 50, "alpha": 4.03e-3},
}

# -------------------- Models --------------------
class ModItem(BaseModel):
    bus_ext: int
    dP_mw: float = 0.0
    dQ_mvar: float = 0.0


class ModBatch(BaseModel):
    items: List[ModItem]


class NoiseConfig(BaseModel):
    enabled: Optional[bool] = None
    v_noise_std: Optional[float] = None
    s_noise_std: Optional[float] = None


class SolverConfig(BaseModel):
    mode: str


class CostUpdate(BaseModel):
    source: str
    value: Optional[float] = None
    detail: Optional[Dict[str, Any]] = None


# -------------------- Build network --------------------
def create_net_from_excel(excel_path: str) -> Tuple[pp.pandapowerNet, Dict[int, int], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lines_df = pd.read_excel(excel_path, sheet_name="coses_matpower_source")
    buses_df = pd.read_excel(excel_path, sheet_name="buses")
    loads_df = pd.read_excel(excel_path, sheet_name="Loads")
    gens_df = pd.read_excel(excel_path, sheet_name="Generator")

    net = pp.create_empty_network()
    for name, params in STD_TYPES.items():
        if name not in net.std_types["line"]:
            pp.create_std_type(net, params, name=name, element='line')

    nodecount_to_busidx: Dict[int, int] = {}
    for idx in range(len(buses_df["Node_count"])):
        ext_id = int(buses_df["Node_count"][idx])
        name = buses_df["Node_names"][idx]
        btype = buses_df["type"][idx]
        bus_index = pp.create_bus(net, name=name, vn_kv=0.40, type=btype)
        nodecount_to_busidx[ext_id] = bus_index

    for idx in range(len(loads_df["bus"])):
        bus_ext = int(loads_df["bus"][idx])
        if bus_ext not in nodecount_to_busidx:
            continue
        p_mw = float(loads_df["pd"][idx]) / KW_TO_MW
        q_mvar = float(loads_df["qd"][idx]) / KW_TO_MW
        pp.create_load(net, bus=nodecount_to_busidx[bus_ext], p_mw=p_mw, q_mvar=q_mvar, name=f"base_load{idx}")
        q_mvar = float(loads_df["qd"][idx]) / KW_TO_MW
        pp.create_load(net, bus=nodecount_to_busidx[bus_ext], p_mw=p_mw, q_mvar=q_mvar, name=f"base_load{idx}")

    def insert_thevenin_source(net: pp.pandapowerNet, anchor_bus: int, s_sc_target_mva: float = 30.0, x_r_ratio: float = 3.0):
        vn_lv_kv = float(net.bus.at[anchor_bus, "vn_kv"])  # likely 0.4kV
        vn_hv_kv = 20.0
        hv_bus = pp.create_bus(net, name="Thevenin HV Bus", vn_kv=vn_hv_kv, type="b")
        z_th = (vn_hv_kv ** 2) / s_sc_target_mva
        r_th = z_th / math.sqrt(1.0 + x_r_ratio ** 2)
        x_th = r_th * x_r_ratio
        ext_bus = pp.create_bus(net, name="External Grid Bus", vn_kv=vn_hv_kv, type="b")
        pp.create_ext_grid(net, bus=ext_bus, vm_pu=1.0, va_degree=0.0, s_sc_max_mva=1e9, s_sc_min_mva=1e9, rx_min=0.0, rx_max=0.0, name="Thevenin External Grid 20kV")
        pp.create_line_from_parameters(net, name="Thevenin Line 20kV", from_bus=ext_bus, to_bus=hv_bus,
                                       length_km=1.0, r_ohm_per_km=r_th, x_ohm_per_km=x_th, c_nf_per_km=0.0, max_i_ka=10.0, type="ol")
        pp.create_transformer_from_parameters(net, hv_bus=hv_bus, lv_bus=anchor_bus, sn_mva=0.25, vn_hv_kv=vn_hv_kv, vn_lv_kv=vn_lv_kv,
                                              vk_percent=4.0, vkr_percent=0.5, pfe_kw=0.5, i0_percent=0.2, shift_degree=0, vector_group="Dyn", name="Thevenin Trafo 250kVA 4%")

    thevenin_anchor_ext = int(pd.read_excel(excel_path, sheet_name="Generator")["bus"].iloc[0])
    thevenin_anchor_busidx = nodecount_to_busidx[thevenin_anchor_ext]
    insert_thevenin_source(net, thevenin_anchor_busidx, 30.0, 3.0)

    max_lines = min(61, len(lines_df))
    for idx in range(max_lines):
        f_ext = int(lines_df["f_bus_id"][idx])
        t_ext = int(lines_df["t_bus_id"][idx])
        if f_ext not in nodecount_to_busidx or t_ext not in nodecount_to_busidx:
            continue
        length_km = float(lines_df["length_km"][idx])
        cfg = str(lines_df["config"][idx])
        if cfg not in net.std_types["line"]:
            cfg = "cfg70"
        pp.create_line(net, name=f"line_{idx}", from_bus=nodecount_to_busidx[f_ext], to_bus=nodecount_to_busidx[t_ext], length_km=length_km, std_type=cfg)

    return net, nodecount_to_busidx, lines_df, buses_df, loads_df, gens_df


# -------------------- Real-time runner --------------------
class RealTimePowerflow:
    def __init__(self, excel_path: str, scenarios_csv: str, pf_rate_hz: float = 50.0, scenario_rate_hz: float = 1.0, v_noise_std: float = 0.0, s_noise_std: float = 0.0, rng_seed: Optional[int] = None):
        self.excel_path = excel_path
        self.scenarios_csv = scenarios_csv
        self.pf_rate_hz = float(pf_rate_hz)
        self.scenario_rate_hz = float(scenario_rate_hz)

        # noise configuration
        self.v_noise_std = float(v_noise_std)
        self.s_noise_std = float(s_noise_std)
        self.rng = np.random.default_rng(rng_seed)
        self.noise_enabled = bool((self.v_noise_std != 0.0) or (self.s_noise_std != 0.0))

        # scenario freeze (True = only row 0)
        self.scenario_frozen = False

        # Build base network and scenario DF
        self.net_template, self.nodecount_to_busidx, *_ = create_net_from_excel(excel_path)
        self.busidx_to_ext = {v: k for k, v in self.nodecount_to_busidx.items()}

        # --- Insert K4 breaker (between buses 7 and 8 if present) ---
        self.k4_sw_idx = None
        self.k4_closed = False
        try:
            bus7 = self.nodecount_to_busidx.get(7)
            bus8 = self.nodecount_to_busidx.get(8)
            if bus7 is not None and bus8 is not None and len(self.net_template.line.index):
                k4_line_idx = None
                for li in list(self.net_template.line.index):
                    fb = int(self.net_template.line.at[li, 'from_bus'])
                    tb = int(self.net_template.line.at[li, 'to_bus'])
                    if (fb == bus7 and tb == bus8) or (fb == bus8 and tb == bus7):
                        k4_line_idx = li
                        break
                if k4_line_idx is not None:
                    self.k4_sw_idx = pp.create_switch(self.net_template, bus=bus7, element=int(k4_line_idx), et='l', type='CB', closed=True, name='K4 breaker')
        except Exception:
            self.k4_sw_idx = None

        self.scenarios_df = pd.read_csv(scenarios_csv)
        self.scenario_load_indices = self._resolve_load_indices(self.net_template, DESIRED_SCENARIO_BUS_ORDER)
        self.ext_to_load_idx = {ext: load_idx for ext, load_idx in zip(DESIRED_SCENARIO_BUS_ORDER, self.scenario_load_indices)}
        self.manual_controlled_buses: List[int] = [ext for ext in DESIRED_SCENARIO_BUS_ORDER if ext in self.ext_to_load_idx]
        self.controlled_buses: List[int] = [ext for ext in JULIA_CONTROLLED_BUSES if ext in self.ext_to_load_idx]
        # Supervise OPF voltage on scenario load nodes (controlled + uncontrolled disturbances),
        # matching the Julia setup scope.
        self.opf_supervised_buses: List[int] = list(self.manual_controlled_buses)

        # Shared state
        self.lock = threading.RLock()
        self.current_row_idx = 0
        self.current_S_targets: Dict[int, complex] = {ext: complex(0.0, 0.0) for ext in DESIRED_SCENARIO_BUS_ORDER}
        self.mods: Dict[int, Tuple[float, float]] = {}
        self.opf_mods: Dict[int, Tuple[float, float]] = {}
        self.last_voltages: Dict[int, complex] = {}
        self.last_powers: Dict[int, complex] = {}
        self.solver_mode: str = "pf"
        self.opf_p_weight: float = 30.0
        self.opf_q_weight: float = 1.0
        self.opf_p_limit_mw: float = 0.3
        self.opf_q_limit_mvar: float = 0.1
        self.opf_v_limit_pu: float = 1.05
        self.opf_v_min_pu: float = 0.90
        self.opf_v_track_band_pu: float = float(os.getenv("OPF_V_TRACK_BAND_PU", "0.03"))
        self.last_opf: Dict[str, Any] = {
            "ok": False,
            "objective_total": None,
            "objective_active": None,
            "objective_reactive": None,
            "max_overvoltage_pre_pu": None,
            "max_vm_pre_pu": None,
            "max_overvoltage_pu": None,
            "solve_time_ms": None,
            "message": "OPF not run yet",
        }
        self.last_cost: Dict[str, Any] = {
            "source": "none",
            "value": None,
            "timestamp": None,
            "detail": {},
        }

        self._stop = threading.Event()
        self.net_working = copy.deepcopy(self.net_template)
        self._thr_scen = threading.Thread(target=self._scenario_loop, name="scenario_loop", daemon=True)
        self._thr_pf = threading.Thread(target=self._pf_loop, name="pf_loop", daemon=True)

    def _resolve_load_indices(self, net: pp.pandapowerNet, desired_exts: List[int]) -> List[int]:
        kept: Dict[int, int] = {}
        for load_idx in list(net.load.index):
            bus_idx = int(net.load.at[load_idx, 'bus'])
            ext = self.busidx_to_ext[bus_idx]
            if ext in desired_exts and ext not in kept:
                kept[ext] = load_idx
        return [kept[ext] for ext in desired_exts if ext in kept]

    # ---------- loops ----------
    def start(self):
        self._thr_scen.start()
        self._thr_pf.start()

    def stop(self):
        self._stop.set()
        for t in (self._thr_scen, self._thr_pf):
            if t.is_alive():
                t.join(timeout=1.0)

    def _scenario_loop(self):
        period = 1.0 / max(1e-6, self.scenario_rate_hz)
        while not self._stop.is_set():
            t0 = time.perf_counter()
            with self.lock:
                n_rows = 1 if self.scenario_frozen else len(self.scenarios_df)
                ridx = self.current_row_idx % max(1, n_rows)
            new_targets: Dict[int, complex] = {}
            for j, ext in enumerate(DESIRED_SCENARIO_BUS_ORDER, start=1):
                p_col = f"P_node{j}"
                q_col = f"Q_node{j}"
                if p_col not in self.scenarios_df.columns or q_col not in self.scenarios_df.columns:
                    raise RuntimeError(f"Scenario columns missing: {p_col}, {q_col}")
                p_mw = float(self.scenarios_df.at[ridx, p_col]) / -1000.0
                q_mvar = float(self.scenarios_df.at[ridx, q_col]) / -1000.0
                new_targets[ext] = complex(p_mw, q_mvar)
            with self.lock:
                self.current_S_targets = new_targets
                self.current_row_idx = (self.current_row_idx + 1) % max(1, len(self.scenarios_df))
            dt = time.perf_counter() - t0
            time.sleep(max(0.0, period - dt))

    def _pf_loop(self):
        period = 1.0 / max(1e-6, self.pf_rate_hz)
        while not self._stop.is_set():
            t0 = time.perf_counter()
            with self.lock:
                S_targets = self.current_S_targets.copy()
                manual_mods = self.mods.copy()
                opf_mods = self.opf_mods.copy()
                solver_mode = str(self.solver_mode)
                net = self.net_working
                try:
                    if self.k4_sw_idx is not None and 'switch' in net and self.k4_sw_idx in net.switch.index:
                        net.switch.at[self.k4_sw_idx, 'closed'] = bool(self.k4_closed)
                except Exception:
                    pass
            # Set controlled loads depending on mode:
            # - PF: baseline targets + manual user mods
            # - OPF: OPF-controlled buses use baseline only (optimized internally);
            #        non-OPF buses keep manual disturbances active.
            for ext, load_idx in zip(DESIRED_SCENARIO_BUS_ORDER, self.scenario_load_indices):
                s = S_targets.get(ext, complex(0.0, 0.0))
                if solver_mode == "opf" and ext in self.controlled_buses:
                    dP, dQ = (0.0, 0.0)
                else:
                    dP, dQ = manual_mods.get(ext, (0.0, 0.0))
                net.load.at[load_idx, 'p_mw'] = float(s.real + dP)
                net.load.at[load_idx, 'q_mvar'] = float(s.imag + dQ)

            opf_info: Optional[Dict[str, Any]] = None
            if solver_mode == "opf":
                opf_info = self._run_opf(net, S_targets)
                if opf_info.get("ok"):
                    opf_mods: Dict[int, Tuple[float, float]] = {}
                    for ext in self.controlled_buses:
                        load_idx = self.ext_to_load_idx.get(ext)
                        if load_idx is None:
                            continue
                        pref = float(S_targets.get(ext, 0.0 + 0.0j).real)
                        qref = float(S_targets.get(ext, 0.0 + 0.0j).imag)
                        p_new = float(net.res_load.at[load_idx, 'p_mw']) if load_idx in net.res_load.index else pref
                        q_new = float(net.res_load.at[load_idx, 'q_mvar']) if load_idx in net.res_load.index else qref
                        # PV physical limit: cannot become net consumer (p > 0)
                        d_p_max = float(max(0.0, -pref))
                        d_p = float(np.clip(p_new - pref, 0.0, d_p_max))
                        d_q = float(np.clip(q_new - qref, -self.opf_q_limit_mvar, self.opf_q_limit_mvar))
                        opf_mods[ext] = (d_p, d_q)
                    # Use fresh OPF actions immediately for this snapshot cycle.
                    active_opf_mods = dict(opf_mods)
                    with self.lock:
                        self.opf_mods = dict(active_opf_mods)
                        self.last_opf = opf_info
                        self._set_cost_locked(
                            source="opf",
                            value=opf_info.get("objective_total"),
                            detail={
                                "mode": "opf",
                                "objective_active": opf_info.get("objective_active"),
                                "objective_reactive": opf_info.get("objective_reactive"),
                                "max_overvoltage_pu": opf_info.get("max_overvoltage_pu"),
                                "solve_time_ms": opf_info.get("solve_time_ms"),
                            },
                        )
                else:
                    active_opf_mods = {}
                    with self.lock:
                        self.opf_mods.clear()
                        self.last_opf = opf_info
            else:
                active_opf_mods = {}
                self._run_powerflow(net)

            # Collect voltages relative to REF bus angle
            ref_rad = 0.0
            if REF_BUS_EXT in self.nodecount_to_busidx:
                ref_idx = self.nodecount_to_busidx[REF_BUS_EXT]
                if ref_idx in net.res_bus.index:
                    ref_deg = float(net.res_bus.at[ref_idx, 'va_degree'])
                    ref_rad = math.radians(ref_deg)
            voltages: Dict[int, complex] = {}
            for ext, bus_idx in self.nodecount_to_busidx.items():
                if bus_idx in net.res_bus.index:
                    vm = float(net.res_bus.at[bus_idx, 'vm_pu'])
                    va_deg = float(net.res_bus.at[bus_idx, 'va_degree'])
                    if not (np.isfinite(vm) and np.isfinite(va_deg)):
                        continue
                    va_rad = math.radians(va_deg) - ref_rad
                    v_complex = vm * (math.cos(va_rad) + 1j * math.sin(va_rad))
                    voltages[ext] = v_complex

            powers: Dict[int, complex] = {}
            for ext in DESIRED_SCENARIO_BUS_ORDER:
                s = S_targets.get(ext, complex(0.0, 0.0))
                if solver_mode == "opf":
                    if ext in self.controlled_buses:
                        dP, dQ = active_opf_mods.get(ext, (0.0, 0.0))
                    else:
                        dP, dQ = manual_mods.get(ext, (0.0, 0.0))
                else:
                    dP, dQ = manual_mods.get(ext, (0.0, 0.0))
                powers[ext] = complex(s.real + dP, s.imag + dQ)

            with self.lock:
                self.last_voltages = voltages
                self.last_powers = powers

            dt = time.perf_counter() - t0
            time.sleep(max(0.0, period - dt))

    # ---------- helpers ----------
    def set_mod(self, bus_ext: int, dP_mw: float, dQ_mvar: float):
        if bus_ext not in self.manual_controlled_buses:
            raise ValueError(f"Bus {bus_ext} is not in manual controlled buses {self.manual_controlled_buses}")
        with self.lock:
            self.mods[bus_ext] = (float(dP_mw), float(dQ_mvar))

    def clear_mods(self):
        with self.lock:
            self.mods.clear()

    def set_breaker_closed(self, closed: bool):
        with self.lock:
            self.k4_closed = bool(closed)
            try:
                if self.k4_sw_idx is not None and self.k4_sw_idx in self.net_working.switch.index:
                    self.net_working.switch.at[self.k4_sw_idx, 'closed'] = self.k4_closed
            except Exception:
                pass

    def toggle_breaker(self) -> bool:
        with self.lock:
            self.k4_closed = not bool(self.k4_closed)
            try:
                if self.k4_sw_idx is not None and self.k4_sw_idx in self.net_working.switch.index:
                    self.net_working.switch.at[self.k4_sw_idx, 'closed'] = self.k4_closed
            except Exception:
                pass
            return self.k4_closed

    def set_scenario_frozen(self, frozen: bool):
        with self.lock:
            self.scenario_frozen = bool(frozen)

    def get_solver_state(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "mode": str(self.solver_mode),
                "opf": dict(self.last_opf),
                "controlled_buses": list(self.controlled_buses),
                "manual_controlled_buses": list(self.manual_controlled_buses),
                "supervised_buses": list(self.opf_supervised_buses),
                "weights": {
                    "p_weight": float(self.opf_p_weight),
                    "q_weight": float(self.opf_q_weight),
                },
            }

    def set_solver_mode(self, mode: str) -> str:
        mode_n = str(mode).strip().lower()
        if mode_n not in ("pf", "opf"):
            raise ValueError("mode must be 'pf' or 'opf'")
        with self.lock:
            self.solver_mode = mode_n
            if self.solver_mode != "opf":
                self.opf_mods.clear()
            return self.solver_mode

    def toggle_solver_mode(self) -> str:
        with self.lock:
            self.solver_mode = "opf" if self.solver_mode == "pf" else "pf"
            if self.solver_mode != "opf":
                self.opf_mods.clear()
            return self.solver_mode

    def toggle_scenario_frozen(self) -> bool:
        with self.lock:
            self.scenario_frozen = not bool(self.scenario_frozen)
            return self.scenario_frozen

    def _set_cost_locked(self, source: str, value: Optional[float], detail: Optional[Dict[str, Any]] = None):
        self.last_cost = {
            "source": str(source),
            "value": (float(value) if value is not None else None),
            "timestamp": time.time(),
            "detail": dict(detail or {}),
        }

    def set_cost(self, source: str, value: Optional[float], detail: Optional[Dict[str, Any]] = None):
        source_n = str(source).strip().lower()
        if not source_n:
            raise ValueError("source must be non-empty")
        with self.lock:
            self._set_cost_locked(source_n, value, detail)

    def get_cost(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.last_cost)

    def _run_powerflow(self, net: pp.pandapowerNet):
        try:
            pp.runpp(net, calculate_voltage_angles=True, init="flat", algorithm="nr", enforce_q_lims=True, max_iteration=30, tolerance_mva=1e-7, numba=True)
        except Exception:
            try:
                pp.runpp(net, calculate_voltage_angles=True, init="dc", algorithm="nr", enforce_q_lims=False, max_iteration=50, tolerance_mva=1e-6, numba=False)
            except Exception:
                pass

    def _run_opf(self, net: pp.pandapowerNet, S_targets: Dict[int, complex]) -> Dict[str, Any]:
            t0 = time.perf_counter()
            try:
                # Baseline PF on the current operating point.
                self._run_powerflow(net)

                supervised_busidx_pre = []
                for ext in self.opf_supervised_buses:
                    bi = self.nodecount_to_busidx.get(ext)
                    if bi is not None and bi in net.res_bus.index:
                        supervised_busidx_pre.append(int(bi))
                if supervised_busidx_pre:
                    max_vm_pre = float(np.nanmax(net.res_bus.loc[supervised_busidx_pre, "vm_pu"]))
                else:
                    max_vm_pre = float(np.nanmax(net.res_bus["vm_pu"])) if len(net.res_bus.index) else 0.0
                max_overvoltage_pre = max(0.0, max_vm_pre - float(self.opf_v_limit_pu))
                inactive_voltage = max_overvoltage_pre <= 1e-9

                if "poly_cost" in net and len(net.poly_cost.index):
                    net.poly_cost.drop(net.poly_cost.index, inplace=True)
                if "pwl_cost" in net and len(net.pwl_cost.index):
                    net.pwl_cost.drop(net.pwl_cost.index, inplace=True)

                net.load["controllable"] = False

                for ext in self.controlled_buses:
                    load_idx = self.ext_to_load_idx.get(ext)
                    if load_idx is None:
                        continue
                    li = int(load_idx)
                    if li not in net.load.index:
                        continue
                    s_ref = complex(S_targets.get(ext, 0.0 + 0.0j))
                    p_ref = float(s_ref.real)
                    q_ref = float(s_ref.imag)

                    net.load.at[li, "controllable"] = True
                    # When already below voltage limit, force zero-modification optimum with a
                    # tiny epsilon box around the reference while still running OPF (for timing).
                    if inactive_voltage:
                        eps = 1e-6
                        p_min = p_ref
                        p_max = min(0.0, p_ref + eps)
                        if p_max < p_min:
                            p_max = p_min
                        q_min = q_ref - eps
                        q_max = q_ref + eps
                    else:
                        # Bounds are deviations around current operating point:
                        # dP in [0, PLIMIT] mapped to p in [p_ref, min(0, p_ref+PLIMIT)]
                        # dQ in [-QLIMIT, QLIMIT] mapped to q in [q_ref-QLIMIT, q_ref+QLIMIT]
                        p_min = p_ref
                        p_max = min(0.0, p_ref + float(self.opf_p_limit_mw))
                        if p_max < p_min:
                            p_max = p_min
                        q_min = q_ref - self.opf_q_limit_mvar
                        q_max = q_ref + self.opf_q_limit_mvar

                    net.load.at[li, "min_p_mw"] = p_min
                    net.load.at[li, "max_p_mw"] = p_max
                    net.load.at[li, "min_q_mvar"] = q_min
                    net.load.at[li, "max_q_mvar"] = q_max

                    cp2 = float(self.opf_p_weight)
                    cp1 = float(-2.0 * self.opf_p_weight * p_ref)
                    cq2 = float(self.opf_q_weight)
                    cq1 = float(-2.0 * self.opf_q_weight * q_ref)
                    pp.create_poly_cost(
                        net,
                        li,
                        "load",
                        cp1_eur_per_mw=cp1,
                        cp2_eur_per_mw2=cp2,
                        cq1_eur_per_mvar=cq1,
                        cq2_eur_per_mvar2=cq2,
                    )

                if "ext_grid" in net and len(net.ext_grid.index):
                    for col, default in (("min_p_mw", -1e3), ("max_p_mw", 1e3), ("min_q_mvar", -1e3), ("max_q_mvar", 1e3)):
                        if col not in net.ext_grid.columns:
                            net.ext_grid[col] = default
                        else:
                            net.ext_grid[col] = net.ext_grid[col].fillna(default)

                net.bus["max_vm_pu"] = 1.20
                net.bus["min_vm_pu"] = float(self.opf_v_min_pu)
                if inactive_voltage:
                    active_min_vm = float(self.opf_v_min_pu)
                else:
                    # Adaptive tracking: keep close to 1.03 for mild violations, relax for severe violations.
                    # - max band for strong violations (feasibility)
                    # - tight band for near-threshold violations (avoid over-droop)
                    vmax_ref = 0.04  # ~1.07 -> 1.03 reference excess
                    band_max = float(self.opf_v_track_band_pu)
                    band_min = min(0.002, band_max)
                    severity = max(0.0, min(1.0, float(max_overvoltage_pre) / vmax_ref))
                    track_band = max(band_min, min(band_max, band_max * severity))
                    active_min_vm = max(float(self.opf_v_min_pu), float(self.opf_v_limit_pu) - track_band)
                supervised_busidx = []
                for ext in self.opf_supervised_buses:
                    bi = self.nodecount_to_busidx.get(ext)
                    if bi is not None and bi in net.bus.index:
                        net.bus.at[bi, "max_vm_pu"] = float(self.opf_v_limit_pu)
                        if (not inactive_voltage) and (ext in self.controlled_buses):
                            net.bus.at[bi, "min_vm_pu"] = active_min_vm
                        else:
                            net.bus.at[bi, "min_vm_pu"] = float(self.opf_v_min_pu)
                        supervised_busidx.append(int(bi))

                pp.runopp(net, verbose=False, calculate_voltage_angles=True, delta=1e-16)

                objective_active = 0.0
                objective_reactive = 0.0
                for ext in self.controlled_buses:
                    li = self.ext_to_load_idx.get(ext)
                    if li is None or li not in net.load.index or li not in net.res_load.index:
                        continue
                    s_ref = complex(S_targets.get(ext, 0.0 + 0.0j))
                    p_ref = float(s_ref.real)
                    q_ref = float(s_ref.imag)
                    p_new = float(net.res_load.at[li, "p_mw"])
                    q_new = float(net.res_load.at[li, "q_mvar"])
                    objective_active += float(self.opf_p_weight) * (p_new - p_ref) ** 2
                    objective_reactive += float(self.opf_q_weight) * (q_new - q_ref) ** 2
                objective_total = objective_active + objective_reactive
                if abs(objective_active) < 1e-9:
                    objective_active = 0.0
                if abs(objective_reactive) < 1e-9:
                    objective_reactive = 0.0
                if abs(objective_total) < 1e-9:
                    objective_total = 0.0

                if supervised_busidx:
                    max_vm = float(np.nanmax(net.res_bus.loc[supervised_busidx, "vm_pu"]))
                else:
                    max_vm = float(np.nanmax(net.res_bus["vm_pu"])) if len(net.res_bus.index) else 0.0
                max_overvoltage = max(0.0, max_vm - float(self.opf_v_limit_pu))
                solve_ms = (time.perf_counter() - t0) * 1000.0

                print(f"[OPF] objective_total={objective_total:.6f} objective_active={objective_active:.6f} objective_reactive={objective_reactive:.6f} max_overvoltage_pu={max_overvoltage:.6f} solve_time_ms={solve_ms:.2f}")

                return {
                    "ok": True,
                    "objective_total": objective_total,
                    "objective_active": objective_active,
                    "objective_reactive": objective_reactive,
                    "max_overvoltage_pre_pu": max_overvoltage_pre,
                    "max_vm_pre_pu": max_vm_pre,
                    "max_overvoltage_pu": max_overvoltage,
                    "solve_time_ms": solve_ms,
                    "message": "ok" if not inactive_voltage else "ok (inactive voltage set: fixed at reference)",
                }
            except Exception as exc:
                solve_ms = (time.perf_counter() - t0) * 1000.0
                return {
                    "ok": False,
                    "objective_total": None,
                    "objective_active": None,
                    "objective_reactive": None,
                    "max_overvoltage_pre_pu": None,
                    "max_vm_pre_pu": None,
                    "max_overvoltage_pu": None,
                    "solve_time_ms": solve_ms,
                    "message": f"OPF failed: {exc}",
                }
    def get_noise_config(self) -> Dict[str, float | bool]:
        with self.lock:
            return {"enabled": bool(self.noise_enabled), "v_noise_std": float(self.v_noise_std), "s_noise_std": float(self.s_noise_std)}

    def set_noise_config(self, enabled: Optional[bool] = None, v_noise_std: Optional[float] = None, s_noise_std: Optional[float] = None):
        with self.lock:
            if enabled is not None:
                self.noise_enabled = bool(enabled)
            if v_noise_std is not None:
                self.v_noise_std = float(v_noise_std)
            if s_noise_std is not None:
                self.s_noise_std = float(s_noise_std)

    def snapshot(self) -> Dict:
        with self.lock:
            power_orig = { ext: complex(self.current_S_targets.get(ext, 0+0j)) for ext in DESIRED_SCENARIO_BUS_ORDER }
            voltages = { ext: complex(v) for ext, v in self.last_voltages.items() }
            powers = { ext: complex(s) for ext, s in self.last_powers.items() }
            if self.solver_mode == "opf":
                merged_mods: Dict[int, Tuple[float, float]] = {}
                for ext in self.manual_controlled_buses:
                    if ext in self.controlled_buses:
                        merged_mods[ext] = self.opf_mods.get(ext, (0.0, 0.0))
                    else:
                        merged_mods[ext] = self.mods.get(ext, (0.0, 0.0))
                mods = {ext: (float(v[0]), float(v[1])) for ext, v in merged_mods.items()}
            else:
                mods = {ext: (float(v[0]), float(v[1])) for ext, v in self.mods.items()}
            # Apply noise to voltages and powers in the snapshot only
            if self.noise_enabled:
                v_std = float(self.v_noise_std)
                s_std = float(self.s_noise_std)
                if v_std != 0.0:
                    for ext in list(voltages.keys()):
                        v = voltages[ext]
                        dv_real = self.rng.normal(0.0, v_std)
                        dv_imag = self.rng.normal(0.0, v_std)
                        voltages[ext] = complex(v.real + dv_real, v.imag + dv_imag)
                if s_std != 0.0:
                    for ext in list(powers.keys()):
                        s = powers[ext]
                        dp = self.rng.normal(0.0, s_std)
                        dq = self.rng.normal(0.0, s_std)
                        powers[ext] = complex(s.real + dp, s.imag + dq)
            return {
                "row_idx": self.current_row_idx,
                "voltages": {f"V{ext}_complex": complex(v) for ext, v in voltages.items()},
                "power_orig": {f"S{ext}_complex": complex(s) for ext, s in power_orig.items()},
                "power": {f"S{ext}_complex": complex(s) for ext, s in powers.items()},
                "mods": {ext: {"dP_mw": v[0], "dQ_mvar": v[1]} for ext, v in mods.items()},
                "breaker": {"name": "K4", "present": self.k4_sw_idx is not None, "closed": bool(self.k4_closed)},
                "solver": self.get_solver_state(),
                "cost": self.get_cost(),
            }


# -------------------- FastAPI app --------------------
def create_app(rtp: RealTimePowerflow) -> FastAPI:
    app = FastAPI(title="Real-time Powerflow Service (with noise)", version="0.1.0")

    @app.get("/status")
    def status():
        return {
            "row_idx": rtp.current_row_idx,
            "pf_rate_hz": rtp.pf_rate_hz,
            "scenario_rate_hz": rtp.scenario_rate_hz,
            "buses": sorted(list(rtp.nodecount_to_busidx.keys())),
            "controlled_buses": list(rtp.manual_controlled_buses),
            "opf_controlled_buses": list(rtp.controlled_buses),
            "opf_supervised_buses": list(rtp.opf_supervised_buses),
            "breaker": {"name": "K4", "present": rtp.k4_sw_idx is not None, "closed": bool(rtp.k4_closed)},
            "solver": rtp.get_solver_state(),
            "cost": rtp.get_cost(),
        }

    @app.get("/buses")
    def buses():
        return {
            "all": sorted(list(rtp.nodecount_to_busidx.keys())),
            "controlled": list(rtp.manual_controlled_buses),
            "opf_controlled": list(rtp.controlled_buses),
            "opf_supervised": list(rtp.opf_supervised_buses),
        }

    @app.get("/state")
    def state():
        snap = rtp.snapshot()
        def c2s(z: complex) -> str:
            try:
                return f"{z.real:+.9f}{z.imag:+.9f}j"
            except Exception:
                return "0+0j"
        volts_mag = {}
        for k, v in snap["voltages"].items():
            try:
                m = abs(v)
                if not np.isfinite(m):
                    m = None
                volts_mag[k.replace("_complex", "_mag")] = m
            except Exception:
                volts_mag[k.replace("_complex", "_mag")] = None
        return {
            "row_idx": snap["row_idx"],
            "voltages": {k: c2s(v) for k, v in snap["voltages"].items()},
            "power_orig": {k: c2s(v) for k, v in snap.get("power_orig", {}).items()},
            "power": {k: c2s(v) for k, v in snap.get("power", {}).items()},
            "mods": snap["mods"],
            "voltages_mag": volts_mag,
            "breaker": snap.get("breaker", {"name": "K4", "present": False, "closed": True}),
            "noise": rtp.get_noise_config(),
            "scenario": {"frozen": bool(rtp.scenario_frozen)},
            "solver": snap.get("solver", rtp.get_solver_state()),
            "cost": snap.get("cost", rtp.get_cost()),
        }

    @app.post("/mods")
    def post_mods(batch: ModBatch):
        for it in batch.items:
            if it.bus_ext not in rtp.nodecount_to_busidx:
                raise HTTPException(400, f"Unknown bus {it.bus_ext}")
            if it.bus_ext not in rtp.manual_controlled_buses:
                raise HTTPException(400, f"Bus {it.bus_ext} is not controllable from GUI/manual mods. Allowed: {rtp.manual_controlled_buses}")
            rtp.set_mod(it.bus_ext, it.dP_mw, it.dQ_mvar)
        return {"ok": True, "count": len(batch.items)}

    @app.delete("/mods")
    def delete_mods():
        rtp.clear_mods()
        return {"ok": True}

    @app.post("/control")
    def control(batch: ModBatch):
        return post_mods(batch)

    # ---- Solver mode ----
    @app.get("/solver")
    def get_solver():
        return rtp.get_solver_state()

    @app.post("/solver")
    def set_solver(cfg: SolverConfig):
        try:
            mode = rtp.set_solver_mode(cfg.mode)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return {"ok": True, "mode": mode, **rtp.get_solver_state()}

    @app.post("/solver/toggle")
    def toggle_solver():
        mode = rtp.toggle_solver_mode()
        return {"ok": True, "mode": mode, **rtp.get_solver_state()}

    # ---- Shared cost channel ----
    @app.get("/cost")
    def get_cost():
        return rtp.get_cost()

    @app.post("/cost")
    def set_cost(cfg: CostUpdate):
        src = str(cfg.source).strip().lower()
        if src not in ("opf", "julia"):
            raise HTTPException(400, "source must be 'opf' or 'julia'")
        try:
            rtp.set_cost(source=src, value=cfg.value, detail=cfg.detail)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return {"ok": True, **rtp.get_cost()}

    # ---- Noise ----
    @app.get("/noise")
    def get_noise():
        return rtp.get_noise_config()

    @app.post("/noise")
    def set_noise(cfg: NoiseConfig):
        rtp.set_noise_config(cfg.enabled, cfg.v_noise_std, cfg.s_noise_std)
        return {"ok": True, **rtp.get_noise_config()}

    @app.post("/noise/toggle")
    def toggle_noise():
        cur = rtp.get_noise_config()
        rtp.set_noise_config(enabled=not cur["enabled"])
        return {"ok": True, **rtp.get_noise_config()}

    # ---- Scenario freeze ----
    @app.get("/scenario")
    def get_scenario():
        return {"frozen": bool(rtp.scenario_frozen), "row_idx": rtp.current_row_idx, "n_rows": len(rtp.scenarios_df)}

    @app.post("/scenario/freeze")
    def set_scenario(frozen: bool):
        rtp.set_scenario_frozen(bool(frozen))
        return {"ok": True, "frozen": bool(rtp.scenario_frozen)}

    @app.post("/scenario/toggle")
    def toggle_scenario():
        fr = rtp.toggle_scenario_frozen()
        return {"ok": True, "frozen": bool(fr)}

    # ---- Breaker ----
    @app.get("/breaker")
    def get_breaker():
        s = rtp.snapshot().get("breaker", {"name": "K4", "present": False, "closed": True})
        return s

    @app.post("/breaker")
    def set_breaker(closed: bool):
        rtp.set_breaker_closed(closed)
        return {"ok": True, "closed": bool(closed)}

    @app.post("/breaker/toggle")
    def toggle_breaker():
        closed = rtp.toggle_breaker()
        return {"ok": True, "closed": bool(closed)}

    # ---- UI ----
    @app.get("/")
    def root():
        return HTMLResponse("""
<!doctype html>
<html>
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Real-time Powerflow UI (with noise)</title>
    <script src=\"https://cdn.plot.ly/plotly-2.32.0.min.js\"></script>
    <style>
        body { font-family: system-ui, sans-serif; margin: 0; padding: 0; }
        header { padding: 10px 16px; background: #1f2937; color: #fff; }
        /* Keep left controls column fixed and align both columns to the top */
        .wrap { display: grid; grid-template-columns: 360px 1fr; gap: 12px; padding: 12px; align-items: start; }
        /* Allow scrolling for long control column and ensure cards don't push each other down */
        .card { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; overflow: auto; }
        /* Make the right column stack its contents and keep plots sized consistently */
        .wrap > .card:last-child { display: flex; flex-direction: column; gap: 8px; }
        label { display:block; margin-top: 8px; font-size: 14px; color: #111827; }
        input[type=range] { width: 100%; }
        .row { display:flex; align-items:center; gap:8px; }
        .small { font-size:12px; color:#4b5563; }
        button { padding: 6px 10px; border-radius: 6px; border: 1px solid #d1d5db; background: #fff; cursor:pointer; }
        button.primary { background: #2563eb; color:#fff; border-color:#2563eb; }
        footer { padding:10px 16px; color:#6b7280; font-size:12px; }
        .fieldset { border-top: 1px dashed #d1d5db; margin-top: 10px; padding-top: 10px; }
        .grid2 { display:grid; grid-template-columns: 1fr 1fr; gap:8px; }
    </style>
    </head>
<body>
    <header>
        <h2>Real-time Powerflow UI (with noise)</h2>
        <div class="small" style="margin-top:4px;">
            OPF nodes: <strong id="opfNodesTop">-</strong>
            &nbsp; | &nbsp;
            Non-OPF nodes: <strong id="nonOpfNodesTop">-</strong>
        </div>
    </header>
    <div class=\"wrap\">
        <div class=\"card\">
            <div class=\"row\">
                <label for=\"bus\">Controlled bus</label>
                <select id=\"bus\"></select>
            </div>
            <div class=\"row\" style=\"margin-top:6px; gap:10px; align-items:center;\">
                <button id=\"toggleK4\" class=\"primary\">Toggle K4 breaker</button>
                <span class=\"small\">K4 state: <strong id=\"k4state\">unknown</strong></span>
            </div>
            <div class=\"fieldset\">
                <h4 style=\"margin:6px 0 4px;\">Solver</h4>
                <div class=\"row\" style=\"gap:8px; flex-wrap:wrap;\">
                    <button id=\"opfToggleBtn\">Enable OPF</button>
                    <span class=\"small\">Mode: <strong id=\"solverMode\">PF</strong></span>
                </div>
                <div class=\"small\" style=\"margin-top:6px;\">
                    Obj: <strong id=\"opfObj\">-</strong> |
                    Pcost: <strong id=\"opfPcost\">-</strong> |
                    Qcost: <strong id=\"opfQcost\">-</strong> |
                    Pre OV [p.u.]: <strong id=\"opfOvPre\">-</strong> |
                    Max OV [p.u.]: <strong id=\"opfOv\">-</strong> |
                    Solve [ms]: <strong id=\"opfMs\">-</strong>
                </div>
                <div class=\"small\" style=\"margin-top:4px;\">
                    Cost API: <strong id=\"costValue\">-</strong> |
                    Source: <strong id=\"costSource\">-</strong>
                </div>
            </div>
            <label for=\"p\">dP [MW] <span id=\"plbl\" class=\"small\">0.000</span></label>
            <input id=\"p\" type=\"range\" min=\"-0.2\" max=\"0.2\" step=\"0.001\" value=\"0\" />
            <label for=\"q\">dQ [MVAr] <span id=\"qlbl\" class=\"small\">0.000</span></label>
            <input id=\"q\" type=\"range\" min=\"-0.2\" max=\"0.2\" step=\"0.001\" value=\"0\" />
            <div style=\"margin-top:10px; display:flex; gap:8px; flex-wrap: wrap;\">
                <button id=\"apply\" class=\"primary\">Apply Mod</button>
                <button id=\"clearOne\">Set 0 for Selected</button>
                <button id=\"clearAll\">Clear All Mods</button>
            </div>

            <div class=\"fieldset grid2\">
                <div>
                    <div class=\"small\">Row: <span id=\"rowidx\">-</span></div>
                    <div class=\"small\">PF Rate: <span id=\"pf\">-</span> Hz</div>
                    <div class=\"small\">Scenario Rate: <span id=\"sc\">-</span> Hz</div>
                </div>
                <div>
                    <div style=\"display:flex; align-items:center; gap:8px; justify-content:flex-end;\">
                        <button id=\"pauseBtn\">Pause Updates</button>
                        <label class=\"row\" style=\"gap:6px; margin:0;\">
                            <span class=\"small\">Samples</span>
                            <select id=\"samplesKeep\">\n                                <option value=\"50\">50</option>\n                                <option value=\"100\">100</option>\n                                <option value=\"250\">250</option>\n                                <option value=\"500\" selected>500</option>\n                                <option value=\"1000\">1000</option>\n                            </select>
                        </label>
                    </div>
                </div>
            </div>

            <div class=\"fieldset\">
                <h4 style=\"margin:6px 0 4px;\">Noise</h4>
                <label class=\"row\" style=\"gap:8px;\">
                    <input type=\"checkbox\" id=\"noise_enabled_chk\" />
                    <span class=\"small\">Enabled</span>
                </label>
                <div class=\"row\" style=\"gap:8px; flex-wrap:wrap;\">
                    <label class=\"small\">v_noise_std <input id=\"v_noise_std\" type=\"number\" step=\"0.0001\" style=\"width:8ch;\" /></label>
                    <label class=\"small\">s_noise_std <input id=\"s_noise_std\" type=\"number\" step=\"0.000001\" style=\"width:10ch;\" /></label>
                    <button id=\"applyNoise\">Apply Noise</button>
                </div>
            </div>
        </div>
        <div class=\"card\">
            <div id=\"chart\" style=\"width:100%;height:52vh;\"></div>
            <div id=\"modsChart\" style=\"width:100%;height:22vh;margin-top:8px;\"></div>
            <div id=\"breakerChart\" style=\"width:100%;height:8vh;margin-top:8px;\"></div>
        </div>
    </div>
    <footer>Magnitudes relative to bus 33 angle.</footer>
    <script>
    const busSel = document.getElementById('bus');
        const pEl = document.getElementById('p');
        const qEl = document.getElementById('q');
        const pLbl = document.getElementById('plbl');
        const qLbl = document.getElementById('qlbl');
        const rowLbl = document.getElementById('rowidx');
        const pfLbl = document.getElementById('pf');
        const scLbl = document.getElementById('sc');
        const chartDiv = document.getElementById('chart');
        const modsChartDiv = document.getElementById('modsChart');
        const breakerChartDiv = document.getElementById('breakerChart');
        const samplesSel = document.getElementById('samplesKeep');
        const k4Btn = document.getElementById('toggleK4');
        const k4Lbl = document.getElementById('k4state');
        const opfToggleBtn = document.getElementById('opfToggleBtn');
        const opfNodesTop = document.getElementById('opfNodesTop');
        const nonOpfNodesTop = document.getElementById('nonOpfNodesTop');
        const solverModeLbl = document.getElementById('solverMode');
        const opfObjLbl = document.getElementById('opfObj');
        const opfPcostLbl = document.getElementById('opfPcost');
        const opfQcostLbl = document.getElementById('opfQcost');
        const opfOvPreLbl = document.getElementById('opfOvPre');
        const opfOvLbl = document.getElementById('opfOv');
        const opfMsLbl = document.getElementById('opfMs');
        const costValueLbl = document.getElementById('costValue');
        const costSourceLbl = document.getElementById('costSource');
        const pauseBtn = document.getElementById('pauseBtn');
        const noiseEnabled = document.getElementById('noise_enabled_chk');
        const vNoiseStd = document.getElementById('v_noise_std');
        const sNoiseStd = document.getElementById('s_noise_std');
        const applyNoiseBtn = document.getElementById('applyNoise');

    let controlled = [];
    let busToIdx = new Map();
    let busToIdxModsP = new Map();
    let busToIdxModsQ = new Map();
    let maxPts = 500;
    let sampleIdx = 0;

        function fmt(x){ return (Math.round(x*1000)/1000).toFixed(3); }

        function refreshCostLabels(cost){
            const c = cost || {};
            const v = c.value;
            const src = c.source;
            costValueLbl.textContent = (v === null || v === undefined) ? '-' : Number(v).toFixed(6);
            costSourceLbl.textContent = (src === null || src === undefined || String(src).trim() === '') ? '-' : String(src).toUpperCase();
        }

        function initPlot(busList){
            const data = [];
            busToIdx = new Map();
            let i = 0;
            for(const ext of busList){
                busToIdx.set(ext, i++);
                data.push({ x: [], y: [], name: `V${ext}_mag`, mode: 'lines+markers' });
            }
            const layout = { title: 'Voltage Magnitudes [p.u.]', yaxis: {autorange: true}, legend: {orientation:'h'} };
            Plotly.newPlot(chartDiv, data, layout, {displaylogo:false, responsive:true});

            // Mods plot: two traces per bus (dP and dQ)
            const modsData = [];
            busToIdxModsP = new Map();
            busToIdxModsQ = new Map();
            let k = 0;
            for(const ext of busList){
                busToIdxModsP.set(ext, k);
                modsData.push({ x: [], y: [], name: `dP${ext} [MW]`, mode: 'lines', line: {shape:'hv'} });
                k += 1;
                busToIdxModsQ.set(ext, k);
                modsData.push({ x: [], y: [], name: `dQ${ext} [MVAr]`, mode: 'lines', line: {dash:'dot', shape:'hv'} });
                k += 1;
            }
            const modsLayout = { title: 'Active Mods dP/dQ', yaxis: {range:[-0.06, 0.06], zeroline:true}, legend: {orientation:'h'} };
            Plotly.newPlot(modsChartDiv, modsData, modsLayout, {displaylogo:false, responsive:true});

            // Breaker plot (single binary trace: 1=closed, 0=open)
            const breakerData = [ { x: [], y: [], name: 'K4 closed', mode: 'lines', line: { shape: 'hv' } } ];
            const breakerLayout = { title: 'Breaker state (0=open, 1=closed)', yaxis: { range: [-0.2, 1.2], dtick: 1 }, margin: { t: 30 }, showlegend: false };
            Plotly.newPlot(breakerChartDiv, breakerData, breakerLayout, {displaylogo:false, responsive:true});
        }

    // update numeric labels when sliders move (both input and change for programmatic updates)
    pEl.addEventListener('input', ()=>{ pLbl.textContent = fmt(parseFloat(pEl.value || 0)); });
    qEl.addEventListener('input', ()=>{ qLbl.textContent = fmt(parseFloat(qEl.value || 0)); });
    pEl.addEventListener('change', ()=>{ pLbl.textContent = fmt(parseFloat(pEl.value || 0)); });
    qEl.addEventListener('change', ()=>{ qLbl.textContent = fmt(parseFloat(qEl.value || 0)); });

        // when the selected controlled bus changes, load current mod values from /state and update sliders
        busSel.addEventListener('change', async ()=>{
            try{
                const r = await fetch('/state');
                const js = await r.json();
                const mods = js.mods || {};
                const bus_ext = parseInt(busSel.value, 10);
                const m = mods[bus_ext] || { dP_mw: 0.0, dQ_mvar: 0.0 };
                pEl.value = m.dP_mw;
                qEl.value = m.dQ_mvar;
                pLbl.textContent = fmt(parseFloat(m.dP_mw));
                qLbl.textContent = fmt(parseFloat(m.dQ_mvar));
            }catch(e){ /* ignore */ }
        });

        function trimTraces(){
            const d = chartDiv.data || [];
            for(let i=0;i<d.length;i++){
                const x = d[i].x || [];
                const y = d[i].y || [];
                const n = Math.max(0, x.length - maxPts);
                if(n > 0){
                    const xs = x.slice(-maxPts);
                    const ys = y.slice(-maxPts);
                    Plotly.restyle(chartDiv, {x: [xs], y: [ys]}, [i]);
                }
            }
        }

        async function refreshStatus(){
            const r = await fetch('/status');
            const js = await r.json();
            document.getElementById('pf').textContent = js.pf_rate_hz;
            document.getElementById('sc').textContent = js.scenario_rate_hz;
            const br = js.breaker || {present:false, closed:true};
            document.getElementById('k4state').textContent = br.present ? (br.closed ? 'Closed' : 'Open') : 'Not present';
            const solver = js.solver || { mode: 'pf', opf: {} };
            const mode = String(solver.mode || 'pf').toLowerCase();
            solverModeLbl.textContent = mode.toUpperCase();
            const manualNodes = (js.controlled_buses || []).map(x => Number(x));
            const opfNodes = (js.opf_controlled_buses || []).map(x => Number(x));
            const nonOpf = manualNodes.filter(x => !opfNodes.includes(x));
            opfNodesTop.textContent = opfNodes.length ? opfNodes.join(', ') : '-';
            nonOpfNodesTop.textContent = nonOpf.length ? nonOpf.join(', ') : '-';
            opfToggleBtn.textContent = (mode === 'opf') ? 'Disable OPF' : 'Enable OPF';
            opfToggleBtn.classList.toggle('primary', mode === 'opf');
            const opf = solver.opf || {};
            opfObjLbl.textContent = (opf.objective_total === null || opf.objective_total === undefined) ? '-' : Number(opf.objective_total).toFixed(6);
            opfPcostLbl.textContent = (opf.objective_active === null || opf.objective_active === undefined) ? '-' : Number(opf.objective_active).toFixed(6);
            opfQcostLbl.textContent = (opf.objective_reactive === null || opf.objective_reactive === undefined) ? '-' : Number(opf.objective_reactive).toFixed(6);
            opfOvPreLbl.textContent = (opf.max_overvoltage_pre_pu === null || opf.max_overvoltage_pre_pu === undefined) ? '-' : Number(opf.max_overvoltage_pre_pu).toFixed(6);
            opfOvLbl.textContent = (opf.max_overvoltage_pu === null || opf.max_overvoltage_pu === undefined) ? '-' : Number(opf.max_overvoltage_pu).toFixed(6);
            opfMsLbl.textContent = (opf.solve_time_ms === null || opf.solve_time_ms === undefined) ? '-' : Number(opf.solve_time_ms).toFixed(2);
            refreshCostLabels(js.cost || {});
            if(controlled.length === 0){
                controlled = js.controlled_buses || [];
                busSel.innerHTML = controlled.map(b => `<option value="${b}">${b}</option>`).join('');
                initPlot(controlled);
                // trigger a change so the initial bus loads its mod values into sliders/labels
                try{ busSel.dispatchEvent(new Event('change')); }catch(e){}
            }
        }

        async function refreshSolver(){
            try{
                const r = await fetch('/solver');
                const js = await r.json();
                const mode = String(js.mode || 'pf').toLowerCase();
                solverModeLbl.textContent = mode.toUpperCase();
                opfToggleBtn.textContent = (mode === 'opf') ? 'Disable OPF' : 'Enable OPF';
                opfToggleBtn.classList.toggle('primary', mode === 'opf');
                const opf = js.opf || {};
                opfObjLbl.textContent = (opf.objective_total === null || opf.objective_total === undefined) ? '-' : Number(opf.objective_total).toFixed(6);
                opfPcostLbl.textContent = (opf.objective_active === null || opf.objective_active === undefined) ? '-' : Number(opf.objective_active).toFixed(6);
                opfQcostLbl.textContent = (opf.objective_reactive === null || opf.objective_reactive === undefined) ? '-' : Number(opf.objective_reactive).toFixed(6);
                opfOvPreLbl.textContent = (opf.max_overvoltage_pre_pu === null || opf.max_overvoltage_pre_pu === undefined) ? '-' : Number(opf.max_overvoltage_pre_pu).toFixed(6);
                opfOvLbl.textContent = (opf.max_overvoltage_pu === null || opf.max_overvoltage_pu === undefined) ? '-' : Number(opf.max_overvoltage_pu).toFixed(6);
                opfMsLbl.textContent = (opf.solve_time_ms === null || opf.solve_time_ms === undefined) ? '-' : Number(opf.solve_time_ms).toFixed(2);
            } catch(e) { /* ignore */ }
        }

        async function refreshNoise(){
            try{
                const r = await fetch('/noise');
                const js = await r.json();
                noiseEnabled.checked = !!js.enabled;
                vNoiseStd.value = js.v_noise_std ?? 0;
                sNoiseStd.value = js.s_noise_std ?? 0;
            } catch(e) { /* ignore */ }
        }

        async function refreshState(){
            const r = await fetch('/state');
            const js = await r.json();
            document.getElementById('rowidx').textContent = js.row_idx;
            refreshCostLabels(js.cost || {});
            const mags = js.voltages_mag || {};
            const yUpdate = [];
            const xUpdate = [];
            const indices = [];
            for(const ext of controlled){
                const idx = busToIdx.get(ext);
                if(idx === undefined) continue;
                const val = mags[`V${ext}_mag`];
                yUpdate.push([ (val === undefined || val === null) ? null : val ]);
                xUpdate.push([ sampleIdx ]);
                indices.push(idx);
            }
            if(indices.length){
                Plotly.extendTraces(chartDiv, { x: xUpdate, y: yUpdate }, indices, maxPts);
            }
            sampleIdx += 1;

            // Mods
            const mods = js.mods || {};
            const modsY = [];
            const modsX = [];
            const modsIdx = [];
            for(const ext of controlled){
                const pIdx = busToIdxModsP.get(ext);
                const qIdx = busToIdxModsQ.get(ext);
                if(pIdx === undefined || qIdx === undefined) continue;
                const m = mods[ext] || { dP_mw: 0.0, dQ_mvar: 0.0 };
                modsY.push([ m.dP_mw ]);
                modsX.push([ sampleIdx ]);
                modsIdx.push(pIdx);
                modsY.push([ m.dQ_mvar ]);
                modsX.push([ sampleIdx ]);
                modsIdx.push(qIdx);
            }
            if(modsIdx.length){
                Plotly.extendTraces(modsChartDiv, { x: modsX, y: modsY }, modsIdx, maxPts);
            }

            // Breaker
            try {
                const br = js.breaker || { present: false, closed: true };
                const brVal = (br.present === false) ? null : (br.closed ? 1 : 0);
                Plotly.extendTraces(breakerChartDiv, { x: [[ sampleIdx ]], y: [[ brVal ]] }, [0], maxPts);
            } catch(err){}
        }

        document.getElementById('apply').addEventListener('click', async ()=>{
            const bus_ext = parseInt(busSel.value);
            const body = { items: [{ bus_ext, dP_mw: parseFloat(pEl.value), dQ_mvar: parseFloat(qEl.value) }] };
            await fetch('/mods', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body)});
        });

        document.getElementById('clearOne').addEventListener('click', async ()=>{
            const bus_ext = parseInt(busSel.value);
            const body = { items: [{ bus_ext, dP_mw: 0.0, dQ_mvar: 0.0 }] };
            await fetch('/mods', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body)});
            pEl.value = 0; qEl.value = 0; pLbl.textContent = '0.000'; qLbl.textContent = '0.000';
        });

        document.getElementById('clearAll').addEventListener('click', async ()=>{
            await fetch('/mods', { method: 'DELETE' });
            pEl.value = 0; qEl.value = 0; pLbl.textContent = '0.000'; qLbl.textContent = '0.000';
        });

        samplesSel.addEventListener('change', ()=>{
            const v = parseInt(samplesSel.value, 10);
            if(Number.isFinite(v) && v > 0){ maxPts = v; trimTraces(); }
        });

        k4Btn.addEventListener('click', async ()=>{ await fetch('/breaker/toggle', { method: 'POST' }); setTimeout(refreshStatus, 150); });
        opfToggleBtn.addEventListener('click', async ()=>{ await fetch('/solver/toggle', { method: 'POST' }); setTimeout(refreshSolver, 150); setTimeout(refreshStatus, 200); });

        // Pause/Resume plotting updates (frontend-only)
        // Replace local pause with server-side scenario freeze toggle
        pauseBtn.addEventListener('click', async ()=>{
            try{
                await fetch('/scenario/toggle', { method: 'POST' });
                await refreshScenario();
                setTimeout(refreshStatus, 150);
            }catch(e){ console.error('toggle scenario failed', e); }
        });

        async function refreshScenario(){
            try{
                const r = await fetch('/scenario');
                const js = await r.json();
                const frozen = !!js.frozen;
                pauseBtn.textContent = frozen ? 'Unfreeze Scenario' : 'Freeze Scenario';
                if(frozen){ pauseBtn.classList.add('primary'); } else { pauseBtn.classList.remove('primary'); }
                // show row idx near controls
                document.getElementById('rowidx').textContent = js.row_idx;
            }catch(e){ /* ignore */ }
        }

        // Noise controls
        document.getElementById('applyNoise').addEventListener('click', async ()=>{
            const body = { enabled: !!noiseEnabled.checked, v_noise_std: parseFloat(vNoiseStd.value||'0'), s_noise_std: parseFloat(sNoiseStd.value||'0') };
            await fetch('/noise', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body)});
            setTimeout(refreshNoise, 200);
        });

    // boot
    refreshStatus();
    refreshNoise();
    refreshSolver();
    refreshScenario();
    setInterval(refreshStatus, 2000);
    // always poll server state so UI (plots/mods) update; scenario freeze is server-side now
    setInterval(()=>{ refreshState(); }, 50);
    setInterval(refreshNoise, 5000);
    setInterval(refreshSolver, 2000);
    setInterval(refreshScenario, 2000);
    </script>
</body>
</html>
        """)

    @app.get("/ui")
    def ui():
        return root()

    return app


# -------------------- main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", default=os.path.join("data", "grid", "coses_matpower_source.xlsx"))
    parser.add_argument("--scenarios", default=os.path.join("data", "scenarios", "scenarios_correlated_pv_rt.csv"))
    parser.add_argument("--pf-rate", type=float, default=20.0)
    parser.add_argument("--scenario-rate", type=float, default=1.0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--v-noise", type=float, default=0.00, help="std-dev of additive Gaussian noise on voltages (p.u.)")
    parser.add_argument("--s-noise", type=float, default=0.00001, help="std-dev of additive Gaussian noise on powers (MW / MVAr)")
    parser.add_argument("--rng-seed", type=int, default=None, help="optional RNG seed for repeatable noise")

    args = parser.parse_args()

    if not os.path.isfile(args.excel):
        raise FileNotFoundError(f"Excel not found: {args.excel}")
    if not os.path.isfile(args.scenarios):
        raise FileNotFoundError(f"Scenarios CSV not found: {args.scenarios}")

    rtp = RealTimePowerflow(args.excel, args.scenarios, pf_rate_hz=args.pf_rate, scenario_rate_hz=args.scenario_rate, v_noise_std=args.v_noise, s_noise_std=args.s_noise, rng_seed=args.rng_seed)
    rtp.start()

    app = create_app(rtp)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
