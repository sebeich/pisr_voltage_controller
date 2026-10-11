"""CoSES low-voltage grid as a pandapower model (Section IV-A, digital twin).

Builds the network from data/grid/coses_matpower_source.xlsx: 0.4 kV buses, standard cable
types, up to 61 line segments, a 30 MVA / X/R=3 Thevenin equivalent at 20 kV and a
250 kVA 20/0.4 kV Dyn transformer with 4 % short-circuit voltage.

Used by the offline dataset generation (Fig. 3) and the offline control replay (Fig. 9).
Ported from the notebook 02_powerflowmodel/powerflow_from_coses_relati ve.ipynb (cell 0)
of the development repository; the network construction is unchanged.
"""
from __future__ import annotations

import ast
import copy
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.shortcircuit as sc

warnings.filterwarnings("ignore", category=FutureWarning, module="pandapower")

ROOT = Path(__file__).resolve().parents[2]
EXCEL_FILE = ROOT / "data" / "grid" / "coses_matpower_source.xlsx"

THEVENIN_MVA = 30.0
KW_TO_MW = 1000.0
# Prosumer buses in the order of the scenario columns P_node1/Q_node1 ... P_node5/Q_node5
DESIRED_SCENARIO_BUS_ORDER = [64, 57, 55, 53, 61]
# Angle reference bus for all published phasors
REF_BUS_EXT = 33

_STD_TYPES = {
    "cfg70":  {"c_nf_per_km": 0, "r_ohm_per_km": 0.268, "x_ohm_per_km": 0.0804,      "max_i_ka": 0.142, "type": "cs", "q_mm2": 50, "alpha": 4.03e-3},
    "cfg95":  {"c_nf_per_km": 0, "r_ohm_per_km": 0.193, "x_ohm_per_km": 0.082309728, "max_i_ka": 0.142, "type": "cs", "q_mm2": 50, "alpha": 4.03e-3},
    "cfg16":  {"c_nf_per_km": 0, "r_ohm_per_km": 0.15,  "x_ohm_per_km": 0.092484,    "max_i_ka": 0.142, "type": "cs", "q_mm2": 50, "alpha": 4.03e-3},
    "cfg150": {"c_nf_per_km": 0, "r_ohm_per_km": 0.127, "x_ohm_per_km": 0.080,       "max_i_ka": 0.142, "type": "cs", "q_mm2": 50, "alpha": 4.03e-3},
    "cfg50":  {"c_nf_per_km": 0, "r_ohm_per_km": 0.39,  "x_ohm_per_km": 0.084915,    "max_i_ka": 0.142, "type": "cs", "q_mm2": 50, "alpha": 4.03e-3},
    "cfgx1":  {"c_nf_per_km": 0, "r_ohm_per_km": 0.127, "x_ohm_per_km": 0.08,        "max_i_ka": 0.142, "type": "cs", "q_mm2": 50, "alpha": 4.03e-3},
}

# Settings of every power flow in the offline pipeline
RUNPP_KW = dict(calculate_voltage_angles=True, init="flat", algorithm="nr", enforce_q_lims=True,
                max_iteration=50, tolerance_mva=1e-8, numba=True)


class CosesGrid:
    """Network template plus the bus/load mappings needed to apply scenario powers."""

    def __init__(self, excel_file: Path | str = EXCEL_FILE, verbose: bool = False):
        self.verbose = verbose
        lines_df = pd.read_excel(excel_file, sheet_name="coses_matpower_source")
        buses_df = pd.read_excel(excel_file, sheet_name="buses")
        loads_df = pd.read_excel(excel_file, sheet_name="Loads")
        gens_df = pd.read_excel(excel_file, sheet_name="Generator")

        net = pp.create_empty_network()
        for name, params in _STD_TYPES.items():
            if name not in net.std_types["line"]:
                pp.create_std_type(net, params, name=name, element="line")

        self.nodecount_to_busidx: Dict[int, int] = {}
        for idx in range(len(buses_df["Node_count"])):
            ext_id = int(buses_df["Node_count"][idx])
            self.nodecount_to_busidx[ext_id] = pp.create_bus(
                net, name=buses_df["Node_names"][idx], vn_kv=0.40, type=buses_df["type"][idx])

        for idx in range(len(loads_df["bus"])):
            bus_ext = int(loads_df["bus"][idx])
            if bus_ext not in self.nodecount_to_busidx:
                raise ValueError(f"Load row {idx} references unknown bus {bus_ext}")
            pp.create_load(net, bus=self.nodecount_to_busidx[bus_ext],
                           p_mw=float(loads_df["pd"][idx]) / KW_TO_MW,
                           q_mvar=float(loads_df["qd"][idx]) / KW_TO_MW, name=f"base_load{idx}")

        # Replace the original slack with a Thevenin equivalent at the first generator bus
        anchor = self.nodecount_to_busidx[int(gens_df["bus"].iloc[0])]
        hv_bus = self._insert_thevenin_source(net, anchor, THEVENIN_MVA, 3.0)
        self._calibrate_sc_impedance(net, hv_bus, THEVENIN_MVA, 0.01)

        for idx in range(min(61, len(lines_df))):
            f_ext, t_ext = int(lines_df["f_bus_id"][idx]), int(lines_df["t_bus_id"][idx])
            if f_ext not in self.nodecount_to_busidx or t_ext not in self.nodecount_to_busidx:
                self._log(f"Skipping line {idx}: unknown bus {f_ext}->{t_ext}")
                continue
            cfg = str(lines_df["config"][idx])
            if cfg not in net.std_types["line"]:
                warnings.warn(f"line config '{cfg}' not a defined std_type; defaulting to 'cfg70'")
                cfg = "cfg70"
            pp.create_line(net, name=f"line_{idx}", from_bus=self.nodecount_to_busidx[f_ext],
                           to_bus=self.nodecount_to_busidx[t_ext],
                           length_km=float(lines_df["length_km"][idx]), std_type=cfg)

        # Voltage recording order: reference bus first, then the prosumer buses
        self.resolved_keep: List[Tuple[int, int]] = []
        if REF_BUS_EXT in self.nodecount_to_busidx:
            self.resolved_keep.append((REF_BUS_EXT, self.nodecount_to_busidx[REF_BUS_EXT]))
        self.resolved_keep += [(ext, self.nodecount_to_busidx[ext]) for ext in DESIRED_SCENARIO_BUS_ORDER
                               if ext in self.nodecount_to_busidx and ext != REF_BUS_EXT]

        # Keep exactly one in-service load per prosumer bus, disable all other loads
        busidx_to_ext = {v: k for k, v in self.nodecount_to_busidx.items()}
        kept: Dict[int, int] = {}
        for load_idx in list(net.load.index):
            ext = busidx_to_ext[int(net.load.at[load_idx, "bus"])]
            keep = ext in DESIRED_SCENARIO_BUS_ORDER and ext not in kept
            if keep:
                kept[ext] = load_idx
            net.load.at[load_idx, "in_service"] = keep
        self.scenario_load_indices = [kept[ext] for ext in DESIRED_SCENARIO_BUS_ORDER if ext in kept]
        if len(self.scenario_load_indices) != len(DESIRED_SCENARIO_BUS_ORDER):
            warnings.warn("Some prosumer buses are missing loads in the Excel input.")
        self.net_template = net

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _insert_thevenin_source(self, net, original_bus: int, s_sc_target_mva: float, x_r_ratio: float) -> int:
        vn_lv_kv = float(net.bus.at[original_bus, "vn_kv"])
        vn_hv_kv = 20.0
        hv_bus = pp.create_bus(net, name="Thevenin HV Bus", vn_kv=vn_hv_kv, type="b")
        z_th = (vn_hv_kv ** 2) / s_sc_target_mva
        r_th = z_th / math.sqrt(1.0 + x_r_ratio ** 2)
        x_th = r_th * x_r_ratio
        ext_bus = pp.create_bus(net, name="External Grid Bus", vn_kv=vn_hv_kv, type="b")
        pp.create_ext_grid(net, bus=ext_bus, vm_pu=1.0, va_degree=0.0, s_sc_max_mva=1e9, s_sc_min_mva=1e9,
                           rx_min=0.0, rx_max=0.0, name="Thevenin External Grid 20kV")
        pp.create_line_from_parameters(net, name="Thevenin Line 20kV", from_bus=ext_bus, to_bus=hv_bus,
                                       length_km=1.0, r_ohm_per_km=r_th, x_ohm_per_km=x_th, c_nf_per_km=0.0,
                                       max_i_ka=10.0, type="ol")
        pp.create_transformer_from_parameters(net, hv_bus=hv_bus, lv_bus=original_bus, sn_mva=0.25,
                                              vn_hv_kv=vn_hv_kv, vn_lv_kv=vn_lv_kv, vk_percent=4.0,
                                              vkr_percent=0.5, pfe_kw=0.5, i0_percent=0.2, shift_degree=0,
                                              vector_group="Dyn", name="Thevenin Trafo 250kVA 4%")
        self._log(f"Thevenin source: R={r_th:.6e} Ohm, X={x_th:.6e} Ohm, |Z|={z_th:.6e} Ohm")
        return hv_bus

    def _calibrate_sc_impedance(self, net, bus: int, target_mva: float, tol: float):
        """Scale the Thevenin line impedance once so that S_sc at `bus` matches target_mva.

        Note: the original notebook calls this before the LV lines exist, so the short-circuit
        calculation runs on the source + transformer only. Kept as-is for identical results.
        """
        line_mask = net.line.name == "Thevenin Line 20kV"
        if not line_mask.any():
            return
        tmp = copy.deepcopy(net)
        sc.calc_sc(tmp, case="max", fault="3ph", topology="auto", lv_tol_percent=10)
        if bus not in tmp.res_bus_sc.index:
            return
        vn_kv = float(tmp.bus.loc[bus, "vn_kv"])
        s_sc_mva = math.sqrt(3) * vn_kv * float(tmp.res_bus_sc.at[bus, "ikss_ka"])
        if abs(s_sc_mva - target_mva) / target_mva <= tol:
            return
        scale = s_sc_mva / target_mva
        net.line.loc[line_mask, "r_ohm_per_km"] *= scale
        net.line.loc[line_mask, "x_ohm_per_km"] *= scale

    # ------------------------------------------------------------------ power flow helpers
    def _solve(self, s_by_ext: Dict[int, complex]) -> pp.pandapowerNet:
        net = copy.deepcopy(self.net_template)
        for ext, load_idx in zip(DESIRED_SCENARIO_BUS_ORDER, self.scenario_load_indices):
            s = s_by_ext.get(ext, 0j)
            net.load.at[load_idx, "p_mw"] = float(s.real)
            net.load.at[load_idx, "q_mvar"] = float(s.imag)
        pp.runpp(net, **RUNPP_KW)
        return net

    def _phasor(self, net, bus_idx, ref_rad: float) -> complex:
        vm = float(net.res_bus.at[bus_idx, "vm_pu"])
        va = math.radians(float(net.res_bus.at[bus_idx, "va_degree"])) - ref_rad
        return vm * complex(math.cos(va), math.sin(va))

    def solve_row(self, s_by_ext: Dict[int, complex]) -> list:
        """Voltages (resolved_keep order, angle relative to bus 33), the input S, and V66."""
        net = self._solve(s_by_ext)
        ref_rad = 0.0
        if REF_BUS_EXT in self.nodecount_to_busidx:
            ref_rad = math.radians(float(net.res_bus.at[self.nodecount_to_busidx[REF_BUS_EXT], "va_degree"]))
        row = [self._phasor(net, bus_idx, ref_rad) for _, bus_idx in self.resolved_keep]
        row += [s_by_ext.get(ext, 0j) for ext in DESIRED_SCENARIO_BUS_ORDER]
        row.append(self._phasor(net, np.int64(66), ref_rad))
        return row

    def columns(self, last_v: str = "V66_complex") -> List[str]:
        return ([f"V{ext}_complex" for ext, _ in self.resolved_keep]
                + [f"S{ext}_complex" for ext in DESIRED_SCENARIO_BUS_ORDER] + [last_v])

    def run_pf_and_collect_complex(self, df_in: pd.DataFrame, start_idx: int, n_samples: int) -> pd.DataFrame:
        """Run one AC power flow per scenario row (1-based start_idx); scenario kW -> MW with sign flip."""
        rows = []
        for ridx in range(start_idx - 1, start_idx - 1 + n_samples):
            s_by_ext = {ext: complex(float(df_in.at[ridx, f"P_node{j}"]) / -1000.0,
                                     float(df_in.at[ridx, f"Q_node{j}"]) / -1000.0)
                        for j, ext in enumerate(DESIRED_SCENARIO_BUS_ORDER, start=1)}
            rows.append(self.solve_row(s_by_ext))
        return pd.DataFrame(rows, columns=self.columns())


def parse_complex(v) -> complex:
    """Parse Python/Julia complex strings such as '(a+bj)', 'a+bj', 'ComplexF64(a + bim)'."""
    if isinstance(v, complex):
        return v
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 0j
    s = str(v).strip().replace("ComplexF64", "").replace("im", "j").replace(" ", "")
    try:
        return complex(ast.literal_eval(s)) if s.startswith("(") else complex(s)
    except Exception:
        try:
            return complex(s.strip("()"))
        except Exception:
            return 0j


def read_complex_csv(path: Path | str) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str)
    return raw.apply(lambda col: col.map(parse_complex))
