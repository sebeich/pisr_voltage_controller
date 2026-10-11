#!/usr/bin/env python3
"""
Real-time OPC UA service with the same FastAPI API and GUI as the powerflow service.

What it does
- Connects to an OPC UA server (NI Veristand-style) and locates the Test object with
  Veristand_to_Outside (read) and Outside_to_Veristand (write) child nodes.
- Reads complex voltages (Vm, angle) from SS{n}_V and SS{n}_a nodes and publishes
  them through /state just like the powerflow service (as V{ext}_complex and *_mag).
- Generates streaming correlated P/Q setpoints for inverters E1, E2, E4, E6 using a
  correlated AR(1) process, driven by the H3_PV_P and H3_PV_Q inputs (if available).
- Writes the setpoints (plus any /mods additive adjustments) to OPC UA nodes E*_P/E*_Q.

Notes
- We keep the same endpoints and HTML UI for continuity; breaker controls are stubbed
  as "not present".
- "pf-rate" becomes the OPC polling rate, and "scenario-rate" becomes the setpoint
  generation rate.
"""
# python experiments/V-E_phil_lab_fig12_fig13/opcua_fastapi_service.py   (lab PC: Windows, Python 3.13)
from __future__ import annotations

import argparse
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

try:
    # opcua client dependency
    from opcua import Client, ua
except Exception as _e:  # pragma: no cover
    Client = None  # type: ignore
    ua = None      # type: ignore


# -------------------- Simple OPC UA wrapper (adapted from opcua_basescript.py) --------------------
class VeristandOpcuaClient:
    def __init__(self, opcua_url: str, test_node_name: str = "Test"):
        self.opcua_url = opcua_url
        self.test_node_name = test_node_name
        self.client = None
        self.objects_node = None
        self.test_node = None
        self.outside_node = None  # Outside_to_Veristand
        self.veristand_node = None  # Veristand_to_Outside

    def connect(self) -> None:
        if Client is None:
            raise RuntimeError("python-opcua not installed")
        self.client = Client(self.opcua_url)
        self.client.connect()
        self.objects_node = self.client.get_objects_node()
        try:
            self.test_node = self.objects_node.get_child(f"0:{self.test_node_name}")
        except Exception:
            self.test_node = None
        if self.test_node is not None:
            try:
                self.outside_node = self.test_node.get_child("0:Outside_to_Veristand")
            except Exception:
                self.outside_node = None
            try:
                self.veristand_node = self.test_node.get_child("0:Veristand_to_Outside")
            except Exception:
                self.veristand_node = None

    def disconnect(self) -> None:
        if self.client is not None:
            try:
                self.client.disconnect()
            finally:
                self.client = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.disconnect()

    def _child_map(self, node) -> Dict[str, Any]:
        m = {}
        if node is None:
            return m
        try:
            for child in node.get_children():
                try:
                    m[child.get_display_name().Text] = child
                except Exception:
                    continue
        except Exception:
            pass
        return m

    def read_values(self, names: List[str]) -> Dict[str, Any]:
        """Read values from Veristand_to_Outside by display name."""
        if self.veristand_node is None:
            raise RuntimeError("Veristand_to_Outside node not available")
        m = self._child_map(self.veristand_node)
        out: Dict[str, Any] = {}
        for n in names:
            node = m.get(n)
            if node is None:
                out[n] = None
                continue
            try:
                out[n] = node.get_value()
            except Exception:
                out[n] = None
        return out

    def write_values(self, kv: Dict[str, Any]) -> Dict[str, str]:
        """Write to Outside_to_Veristand by display name."""
        if self.outside_node is None:
            raise RuntimeError("Outside_to_Veristand node not available")
        m = self._child_map(self.outside_node)
        res: Dict[str, str] = {}
        for name, val in kv.items():
            node = m.get(name)
            if node is None:
                res[name] = "node not found"
                continue
            try:
                node.set_value(val)
                res[name] = "ok"
            except Exception as e:
                # Attempt variant wrapping
                try:
                    if isinstance(val, float):
                        node.set_value(ua.Variant(val, ua.VariantType.Double))
                    elif isinstance(val, int):
                        node.set_value(ua.Variant(val, ua.VariantType.Int64))
                    elif isinstance(val, bool):
                        node.set_value(ua.Variant(val, ua.VariantType.Boolean))
                    else:
                        node.set_value(ua.Variant(str(val), ua.VariantType.String))
                    res[name] = "ok"
                except Exception as e2:
                    res[name] = f"error: {e2}"
        return res


# -------------------- Streaming correlated AR(1) generator --------------------
@dataclass
class Ar1:
    alpha: float
    sigma: float
    x: float = 0.0

    def step(self, mu: float, rng: np.random.Generator) -> float:
        self.x = self.alpha * self.x + (1.0 - self.alpha) * float(mu) + rng.standard_normal() * self.sigma
        return self.x


# -------------------- API models --------------------
class ModItem(BaseModel):
    bus_ext: int
    dP_mw: float = 0.0
    dQ_mvar: float = 0.0


class ModBatch(BaseModel):
    items: List[ModItem]


# -------------------- Real-time OPC UA runner --------------------
E_NODE_IDS: List[int] = [1,2,4,6]  # We have setpoints for E1, E2, E4, E6
SS_VOLTAGE_MAP: Dict[int, str] = {  # ext id -> SS name prefix used for voltage/angle
    3: "SS3",
    4: "SS4",
    6: "SS6",
    5: "SS5",
    1: "SS1",
}


class RealTimeOpcua:
    def __init__(
        self,
        opc_url: str,
        pf_rate_hz: float = 20.0,
        scenario_rate_hz: float = 1.0,
        alpha: float = 0.95,
        muP: float = 10.0,
        sigmaP: float = 0.8,
        muQ: float = 0.0,
        sigmaQ: float = 0.1,
        corr: float = 0.85,
        P_scale: float = 0.05,
        Q_scale: float = 0.01,
        seed: Optional[int] = None,
    ):
        self.opc_url = opc_url
        self.pf_rate_hz = float(pf_rate_hz)
        self.scenario_rate_hz = float(scenario_rate_hz)
        self.alpha = float(alpha)
        self.muP = float(muP)
        self.sigmaP = float(sigmaP)
        self.muQ = float(muQ)
        self.sigmaQ = float(sigmaQ)
        self.corr = float(max(-0.999, min(0.999, corr)))
        self.P_scale = float(P_scale)
        self.Q_scale = float(Q_scale)
        self.s = math.sqrt(max(0.0, 1.0 - self.corr * self.corr))
        self.rng = np.random.default_rng(seed)

        # OPC UA client
        self.client = VeristandOpcuaClient(self.opc_url)
        self.client.connect()

        # Shared state
        self.lock = threading.RLock()
        self._stop = threading.Event()
        self.current_row_idx = 0
        # Targets and mods for E nodes
        self.current_S_targets = {e: 0.0 + 0.0j for e in E_NODE_IDS}
        self.mods = {}
        # Last observed voltages keyed by ext id (from SS nodes available)
        self.last_voltages = {}
        # Last commanded powers (targets + mods)
        self.last_powers = {}
        # Last PV powers read from OPC UA (H3_PV_P, H3_PV_Q)
        self.last_pv_pq = (0.0, 0.0)

        # AR(1) streamers
        self.baseP = Ar1(alpha=self.alpha, sigma=self.sigmaP)
        self.baseQ = Ar1(alpha=self.alpha, sigma=self.sigmaQ)
        self.noiseP = {e: Ar1(alpha=self.alpha, sigma=self.sigmaP) for e in E_NODE_IDS}
        self.noiseQ = {e: Ar1(alpha=self.alpha, sigma=self.sigmaQ) for e in E_NODE_IDS}

        # Threads
        self._thr_scen = threading.Thread(target=self._scenario_loop, name="scenario_loop", daemon=True)
        self._thr_io = threading.Thread(target=self._io_loop, name="io_loop", daemon=True)

    # ---------- control ----------
    def start(self):
        self._thr_scen.start()
        self._thr_io.start()

    def stop(self):
        self._stop.set()
        for t in (self._thr_scen, self._thr_io):
            if t.is_alive():
                t.join(timeout=1.0)
        try:
            self.client.disconnect()
        except Exception:
            pass

    # ---------- loops ----------
    def _scenario_loop(self):
        period = 1.0 / max(1e-6, self.scenario_rate_hz)
        while not self._stop.is_set():
            t0 = time.perf_counter()
            # Read PV inputs (may be None/False); treat non-numeric as 0.0
            try:
                pv_vals = self.client.read_values(["H3_PV_P", "H3_PV_Q"])  # type: ignore
            except Exception:
                pv_vals = {"H3_PV_P": None, "H3_PV_Q": None}
            try:
                pvP = float(pv_vals.get("H3_PV_P") or 0.0)
                pvP = -8000.0
            except Exception:
                pvP = -800.0
            try:
                pvQ = float(pv_vals.get("H3_PV_Q") or 0.0)
                #pvQ = 500.0
            except Exception:
                pvQ = 0.0

            # Advance AR(1) generators with time-varying mean = mu + input
            bP = self.baseP.step(mu=self.muP + pvP, rng=self.rng)
            bQ = self.baseQ.step(mu=self.muQ + pvQ, rng=self.rng)

            new_targets: Dict[int, complex] = {}
            for e in E_NODE_IDS:
                nP = self.noiseP[e].step(mu=0.0, rng=self.rng)
                nQ = self.noiseQ[e].step(mu=0.0, rng=self.rng)
                P = (self.corr * bP + self.s * nP) * self.P_scale
                Q = (self.corr * bQ + self.s * nQ) * self.Q_scale
                new_targets[e] = complex(P, Q)

            with self.lock:
                self.current_S_targets = new_targets
                # Row index is just a counter here
                self.current_row_idx = (self.current_row_idx + 1) % 10_000_000
                # store latest PV inputs
                self.last_pv_pq = (pvP, pvQ)

            # sleep remainder
            dt = time.perf_counter() - t0
            time.sleep(max(0.0, period - dt))

    def _io_loop(self):
        period = 1.0 / max(1e-6, self.pf_rate_hz)
        while not self._stop.is_set():
            t0 = time.perf_counter()
            # Snapshot desired setpoints and mods
            with self.lock:
                S_targets = self.current_S_targets.copy()
                mods = self.mods.copy()

            # Write setpoints to OPC UA
            kv: Dict[str, float] = {}
            for e in E_NODE_IDS:
                s = S_targets.get(e, 0.0 + 0.0j)
                dP, dQ = mods.get(e, (0.0, 0.0))
                P = float(s.real + dP)
                Q = float(s.imag + dQ)
                kv[f"E{e}_P"] = -P
                kv[f"E{e}_Q"] = -Q
            try:
                self.client.write_values(kv)
            except Exception:
                pass

            # Read voltages from SS nodes and compute complex voltages
            voltages: Dict[int, complex] = {}
            names: List[str] = []
            for ext, base in SS_VOLTAGE_MAP.items():
                names.append(f"{base}_V")
                names.append(f"{base}_a")
            try:
                vals = self.client.read_values(names)
            except Exception:
                vals = {n: None for n in names}

            # Read all voltages and angles first
            raw_voltages = {}
            for ext, base in SS_VOLTAGE_MAP.items():
                v = vals.get(f"{base}_V")
                a = vals.get(f"{base}_a")
                try:
                    vm = float(v) / (230 * np.sqrt(2))
                    va_rad = float(a)  # Angles are in radians
                    if math.isfinite(vm) and math.isfinite(va_rad):
                        raw_voltages[ext] = (vm, va_rad)
                except Exception:
                    continue

            # V1 (ext=1) is the reference
            ref_angle_rad = raw_voltages.get(1, (0.0, 0.0))[1]

            for ext, (vm, va_rad) in raw_voltages.items():
                # Calculate angle relative to V1 and convert to degrees
                va_deg = math.degrees(va_rad - ref_angle_rad)
                # Re-convert to radians for complex number calculation
                final_va_rad = math.radians(va_deg)
                voltages[ext] = vm * (math.cos(final_va_rad) + 1j * math.sin(final_va_rad))

            # Record last powers (the ones we attempted to write)
            powers: Dict[int, complex] = {}
            for e in E_NODE_IDS:
                s = S_targets.get(e, 0.0 + 0.0j)
                dP, dQ = mods.get(e, (0.0, 0.0))
                powers[e] = complex(s.real + dP, s.imag + dQ)

            with self.lock:
                self.last_voltages = voltages
                self.last_powers = powers

            # sleep remainder
            dt = time.perf_counter() - t0
            #print(f"dt: {dt}, period: {period}")
            time.sleep(max(0.0, period - dt))
            

    # ---------- API helpers ----------
    def set_mod(self, bus_ext: int, dP_mw: float, dQ_mvar: float):
        with self.lock:
            if bus_ext not in E_NODE_IDS:
                raise KeyError(f"Unknown controlled bus {bus_ext}")
            self.mods[bus_ext] = (float(dP_mw), float(dQ_mvar))

    def clear_mods(self):
        with self.lock:
            self.mods.clear()

    def snapshot(self):
        with self.lock:
            power_orig = {e: complex(self.current_S_targets.get(e, 0+0j)) for e in E_NODE_IDS}
            return {
                "row_idx": self.current_row_idx,
                "voltages": {f"V{ext}_complex": complex(v) for ext, v in self.last_voltages.items()},
                "power_orig": {f"S{e}_complex": complex(s) for e, s in power_orig.items()},
                "power": {f"S{e}_complex": complex(s) for e, s in self.last_powers.items()},
                "mods": {e: {"dP_mw": v[0], "dQ_mvar": v[1]} for e, v in self.mods.items()},
                # Include PV P/Q and complex S for API consumers
                "pv_power": {
                    "P": float(self.last_pv_pq[0]),
                    "Q": float(self.last_pv_pq[1]),
                    "SPV_complex": complex(self.last_pv_pq[0], self.last_pv_pq[1]),
                },
                # Breaker not present in this OPC-only service
                "breaker": {"name": "K4", "present": False, "closed": True},
            }


# -------------------- FastAPI app (UI reused) --------------------
def create_app(rt: RealTimeOpcua) -> FastAPI:
    app = FastAPI(title="Real-time OPC UA Service", version="0.1.0")

    @app.get("/status")
    def status():
        return {
            "row_idx": rt.current_row_idx,
            "pf_rate_hz": rt.pf_rate_hz,
            "scenario_rate_hz": rt.scenario_rate_hz,
            "buses": sorted(list(SS_VOLTAGE_MAP.keys())),
            "controlled_buses": E_NODE_IDS,
            "breaker": {"name": "K4", "present": False, "closed": True},
        }

    @app.get("/buses")
    def buses():
        return {"all": sorted(list(SS_VOLTAGE_MAP.keys())), "controlled": E_NODE_IDS}

    @app.get("/state")
    def state():
        snap = rt.snapshot()

        def c2s(z: complex) -> str:
            try:
                return f"{z.real:+.9f}{z.imag:+.9f}j"
            except Exception:
                return "0+0j"

        volts_mag: Dict[str, Optional[float]] = {}
        for k, v in snap.get("voltages", {}).items():
            try:
                m = abs(v)
                if not math.isfinite(m):
                    m = None
                volts_mag[k.replace("_complex", "_mag")] = m
            except Exception:
                volts_mag[k.replace("_complex", "_mag")] = None

        return {
            "row_idx": snap["row_idx"],
            "voltages": {k: c2s(v) for k, v in snap.get("voltages", {}).items()},
            "power_orig": {k: c2s(v) for k, v in snap.get("power_orig", {}).items()},
            "power": {k: c2s(v) for k, v in snap.get("power", {}).items()},
            "mods": snap.get("mods", {}),
            "voltages_mag": volts_mag,
            # Expose PV powers both as numeric P/Q and complex string
            "pv_power": {
                "P": float(snap.get("pv_power", {}).get("P", 0.0)),
                "Q": float(snap.get("pv_power", {}).get("Q", 0.0)),
                "SPV_complex": c2s(snap.get("pv_power", {}).get("SPV_complex", 0+0j)),
            },
            "breaker": snap.get("breaker", {"name": "K4", "present": False, "closed": True}),
        }

    @app.post("/mods")
    def post_mods(batch: ModBatch):
        for it in batch.items:
            if it.bus_ext not in E_NODE_IDS:
                raise HTTPException(400, f"Unknown bus {it.bus_ext}")
            rt.set_mod(it.bus_ext, it.dP_mw, it.dQ_mvar)
        return {"ok": True, "count": len(batch.items)}

    @app.delete("/mods")
    def delete_mods():
        rt.clear_mods()
        return {"ok": True}

    @app.post("/control")
    def control(batch: ModBatch):
        return post_mods(batch)

    # UI (copied from powerflow service; breaker controls will show "Not present")
    @app.get("/")
    def root():
        return HTMLResponse("""
<!doctype html>
<html>
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Real-time OPC UA UI</title>
    <script src=\"https://cdn.plot.ly/plotly-2.32.0.min.js\"></script>
    <style>
        body { font-family: system-ui, sans-serif; margin: 0; padding: 0; }
        header { padding: 10px 16px; background: #1f2937; color: #fff; }
        .wrap { display: grid; grid-template-columns: 320px 1fr; gap: 12px; padding: 12px; }
        .card { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; }
        label { display:block; margin-top: 8px; font-size: 14px; color: #111827; }
        input[type=range] { width: 100%; }
        .row { display:flex; align-items:center; gap:8px; }
        .small { font-size:12px; color:#4b5563; }
        button { padding: 6px 10px; border-radius: 6px; border: 1px solid #d1d5db; background: #fff; cursor:pointer; }
        button.primary { background: #2563eb; color:#fff; border-color:#2563eb; }
        footer { padding:10px 16px; color:#6b7280; font-size:12px; }
    </style>
    </head>
<body>
    <header>
        <h2>Real-time OPC UA UI</h2>
    </header>
    <div class=\"wrap\">\n        <div class=\"card\">\n            <div class=\"row\">\n                <label for=\"bus\">Controlled bus</label>\n                <select id=\"bus\"></select>\n            </div>\n            <div class=\"row\" style=\"margin-top:6px; gap:10px; align-items:center;\">\n                <button id=\"toggleK4\" class=\"primary\">Toggle K4 breaker</button>\n                <span class=\"small\">K4 state: <strong id=\"k4state\">unknown</strong></span>\n            </div>\n            <label for=\"p\">dP [MW] <span id=\"plbl\" class=\"small\">0.000</span></label>\n            <input id=\"p\" type=\"range\" min=\"-5000.0\" max=\"5000.0\" step=\"1.0\" value=\"0\" />\n            <label for=\"q\">dQ [MVAr] <span id=\"qlbl\" class=\"small\">0.000</span></label>\n            <input id=\"q\" type=\"range\" min=\"-5000.0\" max=\"5000.0\" step=\"1.0\" value=\"0\" />\n            <div style=\"margin-top:10px; display:flex; gap:8px;\">\n                <button id=\"apply\" class=\"primary\">Apply Mod</button>\n                <button id=\"clearOne\">Set 0 for Selected</button>\n                <button id=\"clearAll\">Clear All Mods</button>\n            </div>\n            <div style=\"margin-top:10px; display:flex; flex-wrap:wrap; gap:12px; align-items:center;\" class=\"small\">\n                <div>Row: <span id=\"rowidx\">-</span></div>\n                <div>PF Rate: <span id=\"pf\">-</span> Hz</div>\n                <div>Scenario Rate: <span id=\"sc\">-</span> Hz</div>\n                <div style=\"display:flex; align-items:center;\">\n                    <button id=\"pauseBtn\" style=\"height:28px; margin-left:8px;\">Pause Updates</button>\n                </div>\n                <div class=\"row\" style=\"gap:6px;\">\n                    <label for=\"samplesKeep\" style=\"margin:0;\">Samples to keep</label>\n                    <select id=\"samplesKeep\">\n                        <option value=\"50\">50</option>\n                        <option value=\"100\">100</option>\n                        <option value=\"250\">250</option>\n                        <option value=\"500\" selected>500</option>\n                        <option value=\"1000\">1000</option>\n                    </select>\n                </div>\n            </div>\n        </div>\n        <div class=\"card\">\n            <div id=\"chart\" style=\"width:100%;height:52vh;\"></div>\n            <div id=\"modsChart\" style=\"width:100%;height:22vh;margin-top:8px;\"></div>\n            <div id=\"breakerChart\" style=\"width:100%;height:8vh;margin-top:8px;\"></div>\n        </div>\n    </div>
    <footer>Voltages are read from OPC UA SS nodes.</footer>
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
        const pauseBtn = document.getElementById('pauseBtn');

        let controlled = [];
        let busToIdx = new Map();
        let busToIdxModsP = new Map();
        let busToIdxModsQ = new Map();
        let maxPts = 500;
        let debugTick = 0;
        let paused = false;

        function fmt(x){ return (Math.round(x*1000)/1000).toFixed(3); }

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
            for(const ext of controlled){
                busToIdxModsP.set(ext, k);
                modsData.push({ x: [], y: [], name: `dP${ext} [MW]`, mode: 'lines', line: {shape:'hv'} });
                k += 1;
                busToIdxModsQ.set(ext, k);
                modsData.push({ x: [], y: [], name: `dQ${ext} [MVAr]`, mode: 'lines', line: {dash:'dot', shape:'hv'} });
                k += 1;
            }
            const modsLayout = { title: 'Active Mods dP/dQ', yaxis: {range:[-0.2, 0.2], zeroline:true}, legend: {orientation:'h'} };
            Plotly.newPlot(modsChartDiv, modsData, modsLayout, {displaylogo:false, responsive:true});

            // Breaker plot (single binary trace: 1=closed, 0=open)
            const breakerData = [ { x: [], y: [], name: 'K4 closed', mode: 'lines', line: { shape: 'hv' } } ];
            const breakerLayout = { title: 'Breaker state (0=open, 1=closed)', yaxis: { range: [-0.2, 1.2], dtick: 1 }, margin: { t: 30 }, showlegend: false };
            Plotly.newPlot(breakerChartDiv, breakerData, breakerLayout, {displaylogo:false, responsive:true});
        }

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
            pfLbl.textContent = js.pf_rate_hz;
            scLbl.textContent = js.scenario_rate_hz;
            const br = js.breaker || {present:false, closed:true};
            k4Lbl.textContent = br.present ? (br.closed ? 'Closed' : 'Open') : 'Not present';
            if(controlled.length === 0){
                controlled = js.controlled_buses || [];
                // fill select
                busSel.innerHTML = controlled.map(b => `<option value=\"${b}\">${b}</option>`).join('');
                initPlot(js.buses || controlled);
                console.log('Controlled buses from /status:', controlled);
            }
        }

        let sampleIdx = 0;

        async function refreshState(){
            if(paused){ return; }
            const r = await fetch('/state');
            const js = await r.json();
            rowLbl.textContent = js.row_idx;
            const mags = js.voltages_mag || {};
            const yUpdate = [];
            const xUpdate = [];
            const indices = [];
            const buses = Object.keys(mags).map(k => parseInt(k.match(/V(\\d+)_mag/)?.[1] || '0', 10)).filter(Boolean);
            for(const ext of (js.buses || buses)){
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

            // Mods plot
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

            // Breaker plot (stub)
            const br = js.breaker || { present: false, closed: true };
            const brVal = (br.present === false) ? null : (br.closed ? 1 : 0);
            Plotly.extendTraces(breakerChartDiv, { x: [[ sampleIdx ]], y: [[ brVal ]] }, [0], maxPts);
        }

        pEl.addEventListener('input', ()=> pLbl.textContent = fmt(parseFloat(pEl.value)) );
        qEl.addEventListener('input', ()=> qLbl.textContent = fmt(parseFloat(qEl.value)) );

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
            if(Number.isFinite(v) && v > 0){
                maxPts = v;
                trimTraces();
            }
        });

        // Breaker stub
        k4Btn.addEventListener('click', async ()=>{ setTimeout(refreshStatus, 150); });

        pauseBtn.addEventListener('click', ()=>{
            paused = !paused;
            pauseBtn.textContent = paused ? 'Resume Updates' : 'Pause Updates';
            if(!paused){ try { refreshState(); } catch(_){} }
        });

        // boot
        refreshStatus();
        setInterval(refreshStatus, 2000);
        setInterval(refreshState, 1000/20);
    </script>
</body>
</html>
                """)

    @app.get("/breaker")
    def get_breaker():
        return {"name": "K4", "present": False, "closed": True}

    @app.post("/breaker")
    def set_breaker(closed: bool):
        return {"ok": True, "closed": True}

    @app.post("/breaker/toggle")
    def toggle_breaker():
        return {"ok": True, "closed": True}

    @app.get("/ui")
    def ui():
        return root()

    return app


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opcua-url", default=os.environ.get("OPCUA_URL"), required=not os.environ.get("OPCUA_URL"),
                    help="OPC UA endpoint of the NI VeriStand gateway, e.g. opc.tcp://<gateway-ip>:<port> (or env OPCUA_URL)")
    ap.add_argument("--pf-rate", type=float, default=50.0, help="OPC poll rate [Hz]")
    ap.add_argument("--scenario-rate", type=float, default=1.0, help="Setpoint generation rate [Hz]")
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--muP", type=float, default=800.0)
    ap.add_argument("--sigmaP", type=float, default=50)
    ap.add_argument("--muQ", type=float, default=0.0)
    ap.add_argument("--sigmaQ", type=float, default=10)
    ap.add_argument("--corr", type=float, default=0.85)
    ap.add_argument("--P-scale", type=float, default=0.5)
    ap.add_argument("--Q-scale", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    rt = RealTimeOpcua(
        opc_url=args.opcua_url,
        pf_rate_hz=args.pf_rate,
        scenario_rate_hz=args.scenario_rate,
        alpha=args.alpha,
        muP=args.muP,
        sigmaP=args.sigmaP,
        muQ=args.muQ,
        sigmaQ=args.sigmaQ,
        corr=args.corr,
        P_scale=args.P_scale,
        Q_scale=args.Q_scale,
        seed=args.seed,
    )
    rt.start()
    app = create_app(rt)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
