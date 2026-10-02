#!/usr/bin/env python3
from __future__ import annotations
import os
import time
import argparse
import requests
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_OUT = os.path.join(BASE_DIR, 'results', 'V-D_online_resilience_fig10')

BASE_URL = 'http://127.0.0.1:8012'
RATE = 20.0

# same simple sequence (2s step) as the original tool
SEQUENCE = [
    {"start": 1.0, "duration": 2.0, "buses": [61], "dP_mw": -0.05, "dQ_mvar": -0.05},
]

session = requests.Session()


def post_mods(items: List[Dict[str, Any]]):
    try:
        session.post(f"{BASE_URL}/mods", json={"items": items}, timeout=5.0)
    except Exception as e:
        print("post_mods failed:", e)


def del_mods():
    try:
        session.delete(f"{BASE_URL}/mods", timeout=5.0)
    except Exception as e:
        print("delete mods failed:", e)


def set_breaker(closed: bool) -> Dict[str, Any]:
    try:
        r = session.post(f"{BASE_URL}/breaker", params={"closed": str(bool(closed)).lower()}, timeout=5.0)
        try:
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict):
                return {"ok": False}
            return data
        except Exception:
            return {"ok": False}
    except Exception as e:
        print("set_breaker failed:", e)
        return {"ok": False}


# control_router mode of the routed service for each plotted controller
ROUTER_MODE = {'Uncontrolled': 'none', 'Robust Sensitivity': 'sensitivity', 'PISR': 'pisr'}


def set_router(mode: str, settle_s: float = 2.0):
    r = session.post(f"{BASE_URL}/control_router", json={"mode": mode}, timeout=5.0)
    r.raise_for_status()
    time.sleep(settle_s)


def get_breaker() -> Dict[str, Any]:
    try:
        r = session.get(f"{BASE_URL}/breaker", timeout=5.0); r.raise_for_status(); return r.json()
    except Exception as e:
        print("get_breaker failed:", e)
        return {"name": "K4", "present": False, "closed": True}


def set_noise(enabled: bool = True, v_noise_std: float = 0.005, s_noise_std: float = 0.00001) -> Dict[str, Any]:
    body = {"enabled": enabled, "v_noise_std": float(v_noise_std), "s_noise_std": float(s_noise_std)}
    try:
        r = session.post(f"{BASE_URL}/noise", json=body, timeout=5.0)
        try:
            return r.json()
        except Exception:
            return {}
    except Exception as e:
        print("set_noise failed:", e)
        return {}


def get_state() -> Dict[str, Any]:
    try:
        r = session.get(f"{BASE_URL}/state", timeout=5.0); r.raise_for_status(); return r.json()
    except Exception:
        return {'row_idx': None, 'voltages': {}, 'power': {}, 'mods': {}}


def parse_complex(v):
    if v is None:
        return complex(0,0)
    if isinstance(v, complex):
        return v
    try:
        return complex(v)
    except Exception:
        try:
            return complex(str(v).strip())
        except Exception:
            return complex(0,0)


def run_sequence(run_name: str, rate: float = RATE) -> pd.DataFrame:
    """Run the disturbance sequence and return a DataFrame of the recorded samples."""
    print(f"Starting run: {run_name}")
    # clear any existing mods first
    del_mods()
    time.sleep(0.2)

    rate = max(0.1, float(rate))
    total_time = max(ev['start']+ev['duration'] for ev in SEQUENCE) + 2.0
    samples = int(total_time * rate) + 1
    actions = {}
    for ev in SEQUENCE:
        s = int(ev['start'] * rate); e = int((ev['start']+ev['duration']) * rate)
        items = [{"bus_ext": int(b), "dP_mw": float(ev.get('dP_mw',0.0)), "dQ_mvar": float(ev.get('dQ_mvar',0.0))} for b in ev['buses']]
        clear_items = [{"bus_ext": int(b), "dP_mw": 0.0, "dQ_mvar": 0.0} for b in ev['buses']]
        actions.setdefault(s, []).append(('set', items))
        actions.setdefault(e, []).append(('set', clear_items))

    records = []
    for i in range(samples):
        t0 = time.perf_counter()
        if i in actions:
            for typ, items in actions[i]:
                post_mods(items)
                print(f"@{i}: {typ} {items}")
        st = get_state()
        rec = {'i': i, 'ts': time.time(), 'row': st.get('row_idx')}
        rec.update(st.get('voltages', {}) or {})
        rec.update(st.get('power', {}) or {})
        rec['mods'] = st.get('mods', {})
        records.append(rec)
        dt = time.perf_counter() - t0; time.sleep(max(0, 1.0/rate - dt))

    df = pd.DataFrame(records)
    # convert complex-like columns
    for c in list(df.columns):
        if c.endswith('_complex'):
            df[c] = df[c].apply(parse_complex)
    # compute max voltage
    vcols = [c for c in df.columns if c.startswith('V') and c.endswith('_complex')]
    if vcols:
        df['max_voltage'] = df[vcols].apply(lambda row: max((abs(parse_complex(x)) for x in row)), axis=1)
    else:
        df['max_voltage'] = 0.0
    # compute total_dP/dQ from mods if available
    def sum_mods(mods):
        if not mods:
            return (0.0, 0.0)
        if isinstance(mods, dict):
            try:
                total_p = 0.0; total_q = 0.0
                for v in mods.values():
                    total_p += float(v.get('dP_mw', 0.0))
                    total_q += float(v.get('dQ_mvar', 0.0))
                return (total_p, total_q)
            except Exception:
                return (0.0, 0.0)
        try:
            total_p = 0.0; total_q = 0.0
            for it in mods:
                total_p += float(it.get('dP_mw', 0.0))
                total_q += float(it.get('dQ_mvar', 0.0))
            return (total_p, total_q)
        except Exception:
            return (0.0, 0.0)

    sums = df['mods'].apply(sum_mods)
    df['total_dP_mw'] = sums.apply(lambda x: x[0])
    df['total_dQ_mvar'] = sums.apply(lambda x: x[1])

    # add run name column
    df['run'] = run_name
    print(f"Finished run: {run_name}, samples: {len(df)}")
    return df


def save_and_plot(all_runs: Dict[str, pd.DataFrame], out_dir: str = BASE_OUT, break_times: List[float] | None = None):
    # combine for CSV
    combined = []
    for name, df in all_runs.items():
        d = df.copy(); d['mode'] = name; combined.append(d)
    comb_df = pd.concat(combined, ignore_index=True)
    comb_csv = os.path.join(out_dir, 'sequence_breaker_noise_combined.csv')
    comb_df.to_csv(comb_csv, index=False)
    print('Saved combined CSV', comb_csv)

    # Wider figure (twice as wide)
    plt.figure(figsize=(12,6))
    ax1 = plt.subplot(3,1,1)
    ax1.axhline(1.05, color='r', linestyle='--', label='Voltage Limit')
    # plot max_voltage per mode
    for name, df in all_runs.items():
        if 'ts' in df.columns and not df['ts'].isnull().all():
            t0 = float(df['ts'].iloc[0]); times = df['ts'].apply(lambda x: float(x) - t0)
        else:
            times = pd.Series([i / float(RATE) for i in range(len(df))])
        if 'max_voltage' in df.columns:
            ax1.plot(times, df['max_voltage'], label=name)
    # dashed vertical lines for breaker events
    if break_times:
        for bt in break_times:
            ax1.axvline(bt, color='blue', linestyle='--', linewidth=1)
    ax1.set_ylabel('Max voltage in p.u.')
    ax1.legend(fontsize='small',loc='upper right')
    ax1.grid(True)

    ax2 = plt.subplot(3,1,2, sharex=ax1)
    # disturbance from first run if present
    first_df = next(iter(all_runs.values())) if all_runs else None
    if first_df is not None:
        if 'ts' in first_df.columns and not first_df['ts'].isnull().all():
            t0 = float(first_df['ts'].iloc[0]); times0 = first_df['ts'].apply(lambda x: float(x) - t0)
        else:
            times0 = pd.Series([i / float(RATE) for i in range(len(first_df))])
        if 'disturbance_dP_mw' in first_df.columns:
            ax2.plot(times0, first_df['disturbance_dP_mw']*1000, color='k', linestyle=':', linewidth=2, label='disturbance dP')
    # plot total_dP from runs
    for name, df in all_runs.items():
        if 'ts' in df.columns and not df['ts'].isnull().all():
            t0 = float(df['ts'].iloc[0]); times = df['ts'].apply(lambda x: float(x) - t0)
        else:
            times = pd.Series([i / float(RATE) for i in range(len(df))])
        if 'total_dP_mw' in df.columns:
            ax2.plot(times, df['total_dP_mw']*1000, label=name)
    # dashed verticals on bottom axis too
    if break_times:
        for bt in break_times:
            ax2.axvline(bt, color='blue', linestyle='--', linewidth=1)

    # make it visibly clear what the vertical dashed lines mean (label Q4 open/close)
    if break_times:
        # ensure limits are up to date
        y1_top = ax1.get_ylim()[1]
        y2_top = ax2.get_ylim()[1]
        labels = []
        # if two times provided assume [open_time, close_time]
        #if len(break_times) >= 2:
        #    labels = ['Q4 Open', 'Q4 Closed']
        #else:
        #    labels = ['Q4 Toggle']
        #for i, bt in enumerate(break_times):
        #    lab = labels[i] if i < len(labels) else f'Q4 {i}'
        #    # annotate on top subplot
        #    ax1.annotate(lab, xy=(bt, y1_top), xytext=(bt, y1_top*0.995), rotation=90,
        #                 va='top', ha='right', fontsize=8, color='gray', backgroundcolor='white')
        #    # annotate on bottom subplot
        #    ax2.annotate(lab, xy=(bt, y2_top), xytext=(bt, y2_top*0.995), rotation=90,
        #                va='top', ha='right', fontsize=8, color='gray', backgroundcolor='white')
    ax2.set_ylabel('Total ΔP in kW')
    ax2.legend(fontsize='small',loc='upper right')
    ax2.grid(True)

    ax3 = plt.subplot(3,1,3, sharex=ax1)
    # plot total_dQ from runs
    for name, df in all_runs.items():
        if 'ts' in df.columns and not df['ts'].isnull().all():
            t0 = float(df['ts'].iloc[0]); times = df['ts'].apply(lambda x: float(x) - t0)
        else:
            times = pd.Series([i / float(RATE) for i in range(len(df))])
        if 'total_dQ_mvar' in df.columns:
            ax3.plot(times, df['total_dQ_mvar']*1000, label=name)
    # dashed verticals on bottom axis too
    if break_times:
        for bt in break_times:
            ax3.axvline(bt, color='blue', linestyle='--', linewidth=1)

    ax3.set_ylabel('Total ΔQ in kvar')
    ax3.legend(fontsize='small',loc='upper right')
    ax3.grid(True)

    plt.xlabel('Time in s')
    plt.tight_layout()
    png = os.path.join(out_dir, 'sequence_breaker_noise_comparison.png')
    plt.savefig(png)
    plt.savefig(os.path.join(out_dir, 'sequence_breaker_noise_comparison.eps'))
    print('Saved plot', png)
    plt.show()


def main():
    global BASE_URL, RATE
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default=BASE_URL)
    parser.add_argument('--rate', type=float, default=RATE)
    parser.add_argument('--no-prompt', action='store_true', help='Do not wait for user confirmation between modes')
    parser.add_argument('--manual-router', action='store_true',
                        help='Do not switch /control_router automatically; prompt the operator instead')
    parser.add_argument('--v-noise', type=float, default=0.005)
    parser.add_argument('--s-noise', type=float, default=0.00001)
    parser.add_argument('--out', default=BASE_OUT)
    parser.add_argument('--replot', action='store_true',
                        help='Only redraw the figure from the sequence_<mode>_threephases.csv files in --out')
    args = parser.parse_args()
    if args.replot:
        runs = {m: pd.read_csv(os.path.join(args.out, f'sequence_{m}_threephases.csv'))
                for m in ['Uncontrolled', 'Robust Sensitivity', 'PISR']}
        phase = max(ev['start'] + ev['duration'] for ev in SEQUENCE) + 2.0
        save_and_plot(runs, out_dir=args.out, break_times=[phase, 2.0 * phase])
        return

    BASE_URL = args.base_url
    RATE = args.rate
    os.makedirs(args.out, exist_ok=True)

    modes = ['Uncontrolled', 'Robust Sensitivity', 'PISR']
    all_runs: Dict[str, pd.DataFrame] = {}

    # We'll run the three phases sequentially for each mode and stitch them
    br = get_breaker()
    breaker_present = br.get('present', False)

    # precompute per-phase durations (seconds)
    phase_duration = max(ev['start'] + ev['duration'] for ev in SEQUENCE) + 2.0
    # times where we change breaker state relative to start of full run
    # We'll do: baseline (breaker open), then close it, then open again with noise
    break_times: List[float] = []

    for mode in modes:
        print(f"\n=== Running mode '{mode}' across three phases ===")
        if args.manual_router:
            if not args.no_prompt:
                input(f"Please start the '{mode}' controller now and press Enter to continue...")
        else:
            del_mods()
            set_router(ROUTER_MODE[mode])
            print(f"  control_router -> {ROUTER_MODE[mode]}")
        stitched_parts: List[pd.DataFrame] = []
        t_offset = 0.0

        # Phase 1: baseline (breaker open)
        print(' Phase: baseline (noise off, breaker as-is)')
        set_noise(enabled=False, v_noise_std=0.0, s_noise_std=0.0)
        # ensure breaker open during baseline so we can close later
        br_resp = set_breaker(False)
        time.sleep(0.2)
        if br_resp.get('ok'):
            print("  breaker set to open for baseline")
        else:
            print("  warning: failed to open breaker for baseline", br_resp)
        p1 = run_sequence(f'{mode}_baseline', rate=RATE)
        # add relative time column
        p1['t_rel'] = p1['i'] / float(RATE) + t_offset
        stitched_parts.append(p1)
        t_offset += phase_duration
        # breaker close event at t = phase_duration
        if breaker_present:
            break_times.append(phase_duration)

        # Phase 2: close breaker
        print(' Phase: breaker close (noise off)')
        br_resp = set_breaker(True)
        time.sleep(0.2)
        if br_resp.get('ok'):
            print("  breaker closed for middle phase")
        else:
            print("  warning: failed to close breaker", br_resp)
        p2 = run_sequence(f'{mode}_breaker_closed', rate=RATE)
        p2['t_rel'] = p2['i'] / float(RATE) + t_offset
        stitched_parts.append(p2)
        t_offset += phase_duration
        # breaker re-open event at t = 2*phase_duration
        break_times.append(2.0 * phase_duration)

        # Phase 3: reopen breaker and enable noise
        print(' Phase: breaker open + noise')
        br_resp = set_breaker(False)
        time.sleep(0.2)
        if br_resp.get('ok'):
            print("  breaker opened again for noisy phase")
        else:
            print("  warning: failed to open breaker before noisy phase", br_resp)
        set_noise(enabled=True, v_noise_std=float(args.v_noise), s_noise_std=float(args.s_noise))
        p3 = run_sequence(f'{mode}_breaker_open_noise', rate=RATE)
        p3['t_rel'] = p3['i'] / float(RATE) + t_offset
        stitched_parts.append(p3)
        t_offset += phase_duration

        # cleanup per mode
        set_noise(enabled=False, v_noise_std=0.0, s_noise_std=0.0)
        del_mods()

        # concatenate parts into one DataFrame for this mode
        full = pd.concat(stitched_parts, ignore_index=True)
        # use t_rel as the time base
        full = full.sort_values('t_rel').reset_index(drop=True)
        # save per-mode CSV
        fname = os.path.join(args.out, f'sequence_{mode}_threephases.csv')
        full.to_csv(fname, index=False)
        print('Saved', fname)
        all_runs[mode] = full

    if not args.manual_router:
        set_router('none', settle_s=0.0)

    # Create combined plot with vertical lines at break_times
    save_and_plot(all_runs, out_dir=args.out, break_times=break_times if breaker_present else None)


if __name__ == '__main__':
    main()
