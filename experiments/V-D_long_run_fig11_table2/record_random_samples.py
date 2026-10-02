#!/usr/bin/env python3
"""Record the existing powerflow/controller API without changing its configuration.

    .venv/bin/python experiments/V-D_long_run_fig11_table2/record_random_samples.py --duration 20
    .venv/bin/python experiments/V-D_long_run_fig11_table2/record_random_samples.py --duration 900

Only GET /state is used. The running service supplies the random scenarios,
Bus 61 operating point, and controller actions. No processes are started,
no noise is added, and no breaker/controller/scenario settings are changed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / 'results' / 'V-D_long_run_fig11_table2'


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('must be finite and positive')
    return value


def flatten(state, index, elapsed, latency):
    mods = state.get('mods', {})
    rec = {'i': index, 'ts': time.time(), 't_rel': elapsed,
           'request_s': latency, 'row': state.get('row_idx'), 'run': 'PISR',
           'breaker_closed': state.get('breaker', {}).get('closed'),
           'cost': json.dumps(state.get('cost', {}))}
    rec.update(state['voltages'])
    rec.update(state['power'])
    rec.update({'orig_' + k: v for k, v in state.get('power_orig', {}).items()})
    rec['mods'] = json.dumps(mods, sort_keys=True)
    for bus in (55, 57, 64):
        action = mods.get(str(bus), {})
        rec[f'dP{bus}_mw'] = action.get('dP_mw', 0.0)
        rec[f'dQ{bus}_mvar'] = action.get('dQ_mvar', 0.0)
    # Controller buses only: exclude the uncontrolled Bus 61 disturbance.
    rec['total_dP_mw'] = sum(rec[f'dP{bus}_mw'] for bus in (55, 57, 64))
    rec['total_dQ_mvar'] = sum(rec[f'dQ{bus}_mvar'] for bus in (55, 57, 64))
    rec['max_voltage'] = max(abs(complex(v)) for v in state['voltages'].values())
    rec['bus61_p_mw'] = complex(state['power']['S61_complex']).real
    return rec


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-url', default='http://localhost:8000')
    parser.add_argument('--duration', type=positive, default=3600.0)
    parser.add_argument('--rate', type=positive, default=20.0)
    parser.add_argument('--out', type=Path, default=OUT_DIR / 'random_samples.csv')
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    count, missed, bus61_sum, violations = 0, 0, 0.0, 0
    with requests.Session() as session:
        def get_state():
            response = session.get(args.base_url.rstrip('/') + '/state', timeout=5)
            response.raise_for_status()
            state = response.json()
            if state.get('noise', {}).get('enabled'):
                raise RuntimeError('API measurement noise is enabled; recording requires noise off')
            return state

        initial = get_state()
        flatten(initial, 0, 0, 0)  # Validate signals before opening the output.
        breaker = initial.get('breaker')
        print(f'Recording existing API {args.base_url}: {args.duration:g}s at {args.rate:g} Hz', flush=True)
        try:
            with args.out.open('w', newline='') as stream:
                writer = None
                start = time.monotonic()
                slot = 0
                while slot / args.rate < args.duration:
                    time.sleep(max(0, start + slot / args.rate - time.monotonic()))
                    before = time.monotonic()
                    if before - start >= args.duration:
                        break
                    state = get_state()
                    if state.get('breaker') != breaker:
                        raise RuntimeError('Breaker configuration changed externally during recording')
                    rec = flatten(state, count, before - start, time.monotonic() - before)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(rec))
                        writer.writeheader()
                    writer.writerow(rec)
                    stream.flush()
                    count += 1
                    bus61_sum += rec['bus61_p_mw']
                    violations += rec['max_voltage'] > 1.05
                    next_slot = max(slot + 1, math.ceil((time.monotonic() - start) * args.rate))
                    missed += min(next_slot, math.ceil(args.duration * args.rate)) - slot - 1
                    slot = next_slot
        except KeyboardInterrupt:
            print('Interrupted; completed samples retained.')
        finally:
            if count:
                print(f'Saved {count} samples to {args.out}; missed slots: {missed}')
                print(f'Measured Bus 61 mean: {bus61_sum/count:.6f} MW; '
                      f'max voltage > 1.05 p.u.: {100*violations/count:.1f}% of samples')


if __name__ == '__main__':
    main()
