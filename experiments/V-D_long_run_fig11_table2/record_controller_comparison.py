#!/usr/bin/env python3
"""Record identical API-selected scenario rows with controller OFF then ON.

Each run sets its start row once through /scenario/row with frozen=false.
The service advances scenarios at its own rate; recording is continuous at 20 Hz.
No per-row commands or settling waits are issued.
No service or controller process is started; breaker configuration is untouched.
New recordings overwrite previous CSVs and sweep metadata in the output directory.

    .venv/bin/python experiments/V-D_long_run_fig11_table2/record_controller_comparison.py
    .venv/bin/python experiments/V-D_long_run_fig11_table2/record_controller_comparison.py --phase off-twice
    # Or run separately, enabling PISR manually between commands:
    .venv/bin/python experiments/V-D_long_run_fig11_table2/record_controller_comparison.py --phase off
    .venv/bin/python experiments/V-D_long_run_fig11_table2/record_controller_comparison.py --phase on
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import requests

from record_random_samples import OUT_DIR, flatten, positive


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-url', default='http://localhost:8000')
    parser.add_argument('--duration', type=positive, default=3600)
    parser.add_argument('--rate', type=positive, default=20)
    parser.add_argument('--start-row', type=int, default=1000)
    parser.add_argument('--phase', choices=('both', 'off', 'on', 'off-twice'), default='both')
    parser.add_argument('--out', type=Path, default=OUT_DIR / 'controller_comparison')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / 'sweep.json'
    with requests.Session() as session:
        def api(method, path, body=None):
            response = session.request(method, args.base_url.rstrip('/') + path, json=body, timeout=5)
            response.raise_for_status()
            return response.json()

        status = api('GET', '/status')
        scenario = api('GET', '/scenario')
        if '/scenario/row' not in api('GET', '/openapi.json')['paths']:
            raise RuntimeError('Natural sweep reset requires /scenario/row')
        def mode_matches(state, wanted):
            if 'control_router' in state:
                return state['control_router']['mode'] == wanted
            if state['solver']['mode'] != 'pf':
                return False
            if wanted == 'none':
                return all(abs(float(v.get(k, 0))) < 1e-10
                           for b, v in state.get('mods', {}).items() if int(b) in (55,57,64)
                           for k in ('dP_mw','dQ_mvar'))
            return str(state.get('cost', {}).get('source', '')).lower() in ('pisr', 'julia')
        if args.phase == 'on':
            plan = json.loads(manifest_path.read_text())
            if not plan.get('off_complete'):
                raise RuntimeError('Complete the OFF recording first')
        else:
            if args.start_row < 0 or args.start_row >= scenario['n_rows']:
                parser.error('--start-row is outside the scenario table')
            plan = {'base_url': args.base_url, 'rate': args.rate,
                    'start_row': args.start_row, 'duration': args.duration,
                    'scenario_rate_hz': status['scenario_rate_hz'],
                    'n_rows': scenario['n_rows'], 'sweep_mode': 'natural',
                    'breaker': status['breaker'], 'off_complete': False,
                    'noise': api('GET', '/noise')}
        if plan.get('sweep_mode') != 'natural':
            raise RuntimeError('Previous OFF used manual row stepping. Record a new OFF sweep first.')
        if status['scenario_rate_hz'] != plan['scenario_rate_hz'] or scenario['n_rows'] != plan['n_rows']:
            raise RuntimeError('Scenario rate/table length differs from the OFF sweep')
        if args.base_url != plan['base_url']:
            raise RuntimeError('ON must use the same API as OFF')
        original_noise = api('GET', '/noise')

        def save_plan():
            manifest_path.write_text(json.dumps(plan, indent=2) + '\n')

        def run(phase):
            wanted = 'none' if phase in ('off', 'off_repeat') else 'pisr'
            state = api('GET', '/state')
            if not mode_matches(state, wanted):
                raise RuntimeError(f'Expected controller {wanted}; check manual controller state and adjustments')
            if state['breaker'] != plan['breaker']:
                raise RuntimeError('Breaker differs from OFF configuration')
            if state['noise']['enabled']:
                raise RuntimeError('Disable measurement noise before recording')
            path = args.out / f'{phase}.csv'
            # Invalidate previous results before replacing this phase. A new
            # OFF run must never be paired with an old ON/repeat recording.
            if phase == 'off':
                for name in ('on.csv', 'off_repeat.csv', 'combined.csv'):
                    (args.out / name).unlink(missing_ok=True)
            else:
                (args.out / 'combined.csv').unlink(missing_ok=True)
            plan[f'{phase}_complete'] = False
            save_plan()
            index = 0
            with path.open('w', newline='') as stream:
                writer = None
                api('POST', '/scenario/row', {'row': plan['start_row'], 'frozen': False})
                start = time.monotonic()
                slot = 0
                print(f"{phase.upper()}: natural sweep from row {plan['start_row']} for {plan['duration']:g}s", flush=True)
                while slot / plan['rate'] < plan['duration']:
                    time.sleep(max(0, start + slot / plan['rate'] - time.monotonic()))
                    before = time.monotonic()
                    if before - start >= plan['duration']:
                        break
                    state = api('GET', '/state')
                    if not mode_matches(state, wanted) or state['breaker'] != plan['breaker']:
                        raise RuntimeError('Controller mode or breaker changed during recording')
                    if state['noise']['enabled'] or state['scenario']['frozen']:
                        raise RuntimeError('Noise enabled or scenario frozen during recording')
                    record = flatten(state, index, before - start, time.monotonic() - before)
                    record.update(run=('Uncontrolled repeat' if phase == 'off_repeat' else 'Uncontrolled')
                                  if wanted == 'none' else 'PISR', wall_elapsed_s=time.monotonic()-start)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(record))
                        writer.writeheader()
                    writer.writerow(record)
                    stream.flush()
                    index += 1
                    slot = max(slot+1, math.ceil((time.monotonic()-start)*plan['rate']))
            plan[f'{phase}_complete'] = True
            save_plan()
            print(f'Saved {index} samples: {path}', flush=True)

        try:
            api('POST', '/noise', {'enabled': False})
            if args.phase in ('both', 'off', 'off-twice'):
                # Explicitly select OFF; no external controller process is stopped.
                if 'control_router' in status:
                    api('POST', '/control_router', {'mode': 'none'})
                save_plan()
                run('off')
                if args.phase == 'both':
                    input('OFF saved. Start/enable PISR manually, then press Enter to replay the same rows: ')
            if args.phase == 'off-twice':
                run('off_repeat')
                import pandas as pd
                pd.concat([pd.read_csv(args.out/'off.csv'), pd.read_csv(args.out/'off_repeat.csv')],
                          ignore_index=True).to_csv(args.out/'combined.csv', index=False)
            if args.phase in ('both', 'on'):
                run('on')
                import pandas as pd
                pd.concat([pd.read_csv(args.out/'off.csv'), pd.read_csv(args.out/'on.csv')],
                          ignore_index=True).to_csv(args.out/'combined.csv', index=False)
        finally:
            api('POST', '/noise', original_noise)


if __name__ == '__main__':
    main()
