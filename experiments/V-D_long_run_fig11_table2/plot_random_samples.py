#!/usr/bin/env python3
"""Plot recorded signals in the experiments/V-D_online_resilience_fig10/sequence_breaker_noise.py layout.

    .venv/bin/python experiments/V-D_long_run_fig11_table2/plot_random_samples.py
    .venv/bin/python experiments/V-D_long_run_fig11_table2/plot_random_samples.py --csv results/V-D_long_run_fig11_table2/random_samples.csv --no-show

Show maximum network voltage and total measured P and Q.
Power totals include every S<bus>_complex column
(controlled and uncontrolled buses), excluding orig_ baseline columns.
This script only reads the CSV; it does not contact the API.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / 'results' / 'V-D_long_run_fig11_table2'


def applied_control_values(samples):
    """Estimate applied controls from measured power minus current baseline.

    Only compare within one unchanged baseline segment, after the first
    observed PF power change. Values during baseline/PF skew are not reliable.
    """
    values = []
    for bus in (55, 57, 64):
        difference = samples[f'S{bus}_complex'].map(complex) - samples[f'orig_S{bus}_complex'].map(complex)
        values.append(np.column_stack((difference.map(lambda z: z.real), difference.map(lambda z: z.imag))))
    return np.stack(values, axis=1)


def recomputation_filter(samples, block_samples=2):
    """After a P/Q step, block N following samples, then await a voltage change."""
    if block_samples not in range(1, 6):
        raise ValueError('block_samples must be between 1 and 5')
    columns = [c for c in samples if re.fullmatch(r'orig_S\d+_complex', c)]
    if not columns or 'ts' not in samples:
        raise ValueError('Filtering requires baseline orig_S columns and ts')
    baseline = samples[columns].apply(lambda column: column.map(complex))
    p = baseline.apply(lambda column: column.map(lambda z: z.real))
    q = baseline.apply(lambda column: column.map(lambda z: z.imag))
    changed = (p.diff().abs().gt(1e-8).any(axis=1)
               | q.diff().abs().gt(1e-8).any(axis=1)).to_numpy()
    voltage_changed = samples['max_voltage'].diff().abs().gt(1e-8).to_numpy()
    ts = samples['ts'].to_numpy(dtype=float)
    keep = np.ones(len(samples), dtype=bool)
    events = []
    start, waiting, event = 0, False, None
    for i in range(len(samples)):
        if changed[i]:
            start, waiting = i, True
            event = {'sample_index': int(i), 'change_ts': float(ts[i]),
                     'observed_calculation_delay_s': None, 'filtered_delay_s': None}
            events.append(event)
        # Exclude the step and N subsequent samples. Changes within that
        # block do not qualify; require a new adjacent-sample voltage change.
        if waiting and i > start + block_samples and voltage_changed[i]:
            waiting = False
            event['observed_calculation_delay_s'] = float(ts[i] - ts[start])
            event['filtered_delay_s'] = float(ts[i] - ts[start])
        keep[i] = not waiting
    return pd.Series(~keep, index=samples.index), events


def summarize(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {'count': 0, 'mean': None, 'std': None, 'min': None, 'max': None, 'p95': None}
    return dict(count=int(len(values)), mean=float(values.mean()), std=float(values.std()),
                min=float(values.min()), max=float(values.max()), p95=float(np.quantile(values, .95)))


def violation_figures(samples):
    """Suppress violations at each P step and the following four readings."""
    columns = [c for c in samples if re.fullmatch(r'orig_S\d+_complex', c)]
    if not columns:
        raise ValueError('Violation blocking requires baseline orig_S<bus>_complex columns')
    active = samples[columns].apply(lambda c: c.map(lambda z: complex(z).real))
    changed = active.diff().abs().gt(1e-8).any(axis=1)
    blocked = changed.rolling(5, min_periods=1).max().astype(bool)
    violations = samples['max_voltage'] > 1.05
    n = len(samples)
    exceedance = (samples['max_voltage'] - 1.05).clip(lower=0)

    def metrics(error):
        return {
            'violation_count': int((error > 0).sum()),
            'violation_fraction': float((error > 0).sum()/n),
            'violation_error_pu': summarize(error),
            'violation_error_rmse_pu': float(np.sqrt((error**2).mean())),
            'error_during_violations_pu': summarize(error[error > 0]),
        }
    return {
        'sample_count': n,
        'active_power_change_count': int(changed.sum()),
        'blocked_sample_count': int(blocked.sum()),
        'suppressed_violation_count': int((violations & blocked).sum()),
        'without_block': metrics(exceedance),
        'with_5_sample_block': metrics(exceedance.mask(blocked, 0.0)),
    }


def variant_statistics(samples):
    error = samples['max_voltage'] - 1.05
    return {
        'sample_count': len(samples),
        'voltage_error_pu': summarize(error),
        'voltage_mae_pu': float(error.abs().mean()) if len(error) else None,
        'voltage_rmse_pu': float(np.sqrt((error**2).mean())) if len(error) else None,
        'overvoltage_pu': summarize(error.clip(lower=0)),
        'violation_fraction': float((error > 0).mean()) if len(error) else None,
        'dP_mw': summarize(samples['total_dP_mw']),
        'dQ_mvar': summarize(samples['total_dQ_mvar']),
    }


def _rel(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', type=Path, default=OUT_DIR / 'random_samples.csv')
    parser.add_argument('--comparison-dir', type=Path, help='Read off.csv and on.csv from a comparison recording directory')
    parser.add_argument('--out', type=Path, help='Output filename stem (default: CSV path without suffix)')
    parser.add_argument('--no-show', action='store_true', help='Save plots without opening a window')
    args = parser.parse_args(argv)

    import matplotlib
    if args.no_show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if args.comparison_dir:
        parts = []
        for filename, label in (('off.csv', 'Controller OFF'), ('on.csv', 'PISR ON')):
            path = args.comparison_dir / filename
            if not path.exists() and path.with_suffix('.csv.gz').exists():
                path = path.with_suffix('.csv.gz')   # compressed copy as shipped in the repository
            part = pd.read_csv(path)
            if part.empty:
                parser.error(f'{filename} contains no samples')
            part['run'] = label
            if 'mode' in part:
                part = part.drop(columns='mode')
            parts.append(part)
        df = pd.concat(parts, ignore_index=True)
        if args.out is None:
            args.out = args.comparison_dir / 'comparison'
        plan_path = args.comparison_dir / 'sweep.json'
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            for phase in ('off', 'on'):
                if not plan.get(f'{phase}_complete'):
                    print(f'Warning: {phase.upper()} recording is not marked complete; plotting available samples only.')
    else:
        if not args.csv.exists() and Path(f'{args.csv}.gz').exists():
            args.csv = Path(f'{args.csv}.gz')        # compressed copy as shipped in the repository
        df = pd.read_csv(args.csv)
    if df.empty:
        parser.error('The CSV contains no samples')
    required = {'max_voltage', 'total_dP_mw', 'total_dQ_mvar'}
    missing = required - set(df.columns)
    if missing:
        parser.error(f"Missing CSV columns: {', '.join(sorted(missing))}")
    power_columns = [c for c in df.columns if re.fullmatch(r'S\d+_complex', c)]
    if not power_columns:
        parser.error('The CSV contains no measured S<bus>_complex power columns')
    time_col = next((c for c in ('t_rel', 'ts') if c in df.columns), None)
    if time_col is None:
        parser.error('The CSV must contain t_rel or ts timestamps')

    # Preserve the sequence figure's physical footprint for paper placement.
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 11,
                         'xtick.labelsize': 10, 'ytick.labelsize': 10,
                         'lines.linewidth': 1.3})
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    group_col = next((c for c in ('mode', 'run') if c in df.columns), None)
    groups = df.groupby(group_col, sort=False, dropna=False) if group_col else [('PISR', df)]
    statistics = {}
    first = True
    max_time = 0.0
    for name, samples in groups:
        samples = samples.sort_values(time_col)
        times = pd.to_numeric(samples[time_col], errors='raise')
        times = times - times.iloc[0]
        max_time = max(max_time, float(times.iloc[-1]))
        print(f'{name}: loaded {len(samples)} samples spanning {times.iloc[-1]:.2f}s from {args.comparison_dir or args.csv}')
        color = f'C{len(statistics)}'
        try:
            figures = violation_figures(samples)
        except ValueError as exc:
            parser.error(str(exc))
        statistics[str(name)] = {
            'high_sampling': variant_statistics(samples),
            'violation_comparison': figures,
        }
        raw, blocked = figures['without_block'], figures['with_5_sample_block']
        print(f"{name}: violations without block: {raw['violation_count']} ({raw['violation_fraction']:.2%}); "
              f"with 5-sample P-change block: {blocked['violation_count']} ({blocked['violation_fraction']:.2%})")
        first = False
        axes[0].plot(times, samples['max_voltage'], label=str(name), color=color)

        total_power = samples[power_columns].apply(lambda column: column.map(complex)).sum(axis=1)
        axes[1].plot(times, total_power.map(lambda z: z.real) * 1000,
                     label=f'{name}: all prosumers P' if args.comparison_dir else 'All prosumers: total P', color=color)
        axes[2].plot(times, total_power.map(lambda z: z.imag) * 1000,
                     label=f'{name}: all prosumers Q' if args.comparison_dir else 'All prosumers: total Q', color=color)
    axes[0].axhline(1.05, color='r', linestyle='--', label='Voltage Limit', zorder=10)
    for ax, ylabel in zip(axes, ('Max voltage in p.u.', 'Total P in kW', 'Total Q in kvar')):
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=10, loc='upper right', framealpha=1, 
                  borderpad=0.3, labelspacing=0.25)
        ax.grid(True)
    axes[2].set_xlabel('Time in s')
    if max_time > 0:
        axes[2].set_xlim(0, max_time)
    fig.tight_layout(pad=0.8, h_pad=0.6)
    stem = args.out if args.out else Path(str(args.csv).removesuffix('.gz')).with_suffix('')
    stem.parent.mkdir(parents=True, exist_ok=True)
    stats_path = Path(f'{stem}.stats.json')
    stats_path.write_text(json.dumps({
        'source_csv': [_rel(args.comparison_dir / f) for f in ('off.csv', 'on.csv')] if args.comparison_dir else _rel(args.csv),
        'voltage_limit_pu': 1.05,
        'violation_block_samples': 5,
        'definitions': {
            'violation_error': 'max(max_voltage - 1.05, 0) p.u.; blocked violations become zero. Mean and RMSE use all samples. error_during_violations_pu uses only positive, unsuppressed errors.',
            'voltage_error': 'max_voltage - 1.05; statistics are sample-weighted',
            'violation_block': 'Suppress violations at each baseline active-power change sample and the next four samples. Overlapping blocks merge. Q-only changes do not trigger blocking. Voltage data and plotted curves remain unchanged.',
            'violation_fraction': 'Violation count divided by ALL recorded samples in both variants; blocking suppresses violations without shrinking the denominator.',
            'active_power_change_tolerance_mw': 1e-8,
            'control': 'Total adjustments on controlled buses, MW/MVAr, original API signs',
        }, 'runs': statistics,
    }, indent=2, allow_nan=False) + '\n')
    print(f'Saved {stats_path}')
    for extension in ('png', 'eps'):
        path = Path(f'{stem}.{extension}')
        fig.savefig(path, dpi=300)
        print(f'Saved {path}')
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
