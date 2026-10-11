#!/usr/bin/env python3
"""
Aggregate collect+train batch runs into a cross-run summary and precision boxplots.

This script reads a batch manifest from experiments/V-B_training_robustness_fig8/batch_collect_and_train.sh output,
loads each run's final metrics from training/metrics_progress.csv, and produces:

1) all_runs_final_metrics.csv
2) precision_boxplots.png and precision_boxplots.svg
3) summary_stats.csv

Usage examples:
  .venv/bin/python experiments/V-B_training_robustness_fig8/summarize_collect_train_runs.py \
      --batch-dir models/cross_eval

  .venv/bin/python experiments/V-B_training_robustness_fig8/summarize_collect_train_runs.py \
      --manifest models/cross_eval/manifest.csv
"""

from __future__ import annotations

import argparse
import math
import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_COLS = [
    "mean_mag_mae",
    "worst_mag_max",
    "mean_ang_mae",
    "worst_ang_max",
]

CROSS_METRIC_COLS = [
    "mag_rmse",
    "mag_mae",
    "mag_r2",
    "ang_rmse",
    "ang_mae",
    "ang_r2",
]


def _latest_batch_dir(repo_root: Path) -> Optional[Path]:
    base = repo_root / "results" / "V-B_training_robustness_fig8"
    candidates = sorted(base.glob("multi_collect_train_*"))
    if not candidates:
        return None
    return candidates[-1]


def _resolve_manifest(repo_root: Path, batch_dir: Optional[str], manifest: Optional[str]) -> Path:
    if manifest:
        p = Path(manifest)
        return p if p.is_absolute() else (repo_root / p)
    if batch_dir:
        b = Path(batch_dir)
        b_abs = b if b.is_absolute() else (repo_root / b)
        return b_abs / "manifest.csv"

    latest = _latest_batch_dir(repo_root)
    if latest is None:
        raise FileNotFoundError("No batch directory found under results/V-B_training_robustness_fig8/")
    return latest / "manifest.csv"


def _to_float_or_nan(value) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def _load_final_metrics(run_dir: Path) -> Dict[str, float]:
    metrics_path = run_dir / "training" / "metrics_progress.csv"
    if not metrics_path.exists():
        return {"iteration": math.nan, **{k: math.nan for k in METRIC_COLS}}

    df = pd.read_csv(metrics_path)
    if df.empty:
        return {"iteration": math.nan, **{k: math.nan for k in METRIC_COLS}}

    last = df.iloc[-1]
    out: Dict[str, float] = {
        "iteration": _to_float_or_nan(last.get("iteration")),
    }
    for c in METRIC_COLS:
        out[c] = _to_float_or_nan(last.get(c))
    return out


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for metric in METRIC_COLS:
        vals = pd.to_numeric(df[metric], errors="coerce").dropna()
        if vals.empty:
            rows.append(
                {
                    "metric": metric,
                    "count": 0,
                    "mean": math.nan,
                    "std": math.nan,
                    "min": math.nan,
                    "q25": math.nan,
                    "median": math.nan,
                    "q75": math.nan,
                    "max": math.nan,
                }
            )
            continue

        rows.append(
            {
                "metric": metric,
                "count": int(vals.shape[0]),
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if vals.shape[0] > 1 else 0.0,
                "min": float(vals.min()),
                "q25": float(vals.quantile(0.25)),
                "median": float(vals.quantile(0.5)),
                "q75": float(vals.quantile(0.75)),
                "max": float(vals.max()),
            }
        )
    return pd.DataFrame(rows)


def _make_boxplots(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.flatten()

    titles = {
        "mean_mag_mae": "Mean |V| MAE",
        "worst_mag_max": "Worst |V| Max Error",
        "mean_ang_mae": "Mean Angle MAE",
        "worst_ang_max": "Worst Angle Max Error",
    }

    for ax, metric in zip(axes, METRIC_COLS):
        vals = pd.to_numeric(df[metric], errors="coerce").dropna().values
        if vals.size == 0:
            ax.set_title(titles.get(metric, metric))
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_xticks([])
            continue

        bp = ax.boxplot(
            vals,
            vert=True,
            patch_artist=True,
            tick_labels=["Runs"],
            showfliers=True,
            flierprops={"markersize": 4},
        )
        for box in bp["boxes"]:
            box.set_facecolor("#8cc7ff")
            box.set_alpha(0.8)

        ax.set_title(titles.get(metric, metric))
        ax.set_ylabel(metric)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Cross-run Precision Distribution (Final Metrics)", fontsize=13)
    png = out_dir / "precision_boxplots.png"
    svg = out_dir / "precision_boxplots.svg"
    fig.savefig(png, dpi=180)
    fig.savefig(svg)
    plt.close(fig)


def _resolve_path(repo_root: Path, raw: str) -> Path:
    p = Path(str(raw))
    return p if p.is_absolute() else (repo_root / p)


def _ensure_cross_eval_inputs(repo_root: Path, manifest_df: pd.DataFrame) -> pd.DataFrame:
    required = ["run_id", "model_path", "test_csv"]
    missing = [c for c in required if c not in manifest_df.columns]
    if missing:
        raise RuntimeError(f"Manifest missing required columns for cross-eval: {missing}")

    df = manifest_df.copy()
    df["run_id"] = df["run_id"].astype(str)
    df["model_path_abs"] = [str(_resolve_path(repo_root, p)) for p in df["model_path"]]
    df["test_csv_abs"] = [str(_resolve_path(repo_root, p)) for p in df["test_csv"]]

    bad_model = [p for p in df["model_path_abs"] if not Path(p).is_file()]
    bad_test = [p for p in df["test_csv_abs"] if not Path(p).is_file()]
    if bad_model:
        raise FileNotFoundError(f"Some model files are missing; first missing: {bad_model[0]}")
    if bad_test:
        raise FileNotFoundError(f"Some test CSV files are missing; first missing: {bad_test[0]}")
    return df


def _run_model_batch_eval(
    repo_root: Path,
    batch_eval_script: Path,
    julia_cmd: str,
    julia_project: Optional[str],
    model_path: Path,
    test_list_csv: Path,
    output_csv: Path,
    slack_bus: Optional[int],
) -> pd.DataFrame:
    env = os.environ.copy()
    env["MODEL_PATH"] = str(model_path)
    env["TEST_LIST_CSV"] = str(test_list_csv)
    env["OUTPUT_CSV"] = str(output_csv)
    if slack_bus is not None:
        env["SLACK_BUS_ID"] = str(slack_bus)

    cmd: List[str] = shlex.split(julia_cmd)
    if julia_project:
        cmd.append(f"--project={julia_project}")
    cmd.append(str(batch_eval_script))

    proc = subprocess.run(cmd, cwd=str(repo_root), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        msg = stderr if stderr else stdout
        raise RuntimeError(f"Model batch eval failed for {model_path.name}: {msg[-2000:]}")

    if not output_csv.is_file():
        raise FileNotFoundError(f"Expected output CSV not found: {output_csv}")
    return pd.read_csv(output_csv)


def _plot_cross_eval_boxplots(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes = axes.flatten()

    for ax, metric in zip(axes, CROSS_METRIC_COLS):
        v_all = pd.to_numeric(df[metric], errors="coerce").dropna().values
        v_diag = pd.to_numeric(df.loc[df["same_run"], metric], errors="coerce").dropna().values
        v_off = pd.to_numeric(df.loc[~df["same_run"], metric], errors="coerce").dropna().values

        data = []
        labels = []
        if v_all.size:
            data.append(v_all)
            labels.append("all")
        if v_diag.size:
            data.append(v_diag)
            labels.append("diag")
        if v_off.size:
            data.append(v_off)
            labels.append("offdiag")

        if not data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(metric)
            ax.set_xticks([])
            continue

        bp = ax.boxplot(data, patch_artist=True, tick_labels=labels, showfliers=True, flierprops={"markersize": 3})
        for box in bp["boxes"]:
            box.set_facecolor("#8cc7ff")
            box.set_alpha(0.85)
        ax.set_title(metric)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Cross-Eval Robustness Distributions (Models x Test Sets)", fontsize=13)
    fig.savefig(out_dir / "cross_eval_boxplots.png", dpi=180)
    fig.savefig(out_dir / "cross_eval_boxplots.svg")
    plt.close(fig)


def _plot_cross_eval_heatmap(df: pd.DataFrame, metric: str, out_dir: Path):
    piv = df.pivot(index="model_run_id", columns="test_run_id", values=metric)
    if piv.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(piv.values, aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_yticks(range(len(piv.index)))
    ax.set_xticklabels(list(piv.columns), rotation=45, ha="right")
    ax.set_yticklabels(list(piv.index))
    ax.set_title(f"Cross-Eval Heatmap: {metric}")
    ax.set_xlabel("Test dataset run")
    ax.set_ylabel("Model run")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)
    fig.savefig(out_dir / f"cross_eval_heatmap_{metric}.png", dpi=180)
    fig.savefig(out_dir / f"cross_eval_heatmap_{metric}.svg")
    plt.close(fig)


def _plot_cross_eval_donerspiess(df: pd.DataFrame, out_dir: Path):
    metrics = ["mag_mae", "ang_mae", "mag_rmse", "ang_rmse"]
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    rng = np.random.default_rng(12345)
    for i, metric in enumerate(metrics, start=1):
        vals = pd.to_numeric(df[metric], errors="coerce").dropna().values
        if vals.size == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=vals.size)
        x = np.full(vals.size, i, dtype=float) + jitter
        ax.scatter(x, vals, s=18, alpha=0.45)
        ax.vlines(i, float(np.min(vals)), float(np.max(vals)), colors="black", linewidth=1.2, alpha=0.5)
        ax.hlines(float(np.median(vals)), i - 0.18, i + 0.18, colors="red", linewidth=2.0)

    ax.set_xticks(range(1, len(metrics) + 1))
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Error value")
    ax.set_title("Error Distribution (Donerspiess-style strip)")
    ax.grid(True, axis="y", alpha=0.3)

    fig.savefig(out_dir / "cross_eval_donerspiess.png", dpi=180)
    fig.savefig(out_dir / "cross_eval_donerspiess.svg")
    plt.close(fig)


def _cross_eval_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    groups = {
        "all": df,
        "diag": df[df["same_run"] == True],
        "offdiag": df[df["same_run"] == False],
    }

    for gname, gdf in groups.items():
        for metric in CROSS_METRIC_COLS:
            vals = pd.to_numeric(gdf[metric], errors="coerce").dropna()
            rows.append(
                {
                    "group": gname,
                    "metric": metric,
                    "count": int(vals.shape[0]),
                    "mean": (float(vals.mean()) if not vals.empty else math.nan),
                    "median": (float(vals.quantile(0.5)) if not vals.empty else math.nan),
                    "std": (float(vals.std(ddof=1)) if vals.shape[0] > 1 else (0.0 if vals.shape[0] == 1 else math.nan)),
                    "min": (float(vals.min()) if not vals.empty else math.nan),
                    "max": (float(vals.max()) if not vals.empty else math.nan),
                }
            )
    return pd.DataFrame(rows)


def _rank_models(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["model_run_id", "n_pairs", "mean_mag_mae", "mean_ang_mae", "mean_mag_rmse", "mean_ang_rmse", "robust_score"])

    offdiag = df[df["same_run"] == False].copy()
    src = offdiag if not offdiag.empty else df

    agg = (
        src.groupby("model_run_id", as_index=False)
        .agg(
            n_pairs=("model_run_id", "size"),
            mean_mag_mae=("mag_mae", "mean"),
            mean_ang_mae=("ang_mae", "mean"),
            mean_mag_rmse=("mag_rmse", "mean"),
            mean_ang_rmse=("ang_rmse", "mean"),
        )
    )
    agg["robust_score"] = pd.to_numeric(agg["mean_mag_mae"], errors="coerce") + pd.to_numeric(agg["mean_ang_mae"], errors="coerce")
    agg = agg.sort_values(["robust_score", "mean_mag_mae", "mean_ang_mae"], ascending=[True, True, True]).reset_index(drop=True)
    return agg


def _run_cross_eval(
    repo_root: Path,
    manifest_df: pd.DataFrame,
    out_dir: Path,
    batch_eval_script: Path,
    julia_cmd: str,
    julia_project: Optional[str],
    slack_bus: Optional[int],
    offdiag_only: bool,
    limit_pairs: int,
    fail_fast: bool,
):
    eval_df = _ensure_cross_eval_inputs(repo_root, manifest_df)
    runs = eval_df[["run_id", "model_path_abs", "test_csv_abs"]].to_dict("records")
    run_ids = [str(r["run_id"]) for r in runs]
    run_map = {str(r["run_id"]): r for r in runs}

    pair_rows: List[Dict[str, object]] = []
    err_rows: List[Dict[str, object]] = []
    pair_root = out_dir / "pairs"
    pair_root.mkdir(parents=True, exist_ok=True)

    requested_pairs: List[tuple[str, str]] = []
    for mid in run_ids:
        for tid in run_ids:
            if offdiag_only and mid == tid:
                continue
            requested_pairs.append((mid, tid))
    if limit_pairs > 0:
        requested_pairs = requested_pairs[:limit_pairs]

    tests_by_model: Dict[str, List[str]] = {}
    for mid, tid in requested_pairs:
        tests_by_model.setdefault(mid, []).append(tid)

    for model_run_id, test_run_ids in tests_by_model.items():
        m = run_map[model_run_id]
        model_dir = pair_root / f"model_{model_run_id}"
        model_dir.mkdir(parents=True, exist_ok=True)

        test_rows = []
        for tid in test_run_ids:
            t = run_map[tid]
            test_rows.append({"test_run_id": tid, "test_csv": str(t["test_csv_abs"])})
        test_list_csv = model_dir / "test_list.csv"
        pd.DataFrame(test_rows).to_csv(test_list_csv, index=False)

        output_csv = model_dir / "batch_eval_metrics.csv"
        try:
            model_res = _run_model_batch_eval(
                repo_root=repo_root,
                batch_eval_script=batch_eval_script,
                julia_cmd=julia_cmd,
                julia_project=julia_project,
                model_path=Path(m["model_path_abs"]),
                test_list_csv=test_list_csv,
                output_csv=output_csv,
                slack_bus=slack_bus,
            )
        except Exception as exc:
            for tid in test_run_ids:
                t = run_map[tid]
                err_rows.append(
                    {
                        "model_run_id": model_run_id,
                        "test_run_id": tid,
                        "model_path": str(m["model_path_abs"]),
                        "test_csv": str(t["test_csv_abs"]),
                        "error": str(exc),
                    }
                )
            if fail_fast:
                raise
            continue

        for _, rr in model_res.iterrows():
            tid = str(rr.get("test_run_id", ""))
            status = str(rr.get("status", "")).lower()
            t = run_map.get(tid)
            if t is None:
                continue

            if status != "ok":
                err_rows.append(
                    {
                        "model_run_id": model_run_id,
                        "test_run_id": tid,
                        "model_path": str(m["model_path_abs"]),
                        "test_csv": str(t["test_csv_abs"]),
                        "error": str(rr.get("error", "unknown error")),
                    }
                )
                if fail_fast:
                    raise RuntimeError(f"Pair failed: model={model_run_id}, test={tid}")
                continue

            row: Dict[str, object] = {
                "model_run_id": model_run_id,
                "test_run_id": tid,
                "same_run": bool(model_run_id == tid),
                "model_path": str(m["model_path_abs"]),
                "test_csv": str(t["test_csv_abs"]),
                "n_samples": rr.get("n_samples", np.nan),
                "n_targets": rr.get("n_targets", np.nan),
            }
            for mc in CROSS_METRIC_COLS:
                row[mc] = rr.get(mc, np.nan)
            pair_rows.append(row)

    cross_df = pd.DataFrame(pair_rows)
    cross_csv = out_dir / "cross_eval_metrics.csv"
    cross_df.to_csv(cross_csv, index=False)

    err_csv = out_dir / "cross_eval_errors.csv"
    pd.DataFrame(err_rows).to_csv(err_csv, index=False)

    summary_df = _cross_eval_summary(cross_df) if not cross_df.empty else pd.DataFrame(columns=["group", "metric", "count", "mean", "median", "std", "min", "max"])
    summary_csv = out_dir / "cross_eval_summary_stats.csv"
    summary_df.to_csv(summary_csv, index=False)

    if not cross_df.empty:
        _plot_cross_eval_boxplots(cross_df, out_dir)
        _plot_cross_eval_heatmap(cross_df, "mag_mae", out_dir)
        _plot_cross_eval_heatmap(cross_df, "ang_mae", out_dir)
        _plot_cross_eval_donerspiess(cross_df, out_dir)

    ranking_df = _rank_models(cross_df)
    ranking_csv = out_dir / "model_robustness_ranking.csv"
    ranking_df.to_csv(ranking_csv, index=False)

    best_worst_txt = out_dir / "best_worst_model_runs.txt"
    with open(best_worst_txt, "w", encoding="utf-8") as f:
        if ranking_df.empty:
            f.write("No successful cross-eval pairs; cannot rank models.\n")
        else:
            best = ranking_df.iloc[0]
            worst = ranking_df.iloc[-1]
            f.write(f"Best model run: {best['model_run_id']}  robust_score={best['robust_score']:.8f}\n")
            f.write(f"Worst model run: {worst['model_run_id']}  robust_score={worst['robust_score']:.8f}\n")

    print("Cross-eval saved:")
    print(f"  {cross_csv}")
    print(f"  {err_csv}")
    print(f"  {summary_csv}")
    if not cross_df.empty:
        print(f"  {out_dir / 'cross_eval_boxplots.png'}")
        print(f"  {out_dir / 'cross_eval_heatmap_mag_mae.png'}")
        print(f"  {out_dir / 'cross_eval_heatmap_ang_mae.png'}")
        print(f"  {out_dir / 'cross_eval_donerspiess.png'}")
    print(f"  {ranking_csv}")
    print(f"  {best_worst_txt}")
    print(f"  Cross-eval pairs: ok={len(cross_df)} failed={len(err_rows)}")


def main():
    parser = argparse.ArgumentParser(description="Summarize collect+train batch runs")
    parser.add_argument("--batch-dir", default="", help="Batch directory containing manifest.csv")
    parser.add_argument("--manifest", default="", help="Path to manifest.csv")
    parser.add_argument("--output-dir", default="", help="Output directory (default: <batch_dir>/summary_boxplots)")
    parser.add_argument("--cross-eval", action="store_true", help="Run post-hoc cross-evaluation: each model against each test CSV")
    parser.add_argument("--cross-eval-dir", default="", help="Output directory for cross-eval artifacts (default: <output-dir>/cross_eval)")
    parser.add_argument("--eval-script", default="src/pisr/cross_eval_model_once.jl", help="Batch evaluator Julia script path")
    parser.add_argument("--julia-cmd", default="julia", help="Julia command used for evaluator calls")
    parser.add_argument("--julia-project", default="", help="Julia project path (default: repo root)")
    parser.add_argument("--slack-bus", default="", help="Optional slack bus override for evaluator")
    parser.add_argument("--offdiag-only", action="store_true", help="Only evaluate model_i on test_j for i != j")
    parser.add_argument("--limit-pairs", type=int, default=0, help="Limit number of evaluated pairs (0 = all)")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first cross-eval pair failure")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = _resolve_manifest(
        repo_root,
        batch_dir=(args.batch_dir.strip() or None),
        manifest=(args.manifest.strip() or None),
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    batch_dir = manifest_path.parent
    out_dir = Path(args.output_dir) if args.output_dir else (batch_dir / "summary_boxplots")
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(manifest_path)
    if manifest_df.empty:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")

    rows: List[Dict[str, object]] = []
    for _, rec in manifest_df.iterrows():
        run_id = str(rec.get("run_id", ""))
        run_dir_raw = rec.get("run_dir", "")
        run_dir = Path(str(run_dir_raw)) if str(run_dir_raw) else (batch_dir / "runs" / run_id)
        if not run_dir.is_absolute():
            run_dir = repo_root / run_dir

        final_metrics = _load_final_metrics(run_dir)
        merged: Dict[str, object] = rec.to_dict()
        merged.update(final_metrics)
        merged["metrics_source"] = str(run_dir / "training" / "metrics_progress.csv")
        rows.append(merged)

    final_df = pd.DataFrame(rows)

    # Ensure numeric dtype for summary/plots
    for col in ["iteration", *METRIC_COLS]:
        final_df[col] = pd.to_numeric(final_df[col], errors="coerce")

    all_runs_path = out_dir / "all_runs_final_metrics.csv"
    final_df.to_csv(all_runs_path, index=False)

    stats_df = _summarize(final_df)
    stats_path = out_dir / "summary_stats.csv"
    stats_df.to_csv(stats_path, index=False)

    _make_boxplots(final_df, out_dir)

    print("Saved:")
    print(f"  Manifest used: {manifest_path}")
    print(f"  {all_runs_path}")
    print(f"  {stats_path}")
    print(f"  {out_dir / 'precision_boxplots.png'}")
    print(f"  {out_dir / 'precision_boxplots.svg'}")

    if args.cross_eval:
        evaluator_path = _resolve_path(repo_root, args.eval_script)
        if not evaluator_path.exists():
            raise FileNotFoundError(f"Evaluator script not found: {evaluator_path}")

        cross_eval_dir = Path(args.cross_eval_dir) if args.cross_eval_dir else (out_dir / "cross_eval")
        if not cross_eval_dir.is_absolute():
            cross_eval_dir = repo_root / cross_eval_dir
        cross_eval_dir.mkdir(parents=True, exist_ok=True)

        julia_project = args.julia_project.strip() or str(repo_root)
        slack_bus = int(args.slack_bus) if args.slack_bus.strip() else None

        _run_cross_eval(
            repo_root=repo_root,
            manifest_df=manifest_df,
            out_dir=cross_eval_dir,
            batch_eval_script=evaluator_path,
            julia_cmd=args.julia_cmd,
            julia_project=julia_project,
            slack_bus=slack_bus,
            offdiag_only=bool(args.offdiag_only),
            limit_pairs=max(0, int(args.limit_pairs)),
            fail_fast=bool(args.fail_fast),
        )


if __name__ == "__main__":
    main()
