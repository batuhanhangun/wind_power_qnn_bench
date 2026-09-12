"""Aggregation + mandatory reproduction gate for the regularized ANN rerun.

First runs the reproduction gate: every (seed, train_size) rerun's test_r2 and
train_r2 is compared against reference/ann_reg_classical_per_seed.csv at
absolute tolerance 1e-6. Only if ALL 40 rows match are the aggregated outputs
written to `aggregated/run_<RUN_ID>/`:

1. ann_reg_per_seed.csv           — one row per (seed, train_size), full metrics
2. ann_reg_summary.csv            — per train_size mean/std, run-to-run sigma,
                                    min seed test R2
3. ann_reg_classical_per_seed.csv — reduced schema (model, seed, train_size,
                                    test_r2, train_r2, generalization_gap)
4. reproduction_report.txt        — gate result and per-row comparison summary

If any row mismatches at 1e-6, the mismatch table is printed, the match count
at the looser 1e-3 tolerance is reported, `REPRODUCTION: FAILED` is printed,
and the script exits WITHOUT writing any aggregated outputs. A mismatch means
the environment differs from the original run and must be resolved by

CSV and text only — no plotting.

Aggregation is called automatically by the corresponding scripts.run_* entry point.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qnnbench import config
from qnnbench._io import print_flush


# --- Data collection ---

def collect_results() -> pd.DataFrame:
    """Collect all per-(seed, train_size) metrics JSON files.

    Returns:
        DataFrame with one row per (seed, train_size).
    """
    rows = []
    results_dir = Path(config.RESULTS_DIR)

    if not results_dir.exists():
        print_flush(f"Results directory not found: {results_dir}")
        return pd.DataFrame()

    for seed_dir in sorted(results_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        for metrics_file in sorted(seed_dir.glob("exp_*_metrics.json")):
            try:
                with open(metrics_file) as f:
                    entry = json.load(f)
                rows.append(entry)
            except Exception as e:
                print_flush(f"Warning: could not parse {metrics_file}: {e}")

    return pd.DataFrame(rows)


def check_coverage(df: pd.DataFrame) -> bool:
    """Report which (seed, train_size) combinations are present."""
    print_flush("\nData coverage:")
    print_flush("-" * 50)
    complete = True
    for size in config.TRAIN_SIZES:
        seeds_present = sorted(df[df["train_size"] == size]["seed"].tolist())
        missing = sorted(set(config.SEEDS) - set(seeds_present))
        status = "OK" if not missing else f"MISSING {missing}"
        if missing:
            complete = False
        print_flush(f"  size={size:>4}: {len(seeds_present)}/{len(config.SEEDS)} seeds  [{status}]")
    print_flush("-" * 50)
    print_flush("All combinations present." if complete
                else "WARNING: some combinations are missing.")
    return complete


# --- Reproduction gate ---

def run_reproduction_gate(df: pd.DataFrame):
    """Compare the rerun against the reference CSV row-by-row.

    Args:
        df: Collected rerun results (one row per (seed, train_size)).

    Returns:
        Tuple (passed: bool, report_lines: list[str]).
    """
    atol = config.REPRODUCTION_ATOL
    atol_loose = config.REPRODUCTION_ATOL_LOOSE

    ref_path = Path(config.REFERENCE_CSV)
    if not ref_path.exists():
        print_flush(f"ERROR: reference CSV not found: {ref_path}")
        sys.exit(1)
    ref = pd.read_csv(ref_path)

    lines = []
    lines.append("=" * 70)
    lines.append("REPRODUCTION GATE — rerun vs reference")
    lines.append("=" * 70)
    lines.append(f"Reference: {ref_path}")
    lines.append(f"Tolerance: abs diff <= {atol:g} on test_r2 and train_r2")
    lines.append("")

    mismatches = []
    n_rows = 0
    n_match_loose = 0

    rerun_idx = {(int(r["seed"]), int(r["train_size"])): r
                 for _, r in df.iterrows()}

    for _, ref_row in ref.iterrows():
        seed = int(ref_row["seed"])
        size = int(ref_row["train_size"])
        n_rows += 1
        run_row = rerun_idx.get((seed, size))
        if run_row is None:
            mismatches.append((seed, size, "MISSING", np.nan, np.nan, np.nan))
            continue
        row_loose_ok = True
        row_ok = True
        for metric in ("test_r2", "train_r2"):
            ref_v = float(ref_row[metric])
            run_v = float(run_row[metric])
            diff = abs(run_v - ref_v)
            if diff > atol:
                row_ok = False
                mismatches.append((seed, size, metric, ref_v, run_v, diff))
            if diff > atol_loose:
                row_loose_ok = False
        if row_loose_ok:
            n_match_loose += 1
        if row_ok:
            lines.append(f"  seed={seed} size={size}: MATCH "
                         f"(test_r2 diff={abs(float(run_row['test_r2']) - float(ref_row['test_r2'])):.2e}, "
                         f"train_r2 diff={abs(float(run_row['train_r2']) - float(ref_row['train_r2'])):.2e})")

    passed = not mismatches

    if passed:
        lines.append("")
        lines.append(f"All {n_rows}/{n_rows} rows match within {atol:g}.")
        lines.append("REPRODUCTION: EXACT")
    else:
        lines.append("")
        lines.append(f"{len(mismatches)} metric mismatch(es) exceed {atol:g}:")
        lines.append("")
        header = (f"{'seed':>4} {'size':>5} {'metric':>9} "
                  f"{'reference':>20} {'rerun':>20} {'difference':>14}")
        lines.append(header)
        lines.append("-" * len(header))
        for seed, size, metric, ref_v, run_v, diff in mismatches:
            if metric == "MISSING":
                lines.append(f"{seed:>4} {size:>5} {'MISSING':>9} "
                             f"{'-':>20} {'-':>20} {'-':>14}")
            else:
                lines.append(f"{seed:>4} {size:>5} {metric:>9} "
                             f"{ref_v:>20.12f} {run_v:>20.12f} {diff:>14.3e}")
        lines.append("")
        lines.append(f"Rows matching at looser tolerance {atol_loose:g}: "
                     f"{n_match_loose}/{n_rows}")
        lines.append("REPRODUCTION: FAILED")
        lines.append("")
        lines.append("The environment differs from the original run. Do NOT")
        lines.append("overwrite or 'correct' anything — resolve by re-running")

    for line in lines:
        print_flush(line)

    return passed, lines


# --- Summary ---

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per train_size mean/std across seeds + collapse diagnostics.

    Returns:
        DataFrame with one row per train_size.
    """
    metric_cols = [
        "test_r2", "train_r2", "generalization_gap",
        "test_mse", "test_rmse", "test_mae",
    ]
    rows = []
    for size in config.TRAIN_SIZES:
        sub = df[df["train_size"] == size]
        if sub.empty:
            continue
        row = {"train_size": size, "n_seeds": len(sub)}
        for col in metric_cols:
            row[f"{col}_mean"] = float(sub[col].mean())
            row[f"{col}_std"] = float(sub[col].std(ddof=1))
        # Run-to-run sigma of test R2 across seeds (== test_r2_std; surfaced
        # explicitly for direct comparison with the constrained ANN's sigma).
        row["test_r2_run_to_run_sigma"] = float(sub["test_r2"].std(ddof=1))
        row["test_r2_min"] = float(sub["test_r2"].min())
        row["test_r2_max"] = float(sub["test_r2"].max())
        row["n_params_mean"] = float(sub["n_params"].mean())
        row["n_params_min"] = int(sub["n_params"].min())
        row["n_params_max"] = int(sub["n_params"].max())
        row["n_collapsed_seeds"] = int(
            (sub["test_r2"] < config.SEED_COLLAPSE_THRESHOLD).sum()
        )
        rows.append(row)
    return pd.DataFrame(rows)


# --- Main ---

