"""Aggregate Stage 1 + Stage 2 results into CSVs for local plotting.

Reads every results/run_<RUN_ID>/seed_<seed>/exp_<size>_metrics.json (Stage 1)
and exp_<size>_noise_metrics.json (Stage 2) and emits to
aggregated/run_<RUN_ID>/ (CSV + text only — NO plotting):

  qnn_performance_summary.csv  per train_size: mean/std (ddof=1) over seeds of
                               test R2, MSE, RMSE, MAE, and generalization gap.
  qnn_per_seed.csv             one row per (seed, train_size): test/train R2,
                               gap, MSE, RMSE, MAE.
  noise_summary.csv            per noise level: mean/std over the 40 (seed,size)
                               runs of R2, MSE, RMSE, MAE, and % change in R2
                               vs the p=0 baseline.
  noise_per_run.csv            one row per (seed, train_size, noise_level).
  consistency_report.txt       N=800 mean test R2 vs the expected ~0.903, the
                               p=0/handoff reconciliation, and the CNOT count.

Aggregation is called automatically by the corresponding scripts.run_* entry point.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qnnbench import config

EXPECTED_N800_TEST_R2 = 0.903   # selection-study anchor (consistency, not a target)
P0_RECON_TOL = 1e-6
WEIGHT_RECON_TOL = 1e-10


# --- Collection ---

def collect_stage1():
    """Return a list of per-(seed, train_size) Stage 1 metric dicts."""
    rows = []
    results_dir = Path(config.RESULTS_DIR)
    if not results_dir.exists():
        return rows

    for seed_dir in sorted(results_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        seed = int(seed_dir.name.split("_")[1])
        for mf in sorted(seed_dir.glob("exp_*_metrics.json")):
            if mf.name.endswith("_noise_metrics.json"):
                continue
            try:
                with open(mf) as f:
                    d = json.load(f)
                rows.append({
                    "seed": seed,
                    "train_size": int(d["train_size"]),
                    "test_r2": d["test_r2"],
                    "train_r2": d["train_r2"],
                    "generalization_gap": d.get(
                        "generalization_gap", d["train_r2"] - d["test_r2"]
                    ),
                    "test_mse": d["test_mse"],
                    "test_rmse": d["test_rmse"],
                    "test_mae": d["test_mae"],
                    "cv_mean_r2": d.get("cv_mean_r2"),
                    "cv_std_r2": d.get("cv_std_r2"),
                })
            except Exception as e:
                print(f"Warning: could not parse {mf}: {e}")
    return rows


def collect_stage2():
    """Return (per_run_rows, handoff_rows) from Stage 2 noise JSONs.

    per_run_rows : one dict per (seed, train_size, noise_level).
    handoff_rows : one dict per (seed, train_size) with handoff diagnostics.
    """
    per_run, handoff = [], []
    results_dir = Path(config.RESULTS_DIR)
    if not results_dir.exists():
        return per_run, handoff

    for seed_dir in sorted(results_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        seed = int(seed_dir.name.split("_")[1])
        for mf in sorted(seed_dir.glob("exp_*_noise_metrics.json")):
            try:
                train_size = int(mf.stem.split("_")[1])
                with open(mf) as f:
                    payload = json.load(f)
                # Stage 2 writes {"levels": [...], "handoff": {...}}.
                levels = payload["levels"] if isinstance(payload, dict) else payload
                hand = payload.get("handoff", {}) if isinstance(payload, dict) else {}

                for entry in levels:
                    if entry.get("error") or entry.get("test_r2") is None:
                        print(f"  Skipping failed level: seed={seed}, "
                              f"size={train_size}, p={entry.get('noise_p')}")
                        continue
                    per_run.append({
                        "seed": seed,
                        "train_size": train_size,
                        "noise_p": entry["noise_p"],
                        "test_r2": entry["test_r2"],
                        "test_mse": entry["test_mse"],
                        "test_rmse": entry["test_rmse"],
                        "test_mae": entry["test_mae"],
                    })
                handoff.append({
                    "seed": seed,
                    "train_size": train_size,
                    "weight_handoff_max_abs_diff":
                        hand.get("weight_handoff_max_abs_diff"),
                    "p0_vs_stage1_abs_diff": hand.get("p0_vs_stage1_abs_diff"),
                })
            except Exception as e:
                print(f"Warning: could not parse {mf}: {e}")
    return per_run, handoff


# --- Aggregation ---

def build_performance_summary(stage1_rows):
    """Per train_size: mean/std (ddof=1) over seeds of test metrics + gap."""
    df = pd.DataFrame(stage1_rows)
    metrics = ["test_r2", "test_mse", "test_rmse", "test_mae",
               "generalization_gap"]
    out = []
    for train_size, g in df.groupby("train_size"):
        row = {"train_size": int(train_size), "n_seeds": len(g)}
        for m in metrics:
            row[f"mean_{m}"] = float(g[m].mean())
            row[f"std_{m}"] = float(g[m].std(ddof=1))
        out.append(row)
    return pd.DataFrame(out).sort_values("train_size").reset_index(drop=True)


def build_noise_summary(noise_rows):
    """Per noise level: mean/std over all (seed,size) runs + %dR2 vs p=0."""
    df = pd.DataFrame(noise_rows)
    metrics = ["test_r2", "test_mse", "test_rmse", "test_mae"]
    out = []
    for p, g in df.groupby("noise_p"):
        row = {"noise_p": float(p), "n_runs": len(g)}
        for m in metrics:
            row[f"mean_{m}"] = float(g[m].mean())
            row[f"std_{m}"] = float(g[m].std(ddof=1))
        out.append(row)
    summary = pd.DataFrame(out).sort_values("noise_p").reset_index(drop=True)

    # Percent change in mean R2 versus the p=0 baseline.
    base = summary.loc[summary["noise_p"] == 0.0, "mean_test_r2"]
    baseline_r2 = float(base.values[0]) if len(base) else float("nan")
    summary["pct_change_r2_vs_p0"] = (
        (summary["mean_test_r2"] - baseline_r2) / baseline_r2 * 100.0
    )
    return summary, baseline_r2


# --- Consistency report ---

def reconcile_p0(stage1_rows, noise_rows):
    """Compare Stage 2 p=0 test R2 against Stage 1 test R2 per (seed, size)."""
    s1 = {(r["seed"], r["train_size"]): r["test_r2"] for r in stage1_rows}
    diffs = []
    for r in noise_rows:
        if r["noise_p"] != 0.0:
            continue
        key = (r["seed"], r["train_size"])
        if key in s1:
            diffs.append(abs(r["test_r2"] - s1[key]))
    return diffs


def write_consistency_report(path, perf_df, stage1_rows, noise_rows,
                             handoff_rows, baseline_r2):
    lines = []
    lines.append("=" * 64)
    lines.append("reverse_linear QNN rerun - consistency report")
    lines.append(f"Run ID: {config.RUN_ID}")
    lines.append("=" * 64)
    lines.append("")

    # N=800 anchor
    r800 = perf_df.loc[perf_df["train_size"] == 800]
    lines.append("[1] N=800 noise-free test R2 (mean over seeds)")
    if len(r800):
        mean800 = float(r800["mean_test_r2"].values[0])
        std800 = float(r800["std_test_r2"].values[0])
        n800 = int(r800["n_seeds"].values[0])
        lines.append(f"    measured : {mean800:.4f} +/- {std800:.4f} "
                     f"(n={n800} seeds)")
        lines.append(f"    expected : ~{EXPECTED_N800_TEST_R2:.3f} "
                     f"(selection study)")
        lines.append(f"    |d|      : {abs(mean800 - EXPECTED_N800_TEST_R2):.4f}")
    else:
        lines.append("    MISSING — no N=800 Stage 1 results found.")
    # CV anchor, if present
    cv800 = [r["cv_mean_r2"] for r in stage1_rows
             if r["train_size"] == 800 and r.get("cv_mean_r2") is not None]
    if cv800:
        lines.append(f"    CV val   : {np.mean(cv800):.4f} "
                     f"(expected ~0.902)")
    lines.append("")

    # p=0 / handoff reconciliation
    lines.append("[2] Stage 1 -> Stage 2 handoff reconciliation")
    p0_diffs = reconcile_p0(stage1_rows, noise_rows)
    if p0_diffs:
        max_p0 = max(p0_diffs)
        ok_p0 = max_p0 < P0_RECON_TOL
        lines.append(f"    p=0 vs Stage 1 test R2 : max|d|={max_p0:.3e} "
                     f"over {len(p0_diffs)} runs  "
                     f"-> {'PASS' if ok_p0 else 'FAIL'} (< {P0_RECON_TOL:.0e})")
    else:
        lines.append("    p=0 vs Stage 1: no paired runs found.")

    # Handoff diagnostics recorded at Stage 2 runtime
    wdiffs = [h["weight_handoff_max_abs_diff"] for h in handoff_rows
              if h.get("weight_handoff_max_abs_diff") is not None]
    pdiffs = [h["p0_vs_stage1_abs_diff"] for h in handoff_rows
              if h.get("p0_vs_stage1_abs_diff") is not None]
    if wdiffs:
        lines.append(f"    weight handoff (loaded vs retrained): "
                     f"max|d|={max(wdiffs):.3e} over {len(wdiffs)} runs  "
                     f"-> {'PASS' if max(wdiffs) < WEIGHT_RECON_TOL else 'FAIL'} "
                     f"(< {WEIGHT_RECON_TOL:.0e})")
    if pdiffs:
        lines.append(f"    Stage 2 runtime p=0 check: "
                     f"max|d|={max(pdiffs):.3e}")
    lines.append("    (Channel-at-zero identity is gated separately by "
                 "tests/test_reverse_linear.py.)")
    lines.append("")

    # CNOT record
    lines.append("[3] Architecture record")
    lines.append(f"    entanglement : {config.ENTANGLEMENT}")
    lines.append(f"    CNOT count   : {config.CNOT_COUNT}  "
                 f"(vs circular's 12)")
    lines.append(f"    parameters   : {config.QNN_CONFIG['num_parameters']}")
    lines.append("")
    lines.append(f"    noise-free baseline mean R2 (pooled 40 runs): "
                 f"{baseline_r2:.4f}")
    lines.append("=" * 64)

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# --- Main ---

