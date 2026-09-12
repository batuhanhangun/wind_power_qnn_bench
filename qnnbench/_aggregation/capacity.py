"""Aggregate the capacity-ablation results into CSVs + a Q9 finding report.

Reads every results/run_<RUN_ID>/seed_<seed>/exp_reps<reps>_metrics.json and
emits to aggregated/run_<RUN_ID>/ (CSV + text only — NO plotting):

  capacity_per_seed.csv  one row per (reps, seed): reps, num_params, num_cnots,
                         seed, train_size, test_r2, train_r2,
                         generalization_gap, test_mse, test_rmse, test_mae.
  capacity_summary.csv   one row per reps, aggregated over the 10 seeds: mean and
                         std (ddof=1) of test_r2, train_r2, generalization_gap,
                         plus the across-seed run-to-run sigma of test_r2.
  capacity_report.txt    the gap-vs-capacity curve (mean gap and mean test R2 at
                         each reps), the explicit Q9 finding (does the mean gap
                         stay near zero as params grow 8->24, or widen past some
                         capacity — with the threshold if one appears), and the
                         reps=3 consistency check vs the deployed N=800 ~0.903.

Aggregation is called automatically by the corresponding scripts.run_* entry point.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qnnbench import config

# A mean generalization gap below this (in R2 units) is treated as "near zero"
# for the Q9 verdict. 0.05 R2 is a generous band: a model that was merely too
# small to overfit would, once given capacity, blow well past this.
GAP_NEAR_ZERO_THRESH = 0.05

PER_SEED_COLUMNS = [
    "reps", "num_params", "num_cnots", "seed", "train_size",
    "test_r2", "train_r2", "generalization_gap",
    "test_mse", "test_rmse", "test_mae",
]


# --- Collection ---

def collect_rows():
    """Return a list of per-(reps, seed) metric dicts from the results tree."""
    rows = []
    results_dir = Path(config.RESULTS_DIR)
    if not results_dir.exists():
        return rows

    for seed_dir in sorted(results_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        for mf in sorted(seed_dir.glob("exp_reps*_metrics.json")):
            try:
                with open(mf) as f:
                    d = json.load(f)
                rows.append({
                    "reps": int(d["reps"]),
                    "num_params": int(d["num_params"]),
                    "num_cnots": int(d["num_cnots"]),
                    "seed": int(d["seed"]),
                    "train_size": int(d["train_size"]),
                    "test_r2": float(d["test_r2"]),
                    "train_r2": float(d["train_r2"]),
                    "generalization_gap": float(
                        d.get("generalization_gap",
                              d["train_r2"] - d["test_r2"])
                    ),
                    "test_mse": float(d["test_mse"]),
                    "test_rmse": float(d["test_rmse"]),
                    "test_mae": float(d["test_mae"]),
                })
            except Exception as e:
                print(f"Warning: could not parse {mf}: {e}")
    return rows


# --- Aggregation ---

def build_summary(df):
    """Per reps: mean/std (ddof=1) of test_r2, train_r2, gap + run-to-run sigma."""
    out = []
    for reps, g in df.groupby("reps"):
        row = {
            "reps": int(reps),
            "num_params": int(g["num_params"].iloc[0]),
            "num_cnots": int(g["num_cnots"].iloc[0]),
            "n_seeds": int(len(g)),
        }
        for m in ("test_r2", "train_r2", "generalization_gap"):
            row[f"mean_{m}"] = float(g[m].mean())
            row[f"std_{m}"] = float(g[m].std(ddof=1))
        # Across-seed run-to-run sigma of test_r2 (sample std of the 10 seeds'
        # test R2; same definition as std_test_r2, surfaced under the stability
        # name used throughout the study).
        row["run_to_run_sigma_test_r2"] = float(g["test_r2"].std(ddof=1))
        out.append(row)
    return pd.DataFrame(out).sort_values("reps").reset_index(drop=True)


# --- Q9 finding ---

def derive_q9_finding(summary):
    """Return (verdict_lines, threshold_reps_or_None) from the gap curve."""
    lines = []
    s = summary.sort_values("num_params").reset_index(drop=True)

    gaps = list(zip(s["reps"], s["num_params"], s["mean_generalization_gap"]))
    # First reps (by increasing capacity) whose mean gap exceeds the band.
    onset = next((r for (r, p, gp) in gaps if gp > GAP_NEAR_ZERO_THRESH), None)

    gap_lo = float(s["mean_generalization_gap"].iloc[0])    # 8 params
    gap_hi = float(s["mean_generalization_gap"].iloc[-1])   # 24 params
    max_gap = float(s["mean_generalization_gap"].max())
    max_gap_reps = int(s.loc[s["mean_generalization_gap"].idxmax(), "reps"])

    lines.append(f"Q9 verdict band: a mean generalization gap <= "
                 f"{GAP_NEAR_ZERO_THRESH:.2f} R2 is treated as 'near zero'.")
    lines.append(f"Mean gap at 8 params (reps=1):  {gap_lo:+.4f}")
    lines.append(f"Mean gap at 24 params (reps=5): {gap_hi:+.4f}")
    lines.append(f"Largest mean gap on the grid:  {max_gap:+.4f} "
                 f"(reps={max_gap_reps})")
    lines.append("")

    if onset is None:
        lines.append(
            "FINDING: The mean generalization gap remains near zero as the "
            "parameter count grows from 8 to 24 (every rep value stays within "
            f"the {GAP_NEAR_ZERO_THRESH:.2f} band). The small gap is therefore "
            "an ARCHITECTURAL PROPERTY of the reverse_linear QNN, not an "
            "artifact of a model too small to overfit: tripling the parameter "
            "count does not induce overfitting. The deployed reps=3 model is "
            "representative of this stable regime, not a uniquely lucky point."
        )
    else:
        onset_params = int(s.loc[s["reps"] == onset, "num_params"].iloc[0])
        deployed_below = config.DEPLOYED_REPS < onset
        lines.append(
            f"FINDING: The mean generalization gap WIDENS past the "
            f"{GAP_NEAR_ZERO_THRESH:.2f} band starting at reps={onset} "
            f"({onset_params} params) — this locates a capacity threshold. "
            f"The deployed reps=3 (16 params) model sits "
            f"{'BELOW' if deployed_below else 'AT/ABOVE'} that threshold, so "
            f"its near-zero gap reflects operating "
            f"{'under' if deployed_below else 'at'} the onset of overfitting "
            f"capacity rather than a structural inability to overfit."
        )
    return lines, onset


# --- Report ---

def write_report(path, summary, per_seed_df):
    lines = []
    lines.append("=" * 64)
    lines.append("QNN capacity ablation (reps sweep, reverse_linear) — Q9 report")
    lines.append(f"Run ID: {config.RUN_ID}")
    lines.append(f"Regime: N={config.TRAIN_SIZE}, seeds {config.SEEDS[0]}.."
                 f"{config.SEEDS[-1]}, noise-free")
    lines.append("=" * 64)
    lines.append("")

    # Gap-vs-capacity curve.
    lines.append("[1] Gap-vs-capacity curve (mean over 10 seeds)")
    lines.append(f"    {'reps':>4} {'params':>7} {'CNOTs':>6} "
                 f"{'mean_test_r2':>13} {'mean_gap':>10} {'sigma_test_r2':>14}")
    for _, r in summary.iterrows():
        lines.append(
            f"    {int(r['reps']):>4} {int(r['num_params']):>7} "
            f"{int(r['num_cnots']):>6} {r['mean_test_r2']:>13.4f} "
            f"{r['mean_generalization_gap']:>10.4f} "
            f"{r['run_to_run_sigma_test_r2']:>14.4f}"
        )
    lines.append("")

    # Q9 finding.
    lines.append("[2] Q9 finding — capacity vs the generalization gap")
    finding_lines, onset = derive_q9_finding(summary)
    for fl in finding_lines:
        lines.append(f"    {fl}" if fl else "")
    lines.append("")

    # reps=3 consistency vs deployed ~0.903.
    lines.append("[3] Consistency check — reps=3 reproduces the deployed N=800 R2")
    dep = summary.loc[summary["reps"] == config.DEPLOYED_REPS]
    if len(dep):
        measured = float(dep["mean_test_r2"].iloc[0])
        std = float(dep["std_test_r2"].iloc[0])
        n = int(dep["n_seeds"].iloc[0])
        lines.append(f"    measured (reps=3): {measured:.4f} +/- {std:.4f} "
                     f"(n={n} seeds)")
        lines.append(f"    expected (deployed): ~{config.EXPECTED_N800_TEST_R2:.3f}")
        lines.append(f"    |measured - expected|: "
                     f"{abs(measured - config.EXPECTED_N800_TEST_R2):.4f}")
    else:
        lines.append("    MISSING — no reps=3 results found.")
    lines.append("")
    lines.append("=" * 64)

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# --- Main ---

