"""Aggregate structure-selection results and apply the stability-aware rule.

Reads every per-(config, seed) result JSON written by ``src/run_selection.py`` and
produces three files under ``aggregated/run_<RUN_ID>/``:

  1. selection_per_seed.csv   - one row per (config, seed)
  2. selection_summary.csv    - one row per config, aggregated over seeds
  3. selection_decision.txt   - the stability-aware decision with full reasoning

The decision uses cross-validation metrics ONLY. Test-set columns are carried
through for transparency and reproduction checking but never inform the choice.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `import config` and `from src...` from the project root.
from qnnbench import config
from qnnbench._io import print_flush
from qnnbench.config import CONFIGS  # dependency-free: no PennyLane needed here


def collect_results():
    """Load every per-(config, seed) result JSON under config.RESULTS_DIR."""
    rows = []
    base = Path(config.RESULTS_DIR)
    for cfg in CONFIGS:
        for seed in config.SEEDS:
            path = base / cfg / f"seed_{seed}" / "selection_metrics.json"
            if path.exists():
                with open(path) as f:
                    rows.append(json.load(f))
            else:
                print_flush(f"WARNING: missing result {path}")
    return rows


def _pooled_sd(per_config_sds):
    """Pooled standard deviation = RMS of the per-config across-seed SDs."""
    vals = np.asarray(per_config_sds, dtype=float)
    return float(np.sqrt(np.mean(vals ** 2)))


def build_summary(df):
    """One row per config: means, stds (ddof=1), and run-to-run sigma."""
    rows = []
    for cfg in CONFIGS:
        g = df[df["config"] == cfg]
        rows.append({
            "config": cfg,
            "feature_map": g["feature_map"].iloc[0],
            "entanglement": cfg,
            "num_params": int(g["num_params"].iloc[0]),
            "n_seeds": int(len(g)),
            "cv_val_r2_mean": float(g["cv_val_r2"].mean()),
            "cv_val_r2_std": float(g["cv_val_r2"].std(ddof=1)),
            "cv_gap_mean": float(g["cv_gap"].mean()),
            "cv_gap_std": float(g["cv_gap"].std(ddof=1)),
            "test_r2_mean": float(g["test_r2"].mean()),
            "test_r2_std": float(g["test_r2"].std(ddof=1)),
            "test_gap_mean": float(g["test_gap"].mean()),
            "test_gap_std": float(g["test_gap"].std(ddof=1)),
            # Across-seed std of cv_val_r2 = the run-to-run stability measure.
            "cv_val_r2_run_to_run_sigma": float(g["cv_val_r2"].std(ddof=1)),
        })
    return pd.DataFrame(rows)


def decide(summary):
    """Apply the stability-aware selection rule. Returns (decision, lines).

    Rule: among the three configs identify the accuracy leader (highest mean
    cv_val_r2), the smallest mean cv_gap, and the smallest run-to-run sigma.
    Select the highest-accuracy config whose mean cv_gap is within one pooled SD
    of the smallest gap AND whose run-to-run sigma is within one pooled SD of the
    smallest sigma. If the accuracy leader qualifies it wins; otherwise fall to
    the next-highest-accuracy config that does. A marginal accuracy advantage
    must not override a materially worse generalization gap or run-to-run variance.
    """
    if (summary["n_seeds"] < 2).any():
        return None, (
            "Selection decision unavailable for a single-seed execution check.\n"
            "Across-seed sample standard deviations require at least two seeds.\n"
            "Per-candidate CV and test metrics are saved; run all ten seeds "
            "to reproduce the manuscript selection study."
        )
    s = summary.set_index("config")
    acc = s["cv_val_r2_mean"]
    gap = s["cv_gap_mean"]
    sigma = s["cv_val_r2_run_to_run_sigma"]

    accuracy_leader = acc.idxmax()
    smallest_gap_cfg = gap.idxmin()
    smallest_sigma_cfg = sigma.idxmin()

    min_gap = float(gap.min())
    min_sigma = float(sigma.min())

    # Pooled yardsticks: RMS of the per-config across-seed SDs.
    gap_pooled_sd = _pooled_sd(summary["cv_gap_std"].values)
    sigma_pooled_sd = _pooled_sd(summary["cv_val_r2_run_to_run_sigma"].values)

    tol = 1e-12  # guard against floating-point ties at the boundary
    order = acc.sort_values(ascending=False).index.tolist()

    decision = None
    per_cfg = {}
    for cfg in order:
        gap_dist = float(gap[cfg] - min_gap)
        sigma_dist = float(sigma[cfg] - min_sigma)
        gap_ok = gap_dist <= gap_pooled_sd + tol
        sigma_ok = sigma_dist <= sigma_pooled_sd + tol
        per_cfg[cfg] = dict(
            acc=float(acc[cfg]), gap=float(gap[cfg]), gap_dist=gap_dist,
            gap_ok=gap_ok, sigma=float(sigma[cfg]), sigma_dist=sigma_dist,
            sigma_ok=sigma_ok,
        )
        if decision is None and gap_ok and sigma_ok:
            decision = cfg

    fallback_note = ""
    if decision is None:
        # No config satisfied both constraints; fall back to the accuracy leader.
        decision = accuracy_leader
        fallback_note = ("No config satisfied both stability constraints; "
                         "fell back to the accuracy leader.")

    # --- Build the human-readable reasoning ---
    L = []
    L.append("=" * 72)
    L.append("QNN STRUCTURE SELECTION — STABILITY-AWARE DECISION")
    L.append("=" * 72)
    L.append("")
    L.append("Decision basis: CROSS-VALIDATION ONLY. The held-out test set was "
             "NOT consulted")
    L.append("to make this decision; test_r2 / test_gap are reported for "
             "transparency and")
    L.append("cross-environment reproduction checking only.")
    L.append("")
    L.append("Leaders:")
    L.append(f"  - Accuracy leader (max mean cv_val_r2): {accuracy_leader} "
             f"({acc[accuracy_leader]:.4f})")
    L.append(f"  - Smallest mean cv_gap:                 {smallest_gap_cfg} "
             f"({min_gap:.4f})")
    L.append(f"  - Smallest run-to-run sigma:            {smallest_sigma_cfg} "
             f"({min_sigma:.4f})")
    L.append("")
    L.append("Pooled standard-deviation yardsticks (RMS of per-config across-seed SDs):")
    L.append(f"  - cv_gap pooled SD:               {gap_pooled_sd:.4f}")
    L.append(f"  - run-to-run sigma pooled SD:     {sigma_pooled_sd:.4f}")
    L.append("")
    L.append("A config passes a constraint when its distance from the best value "
             "is within")
    L.append("one pooled SD. Per-config evaluation, ordered by mean cv_val_r2 "
             "(descending):")
    L.append("")
    for rank, cfg in enumerate(order, 1):
        c = per_cfg[cfg]
        L.append(f"  [{rank}] {cfg}")
        L.append(f"        mean cv_val_r2      = {c['acc']:.4f}")
        L.append(f"        mean cv_gap         = {c['gap']:.4f}  "
                 f"(distance to smallest = {c['gap_dist']:.4f}; "
                 f"<= {gap_pooled_sd:.4f}? {'YES' if c['gap_ok'] else 'NO'})")
        L.append(f"        run-to-run sigma    = {c['sigma']:.4f}  "
                 f"(distance to smallest = {c['sigma_dist']:.4f}; "
                 f"<= {sigma_pooled_sd:.4f}? {'YES' if c['sigma_ok'] else 'NO'})")
        both = c["gap_ok"] and c["sigma_ok"]
        L.append(f"        satisfies BOTH stability constraints? "
                 f"{'YES' if both else 'NO'}")
        L.append("")
    if fallback_note:
        L.append(f"NOTE: {fallback_note}")
        L.append("")

    # Rationale
    if decision == accuracy_leader and not fallback_note:
        rationale = ("the accuracy leader also satisfies both stability "
                     "constraints, so it wins outright.")
    elif not fallback_note:
        rationale = (f"the accuracy leader ({accuracy_leader}) failed a stability "
                     f"constraint, so selection fell to the highest-accuracy "
                     f"config that satisfies both.")
    else:
        rationale = fallback_note

    L.append("-" * 72)
    L.append(f"RECOMMENDED CONFIG: {decision}")
    L.append(f"Rationale: {rationale}")
    L.append(f"  mean cv_val_r2 = {per_cfg[decision]['acc']:.4f} | "
             f"mean cv_gap = {per_cfg[decision]['gap']:.4f} | "
             f"run-to-run sigma = {per_cfg[decision]['sigma']:.4f}")
    test_r2_dec = float(summary.set_index("config").loc[decision, "test_r2_mean"])
    L.append(f"  (transparency) mean test_r2 = {test_r2_dec:.4f}")
    L.append("Decision used cross-validation only; the test set was not consulted.")
    L.append("=" * 72)
    return decision, "\n".join(L)


