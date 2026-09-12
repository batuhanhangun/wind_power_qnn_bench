"""Aggregate the Q5 per-seed runs into the head-to-head deliverables.

Reads every results/run_<RUN_ID>/seed_<seed>/noise_aware_metrics.json and emits
to aggregated/run_<RUN_ID>/ (CSV + text only - NO plotting):

  noise_aware_per_seed.csv   one row per (seed, noise_p, mode), mode in
                             {freeze_eval, freeze_eval_calibrated, noise_aware};
                             columns seed, noise_p, mode, train_size, test_r2,
                             train_r2, generalization_gap, test_mse, test_rmse,
                             test_mae.
  noise_aware_summary.csv    per (noise_p, mode): mean/std (ddof=1) over the 10
                             seeds of test_r2 and the error metrics, plus the
                             across-seed run-to-run sigma of test_r2.
  noise_aware_report.txt     [1] fidelity-gate table (measured vs published, all
                             four levels, PASS/FAIL) at the top; [2] head-to-head
                             (one row per p): freeze_eval | calibrated |
                             noise_aware | d(NA-FE) | d(NA-cal) | Wilcoxon p NA-FE
                             | Wilcoxon p NA-cal; [3] two-part Q5 verdict — (i) NA
                             vs plain freeze_eval and (ii) NA vs the calibrated
                             baseline (part ii decides "retrain under noise" vs
                             merely "calibrate the output"); [4] the noise-free
                             mean R2 ~0.903 anchor gate.

The fidelity gate is a 10-seed-MEAN check, so it lives here rather than in the
per-seed driver. If it fails, the noise injection was altered and the Q5
comparison is invalid - the script writes all outputs, then exits non-zero.

Aggregation is called automatically by the corresponding scripts.run_* entry point.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from qnnbench import config

ALPHA = 0.05   # significance level for the paired Wilcoxon test
METRICS = ["test_r2", "test_mse", "test_rmse", "test_mae"]


# --- Collection ---

def collect():
    """Return (per_seed_rows, noise_free_by_seed) from per-seed JSONs."""
    rows = []
    noise_free = {}
    results_dir = Path(config.RESULTS_DIR)
    if not results_dir.exists():
        return rows, noise_free

    for seed_dir in sorted(results_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        seed = int(seed_dir.name.split("_")[1])
        # Prefer the full-run file; ignore single-level debug files.
        mf = seed_dir / "noise_aware_metrics.json"
        if not mf.exists():
            print(f"  Skipping {seed_dir.name}: no full noise_aware_metrics.json")
            continue
        try:
            with open(mf) as f:
                payload = json.load(f)
            noise_free[seed] = payload.get("noise_free_test_r2")
            for r in payload["rows"]:
                rows.append(r)
        except Exception as e:
            print(f"Warning: could not parse {mf}: {e}")
    return rows, noise_free


# --- Summary ---

def build_summary(df):
    """Per (noise_p, mode): mean/std (ddof=1) + run-to-run sigma of test_r2."""
    out = []
    for (p, mode), g in df.groupby(["noise_p", "mode"]):
        row = {"noise_p": float(p), "mode": mode, "n_seeds": len(g)}
        for m in METRICS:
            row[f"mean_{m}"] = float(g[m].mean())
            row[f"std_{m}"] = float(g[m].std(ddof=1))
        # Across-seed run-to-run sigma of test R2 (== std_test_r2; named
        # explicitly because it is the headline stability number for Q5).
        row["run_to_run_sigma_test_r2"] = float(g["test_r2"].std(ddof=1))
        out.append(row)
    return (pd.DataFrame(out)
            .sort_values(["noise_p", "mode"])
            .reset_index(drop=True))


# --- Head-to-head + Wilcoxon ---

def _paired_wilcoxon(a_v, b_v):
    """Two-sided paired Wilcoxon p-value for a_v vs b_v (None if undefined)."""
    if np.allclose(a_v - b_v, 0.0):
        return None   # undefined when every paired difference is zero
    try:
        return float(wilcoxon(a_v, b_v, alternative="two-sided").pvalue)
    except ValueError:
        return None


def paired_head_to_head(df):
    """Per p: paired test_r2 across seeds for the three modes + both Wilcoxons.

    Modes: freeze_eval (FE), freeze_eval_calibrated (CAL), noise_aware (NA). The
    two head-to-head questions are NA vs FE (does training under noise beat plain
    freeze-and-evaluate?) and NA vs CAL (does it still beat the cheap linear
    calibration control?). Returns a list of dicts with per-mode means, both
    deltas, both two-sided paired Wilcoxon p-values, and the paired-seed count.
    """
    results = []
    for p in config.NOISE_AWARE_LEVELS:
        def series(mode):
            return (df[(df.noise_p == p) & (df["mode"] == mode)]
                    .set_index("seed")["test_r2"])
        fe, cal, na = series("freeze_eval"), \
            series("freeze_eval_calibrated"), series("noise_aware")

        # Pair on seeds present in ALL three modes so every column is comparable.
        common = sorted(set(fe.index) & set(cal.index) & set(na.index))
        if len(common) == 0:
            results.append({"noise_p": p, "n_pairs": 0,
                            "freeze_eval_mean_r2": None,
                            "calibrated_mean_r2": None,
                            "noise_aware_mean_r2": None,
                            "delta_na_fe": None, "delta_na_cal": None,
                            "wilcoxon_p_na_fe": None, "wilcoxon_p_na_cal": None})
            continue

        fe_v = fe.loc[common].to_numpy()
        cal_v = cal.loc[common].to_numpy()
        na_v = na.loc[common].to_numpy()

        results.append({
            "noise_p": p,
            "n_pairs": len(common),
            "freeze_eval_mean_r2": float(np.mean(fe_v)),
            "calibrated_mean_r2": float(np.mean(cal_v)),
            "noise_aware_mean_r2": float(np.mean(na_v)),
            "delta_na_fe": float(np.mean(na_v) - np.mean(fe_v)),
            "delta_na_cal": float(np.mean(na_v) - np.mean(cal_v)),
            "wilcoxon_p_na_fe": _paired_wilcoxon(na_v, fe_v),
            "wilcoxon_p_na_cal": _paired_wilcoxon(na_v, cal_v),
        })
    return results


# --- Gates ---

def gate_noise_free(noise_free_by_seed):
    """Strict gate: mean noise-free test R2 reproduces ~0.903."""
    vals = [v for v in noise_free_by_seed.values() if v is not None]
    if not vals:
        return False, None, None
    mean = float(np.mean(vals))
    ok = abs(mean - config.EXPECTED_NOISE_FREE_R2) <= config.FIDELITY_TOL
    return ok, mean, abs(mean - config.EXPECTED_NOISE_FREE_R2)


def gate_fidelity(head):
    """Strict gate: freeze-eval means reproduce the published table within tol.

    Returns (all_ok, [(p, measured, expected, |d|, ok), ...]).
    """
    details, all_ok = [], True
    by_p = {h["noise_p"]: h for h in head}
    for p, exp in config.EXPECTED_FREEZE_EVAL_R2.items():
        h = by_p.get(p)
        measured = h["freeze_eval_mean_r2"] if h else None
        if measured is None:
            details.append((p, None, exp, None, False))
            all_ok = False
            continue
        d = abs(measured - exp)
        ok = d <= config.FIDELITY_TOL
        all_ok = all_ok and ok
        details.append((p, measured, exp, d, ok))
    return all_ok, details


# --- Verdict ---

def _sig(delta, wp):
    """True if delta>0 and the paired Wilcoxon p-value is significant."""
    if delta is None or wp is None:
        return False
    return (delta > 0) and (wp < ALPHA)


def _sig_na_fe(h):
    """noise_aware significantly beats plain freeze_eval at this level."""
    return _sig(h.get("delta_na_fe"), h.get("wilcoxon_p_na_fe"))


def _sig_na_cal(h):
    """noise_aware significantly beats the calibrated control at this level."""
    return _sig(h.get("delta_na_cal"), h.get("wilcoxon_p_na_cal"))


def build_verdict(head):
    """Return (lines, beats_fe, beats_cal) summarizing the two-part Q5 finding.

    Part (i)  : does noise_aware beat plain freeze-and-evaluate?
    Part (ii) : does noise_aware STILL beat the calibrated baseline? This is the
                one that decides "retrain under noise" vs merely "calibrate the
                output".
    """
    by_p = {h["noise_p"]: h for h in head}
    lines = []

    def verdict_for(p):
        h = by_p.get(p)
        if h is None or h["delta_na_fe"] is None:
            return f"p={p}: no paired data."
        wp_fe = ("n/a" if h["wilcoxon_p_na_fe"] is None
                 else f"{h['wilcoxon_p_na_fe']:.4f}")
        wp_cal = ("n/a" if h["wilcoxon_p_na_cal"] is None
                  else f"{h['wilcoxon_p_na_cal']:.4f}")
        tag_fe = " WINS" if _sig_na_fe(h) else " n.s."
        tag_cal = " WINS" if _sig_na_cal(h) else " n.s."
        return (f"p={p}: NA-vs-FE  delta={h['delta_na_fe']:+.4f} "
                f"(Wilcoxon p={wp_fe}){tag_fe}   |   "
                f"NA-vs-CAL delta={h['delta_na_cal']:+.4f} "
                f"(Wilcoxon p={wp_cal}){tag_cal}")

    envelope_ps = [p for p in config.NOISE_AWARE_LEVELS
                   if p <= config.ENVELOPE_MAX_P]
    lines.append(f"Operational envelope (p <= {config.ENVELOPE_MAX_P}):")
    for p in envelope_ps:
        lines.append("    " + verdict_for(p))
    lines.append(f"Envelope edge (p = {config.EDGE_P}):")
    lines.append("    " + verdict_for(config.EDGE_P))
    lines.append(f"Stress level (p = {config.STRESS_P}):")
    lines.append("    " + verdict_for(config.STRESS_P))

    beats_fe = any(_sig_na_fe(by_p[p])
                   for p in config.NOISE_AWARE_LEVELS if p in by_p)
    beats_cal = any(_sig_na_cal(by_p[p])
                    for p in config.NOISE_AWARE_LEVELS if p in by_p)
    return lines, beats_fe, beats_cal


# --- Report ---

def write_report(path, summary_df, head, noise_free_by_seed):
    L = []
    L.append("=" * 88)
    L.append("Q5 - noise-aware training vs freeze-and-evaluate vs calibrated "
             "control (reverse_linear, N=800)")
    L.append(f"Run ID: {config.RUN_ID}")
    L.append("=" * 88)
    L.append("")

    # --- [1] Fidelity gate at the TOP (measured vs published, all four levels) ---
    L.append("[1] Noise-model fidelity gate - freeze-eval means vs published "
             "reverse_linear table")
    L.append("")
    L.append(f"    {'p':>7} | {'measured':>9} | {'published':>9} | {'|d|':>7} | "
             f"{'result':>6}")
    L.append("    " + "-" * 52)
    ok_fid, details = gate_fidelity(head)
    for p, measured, exp, d, ok in details:
        if measured is None:
            L.append(f"    {p:>7} | {'--':>9} | {exp:>9.3f} | {'--':>7} | "
                     f"{'FAIL':>6} (missing)")
        else:
            L.append(f"    {p:>7} | {measured:>9.3f} | {exp:>9.3f} | {d:>7.3f} | "
                     f"{'PASS' if ok else 'FAIL':>6}")
    L.append("")
    L.append(f"    Fidelity gate: {'PASS' if ok_fid else 'FAIL'} (tol "
             f"{config.FIDELITY_TOL}) - "
             + ("noise injection matches the published reverse_linear table."
                if ok_fid else
                "noise injection DIFFERS from the paper; comparison is INVALID."))
    L.append("")

    # --- [2] Head-to-head (three modes, two deltas, two Wilcoxons) ---
    L.append("[2] Head-to-head: test R2 at matched noise levels (mean over seeds)")
    L.append("")
    L.append(f"    {'p':>7} | {'freeze_eval':>11} | {'calibrated':>11} | "
             f"{'noise_aware':>11} | {'d(NA-FE)':>9} | {'d(NA-cal)':>9} | "
             f"{'W p NA-FE':>10} | {'W p NA-cal':>10} | n")
    L.append("    " + "-" * 108)
    for h in sorted(head, key=lambda x: x["noise_p"]):
        if h["freeze_eval_mean_r2"] is None:
            L.append(f"    {h['noise_p']:>7} | " + " | ".join(["--".rjust(w)
                     for w in (11, 11, 11, 9, 9, 10, 10)]) + f" | {h['n_pairs']}")
            continue
        wp_fe = ("n/a" if h["wilcoxon_p_na_fe"] is None
                 else f"{h['wilcoxon_p_na_fe']:.4f}")
        wp_cal = ("n/a" if h["wilcoxon_p_na_cal"] is None
                  else f"{h['wilcoxon_p_na_cal']:.4f}")
        L.append(
            f"    {h['noise_p']:>7} | {h['freeze_eval_mean_r2']:>11.4f} | "
            f"{h['calibrated_mean_r2']:>11.4f} | {h['noise_aware_mean_r2']:>11.4f} | "
            f"{h['delta_na_fe']:>+9.4f} | {h['delta_na_cal']:>+9.4f} | "
            f"{wp_fe:>10} | {wp_cal:>10} | {h['n_pairs']}")
    L.append("")
    L.append("    d(NA-FE)  = noise_aware - freeze_eval  (paper's protocol comparison)")
    L.append("    d(NA-cal) = noise_aware - calibrated   (vs the cheap rescale control)")
    L.append(f"    Paired two-sided Wilcoxon over the seeds; alpha={ALPHA}. "
             "Positive delta => noise-aware better.")
    L.append("")

    # --- [3] Two-part Q5 verdict ---
    L.append("[3] Q5 verdict")
    verdict_lines, beats_fe, beats_cal = build_verdict(head)
    for ln in verdict_lines:
        L.append("    " + ln)
    L.append("")
    L.append("    PART (i) - does noise-aware beat plain freeze-and-evaluate?")
    if beats_fe:
        L.append("      YES at one or more levels (significant test-R2 gain over "
                 "freeze_eval).")
    else:
        L.append("      NO - noise-aware does not significantly beat freeze_eval "
                 "at any level.")
    L.append("")
    L.append("    PART (ii) - does noise-aware STILL beat the calibrated control?")
    if beats_cal:
        L.append("      YES at one or more levels (significant test-R2 gain over "
                 "the calibrated baseline).")
    else:
        L.append("      NO - a single linear rescale of the frozen model's noisy "
                 "outputs captures")
        L.append("      the gain; noise-aware training adds nothing beyond it.")
    L.append("")
    L.append("    RECOMMENDATION:")
    if beats_cal:
        L.append("      RETRAIN UNDER NOISE - noise-aware training beats BOTH "
                 "freeze-and-evaluate")
        L.append("      and the cheap calibration control, so the retraining "
                 "cost is justified.")
    elif beats_fe:
        L.append("      CALIBRATE THE OUTPUT - noise-aware beats freeze_eval, but "
                 "NOT the calibrated")
        L.append("      baseline: the apparent gain is merely a global rescale "
                 "correcting the")
        L.append("      deterministic depolarizing contraction, obtainable "
                 "WITHOUT retraining.")
    else:
        L.append("      KEEP FREEZE-AND-EVALUATE - noise-aware training beats "
                 "neither freeze_eval")
        L.append("      nor the calibration control; the paper's cheaper "
                 "protocol is justified.")
    L.append("")

    # --- [4] Correctness gate - noise-free anchor ---
    L.append("[4] Correctness gate - noise-free N=800 mean test R2")
    ok_nf, mean_nf, d_nf = gate_noise_free(noise_free_by_seed)
    if mean_nf is None:
        L.append("    MISSING - no noise-free results found.")
    else:
        L.append(f"    measured : {mean_nf:.4f}   expected : "
                 f"~{config.EXPECTED_NOISE_FREE_R2:.3f}   |d| : {d_nf:.4f}   "
                 f"-> {'PASS' if ok_nf else 'FAIL'} (tol {config.FIDELITY_TOL})")
    L.append("=" * 88)

    text = "\n".join(L) + "\n"
    Path(path).write_text(text, encoding="utf-8")
    print(text)
    return ok_nf, ok_fid


# --- Main ---

