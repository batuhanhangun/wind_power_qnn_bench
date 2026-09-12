"""Step 3 fidelity re-verify — the mandatory gate before the expensive array.

Recomputes ONLY the freeze-and-evaluate curve (train noise-free on
lightning.qubit, freeze the weights, evaluate under the fixed noisy circuit) at
p in {0.001, 0.005, 0.01, 0.02} across the 10 seeds, and asserts the per-level
mean reproduces the N=800 reverse_linear reference values within FIDELITY_TOL
(0.01) at ALL FOUR levels:

    0.001 -> 0.893
    0.005 -> 0.814
    0.010 -> 0.659
    0.020 -> 0.299

(NOT the pooled published table 0.892/0.809/0.649/0.277, which averages four
training sizes and is the wrong reference for an N=800-only measurement.)

This is cheap relative to the noise-aware sweep (no density-matrix TRAINING —
only one lightning.qubit noise-free train per seed plus frozen-weight noisy
evaluation). It exists so the expensive noise-aware array is NEVER launched on a
noise model that fails to reproduce the anchors. Prints measured vs expected per
level; exits 0 iff all four levels pass, non-zero otherwise.

Usage:
    python -m qnnbench.verify_fidelity --seeds 42
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from qnnbench import config
from qnnbench.circuits import circuit, make_noisy_circuit
from qnnbench.noise_aware import scale_data, train_model, predict_original_scale
from qnnbench._io import compute_metrics, load_data, print_flush


def freeze_eval_curve_for_seed(seed, levels, maxiter):
    """Train noise-free, freeze, and return {p: test_r2} over the given levels."""
    X_train, y_train, X_test, y_test = load_data(seed, config.TRAIN_SIZE)
    X_train_scaled, y_train_scaled, X_test_scaled, _, scaler_y = scale_data(
        X_train, y_train, X_test
    )
    weights_clean = train_model(
        circuit, X_train_scaled, y_train_scaled, maxiter, seed
    )
    out = {}
    for p in levels:
        eval_fn = make_noisy_circuit(p)
        y_pred = predict_original_scale(
            eval_fn, weights_clean, X_test_scaled, scaler_y
        )
        out[p] = compute_metrics(y_test, y_pred)["r2"]
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: re-verify the freeze-eval fidelity gate (all levels)"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=config.SEEDS,
                        help="Seeds to average over (default: all 10).")
    args = parser.parse_args()

    levels = config.NOISE_AWARE_LEVELS
    maxiter = config.QNN_CONFIG["maxiter"]

    print_flush("=" * 64)
    print_flush("Step 3 — freeze-and-evaluate fidelity re-verify")
    print_flush(f"  Seeds: {args.seeds}")
    print_flush(f"  Levels: {levels}   tol: {config.FIDELITY_TOL}")
    print_flush("=" * 64)

    start = time.time()
    per_seed = {}
    for seed in args.seeds:
        t0 = time.time()
        per_seed[seed] = freeze_eval_curve_for_seed(seed, levels, maxiter)
        print_flush(
            f"  seed {seed}: "
            + "  ".join(f"p={p}:{per_seed[seed][p]:.4f}" for p in levels)
            + f"   ({time.time() - t0:.1f}s)"
        )

    # Per-level mean over seeds, compared to the published anchors.
    print_flush("\n  measured vs published (mean over seeds):")
    print_flush(f"    {'p':>7} | {'measured':>9} | {'published':>9} | "
                f"{'|d|':>7} | {'result':>6}")
    print_flush("    " + "-" * 52)
    all_ok = True
    for p in levels:
        vals = [per_seed[s][p] for s in args.seeds]
        measured = float(np.mean(vals))
        exp = config.EXPECTED_FREEZE_EVAL_R2[p]
        d = abs(measured - exp)
        ok = d <= config.FIDELITY_TOL
        all_ok = all_ok and ok
        print_flush(f"    {p:>7} | {measured:>9.4f} | {exp:>9.3f} | "
                    f"{d:>7.4f} | {'PASS' if ok else 'FAIL':>6}")

    print_flush("")
    print_flush(f"  Total time: {time.time() - start:.1f}s")
    print_flush("=" * 64)
    if all_ok:
        print_flush("FIDELITY GATE PASSED at all four levels — safe to launch "
                    "the noise-aware array.")
        sys.exit(0)
    else:
        print_flush("FIDELITY GATE FAILED — noise model does not reproduce the "
                    "published anchors. STOP: do NOT launch the array.")
        sys.exit(1)


if __name__ == "__main__":
    main()
