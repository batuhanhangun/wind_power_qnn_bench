"""Stage 2 — depolarizing-noise sweep on the reverse_linear QNN (frozen weights).

Depends on Stage 1 (src/run_main.py). For each (seed, train_size):
  1. Load the Stage 1 weights  results/run_<RUN_ID>/seed_<seed>/exp_<size>_weights.npy
     (REQUIRED — Stage 2 never trains from scratch as its source of truth).
  2. Handoff consistency check: retrain noise-free with the same seed and assert
     the freshly retrained weights match the loaded weights within 1e-10. This
     proves determinism and that the handoff is sound; mismatch → fail loudly.
  3. Evaluate the LOADED frozen weights at the six noise levels in
     config.NOISE_LEVELS using default.mixed with a DepolarizingChannel after
     every gate. p=0 routes through lightning.qubit.
  4. The p=0 test R2 must match Stage 1's noise-free test R2 within 1e-6;
     mismatch → fail loudly.
  5. Write  results/run_<RUN_ID>/seed_<seed>/exp_<size>_noise_metrics.json.

Usage:
    python -m scripts.run_noise --seeds 42 --qnn-results results/runs/qnn
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from qnnbench import config
from qnnbench.circuits import WEIGHT_SHAPE, make_noisy_circuit
from qnnbench.qnn import train_model
from qnnbench._io import compute_metrics, load_data, print_flush, save_metrics

WEIGHT_MATCH_TOL = 1e-10   # loaded vs retrained weights
P0_MATCH_TOL = 1e-6        # Stage 2 p=0 vs Stage 1 noise-free test R2


class HandoffError(RuntimeError):
    """Raised when a Stage 1 -> Stage 2 consistency check fails."""


# --- Noisy evaluation ---

def evaluate_under_noise(weights_2d, X_test_scaled, scaler_y, y_test):
    """Evaluate frozen weights at every noise level in config.NOISE_LEVELS.

    Returns a list of dicts, one per noise level, each with noise_p + metrics.
    """
    results = []

    for p in config.NOISE_LEVELS:
        label = "noise-free" if p == 0.0 else f"p={p}"
        t0 = time.time()
        try:
            circuit_fn = make_noisy_circuit(p)

            y_pred_scaled = np.array([
                float(circuit_fn(weights_2d, x)) for x in X_test_scaled
            ])

            n_bad = int(np.sum(~np.isfinite(y_pred_scaled)))
            if n_bad > 0:
                print_flush(
                    f"    [{label}]  WARNING: {n_bad}/{len(y_pred_scaled)} "
                    f"non-finite predictions — replacing with 0"
                )
                y_pred_scaled = np.where(
                    np.isfinite(y_pred_scaled), y_pred_scaled, 0.0
                )

            y_pred = scaler_y.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).ravel()
            y_pred = np.clip(y_pred, 0, None)

            metrics = compute_metrics(y_test, y_pred)
            elapsed = time.time() - t0

            print_flush(
                f"    [{label}]  R2={metrics['r2']:.4f}  "
                f"RMSE={metrics['rmse']:.2f}  t={elapsed:.1f}s"
            )

            results.append({
                "noise_p": p,
                "test_r2": metrics["r2"],
                "test_mse": metrics["mse"],
                "test_rmse": metrics["rmse"],
                "test_mae": metrics["mae"],
            })

        except Exception as e:
            elapsed = time.time() - t0
            print_flush(f"    [{label}]  FAILED after {elapsed:.1f}s: {e}")
            results.append({
                "noise_p": p,
                "test_r2": None,
                "test_mse": None,
                "test_rmse": None,
                "test_mae": None,
                "error": str(e),
            })

    return results


# --- Experiment runner ---

def run_experiment(train_size, seed):
    """Run one Stage 2 (seed, train_size) experiment with handoff checks."""
    print_flush(f"\n{'='*60}")
    print_flush(f"Stage 2 (noise sweep) — reverse_linear QNN")
    print_flush(f"  Train size: {train_size}  Seed: {seed}")
    print_flush(f"  Noise levels: {config.NOISE_LEVELS}")
    print_flush(f"{'='*60}")

    start = time.time()

    X_train, y_train, X_test, y_test = load_data(seed, train_size)
    print_flush(f"Data: {len(X_train)} train, {len(X_test)} test")

    # Scale identically to Stage 1's final phase (fit on full train, transform test).
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(
        np.array(y_train).reshape(-1, 1)
    ).ravel()
    X_test_scaled = scaler_X.transform(X_test)

    seed_dir = Path(config.RESULTS_DIR) / f"seed_{seed}"
    weights_path = seed_dir / f"exp_{train_size}_weights.npy"
    metrics_path = seed_dir / f"exp_{train_size}_metrics.json"

    # --- Step 1: load Stage 1 weights (required) ---
    if not weights_path.exists():
        raise HandoffError(
            f"Stage 1 weights not found: {weights_path}. "
            f"Run Stage 1 (src.run_main) for this run id first."
        )
    loaded_weights = np.load(str(weights_path)).reshape(WEIGHT_SHAPE)
    print_flush(f"Loaded Stage 1 weights: {weights_path}")

    # --- Step 2: handoff check — retrain and assert determinism (1e-10) ---
    print_flush("Handoff check: retraining noise-free (deterministic)...")
    t_train = time.time()
    retrained_weights = train_model(
        X_train_scaled, y_train_scaled,
        maxiter=config.QNN_CONFIG["maxiter"], seed=seed,
    )
    retrained_weights = np.asarray(retrained_weights).reshape(WEIGHT_SHAPE)
    weight_diff = float(np.max(np.abs(loaded_weights - retrained_weights)))
    print_flush(f"  retrain in {time.time() - t_train:.1f}s, "
                f"max|loaded - retrained| = {weight_diff:.3e}")
    if not (weight_diff < WEIGHT_MATCH_TOL):
        raise HandoffError(
            f"Weight handoff FAILED for seed={seed}, size={train_size}: "
            f"max|d|={weight_diff:.3e} >= {WEIGHT_MATCH_TOL:.0e}. "
            f"Loaded weights are not reproducible — refusing to proceed."
        )
    print_flush(f"  Handoff OK (weights match within {WEIGHT_MATCH_TOL:.0e}).")

    # --- Step 3: noise sweep on the LOADED frozen weights ---
    print_flush("\nEvaluating under noise:")
    noise_results = evaluate_under_noise(
        loaded_weights, X_test_scaled, scaler_y, y_test
    )

    if not noise_results:
        raise HandoffError("No noise-level results produced.")

    # --- Step 4: p=0 must match Stage 1 noise-free test R2 within 1e-6 ---
    p0_entry = next((e for e in noise_results if e["noise_p"] == 0.0), None)
    if p0_entry is None or p0_entry.get("test_r2") is None:
        raise HandoffError("p=0 noise-free evaluation missing or failed.")

    if not metrics_path.exists():
        raise HandoffError(f"Stage 1 metrics not found: {metrics_path}.")
    with open(metrics_path) as f:
        stage1 = json.load(f)
    stage1_test_r2 = float(stage1["test_r2"])
    p0_diff = abs(float(p0_entry["test_r2"]) - stage1_test_r2)
    print_flush(
        f"\np=0 check: Stage2 R2={p0_entry['test_r2']:.10f}  "
        f"Stage1 R2={stage1_test_r2:.10f}  |d|={p0_diff:.3e}"
    )
    if not (p0_diff < P0_MATCH_TOL):
        raise HandoffError(
            f"p=0 mismatch for seed={seed}, size={train_size}: "
            f"|d|={p0_diff:.3e} >= {P0_MATCH_TOL:.0e}. "
            f"Noise-free baseline does not reproduce Stage 1 — refusing to save."
        )
    print_flush(f"  p=0 OK (matches Stage 1 within {P0_MATCH_TOL:.0e}).")

    # Attach metadata + handoff diagnostics
    for entry in noise_results:
        entry.update({
            "model": "qnn",
            "architecture": config.ENTANGLEMENT,
            "train_size": train_size,
            "seed": seed,
        })
    noise_results_meta = {
        "weight_handoff_max_abs_diff": weight_diff,
        "p0_vs_stage1_abs_diff": p0_diff,
    }

    # --- Step 5: save ---
    out_path = seed_dir / f"exp_{train_size}_noise_metrics.json"
    # Persist the per-level list (consumed by aggregate_rerun) plus a sidecar
    # of handoff diagnostics under a reserved key the aggregator ignores.
    payload = {"levels": noise_results, "handoff": noise_results_meta}
    save_metrics(payload, str(out_path))
    print_flush(f"Saved: {out_path}")
    print_flush(f"Total time: {time.time() - start:.1f}s")


# --- Entry point ---

