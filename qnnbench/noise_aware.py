"""Q5 — noise-aware vs freeze-and-evaluate vs calibrated control, per seed, N=800.

For one seed, produces the THREE curves of the Q5 head-to-head at each noise level:

  1. Train noise-free on lightning.qubit (adjoint) -> weights_clean.
     Record its noise-free test R2 (expected ~0.903).
  2. FREEZE-AND-EVALUATE curve (freeze_eval): with weights_clean frozen, evaluate
     on default.mixed at each p in config.NOISE_AWARE_LEVELS (the paper's protocol).
  2b. CALIBRATED control (freeze_eval_calibrated): take the SAME frozen noise-free
     weights, evaluate at p, fit a single slope+intercept linear calibration on
     the TRAINING predictions -> targets (no test leakage), apply to test. This
     captures how much of any noise-aware gain is merely a global rescale
     correcting the deterministic depolarizing contraction, WITHOUT retraining.
  3. NOISE-AWARE curve (noise_aware): for each p, train a FRESH model on
     default.mixed with DepolarizingChannel(p) active during training (same weight
     init as the noise-free run for that seed) -> weights_p; evaluate at the same p.

The two curves use the SAME 10 seeds and the SAME noise injection (the latter
guaranteed by tests/test_noise_aware.py). Writes one JSON per seed:
    results/run_<RUN_ID>/seed_<seed>/noise_aware_metrics.json

Reused verbatim from the reverse_linear rerun: the circuit, the L-BFGS-B /
qml.grad(argnums=0) training recipe, the weight init, the MinMaxScaler scaling,
and the frozen-weight noisy evaluation path. The only addition is training the
noisy circuit (make_trainable_noisy_circuit).

Devices / differentiation:
  noise-free training & p=0 eval : lightning.qubit, adjoint  (fast, matches paper)
  noise-aware training           : default.mixed, backprop, autograd interface
  evaluation under noise (both modes): default.mixed (exact density matrix)

Per-seed correctness signal (informational; the strict gates run at aggregation,
see src/aggregate_noise_aware.py and tests/test_noise_aware.py):
  prints measured-vs-expected noise-free test R2 and freeze-eval test R2 per p.

Usage:
    python -m scripts.run_noise_aware --seeds 42
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler

from qnnbench import config
from qnnbench.circuits import (
    circuit,
    make_noisy_circuit,
    make_trainable_noisy_circuit,
    NUM_QUBITS,
    ANSATZ_REPS,
    WEIGHT_SHAPE,
)
from qnnbench._io import compute_metrics, load_data, print_flush, save_metrics


# --- Cost / training (generic over the circuit; argnums=0 gradient on weights) ---

def batch_predict(circuit_fn, weights_flat, X):
    """Run ``circuit_fn`` over a batch of inputs with flat weights."""
    weights_2d = weights_flat.reshape(WEIGHT_SHAPE)
    return pnp.array([circuit_fn(weights_2d, x) for x in X])


def cost_fn(circuit_fn, weights_flat, X, y):
    """MSE cost for training ``circuit_fn``."""
    preds = batch_predict(circuit_fn, weights_flat, X)
    return pnp.mean((preds - y) ** 2)


def train_model(circuit_fn, X_train_scaled, y_train_scaled, maxiter, seed):
    """Train a QNN with scipy L-BFGS-B (deterministic given the seed).

    Identical recipe to the reverse_linear rerun: weight init
    ``np.random.RandomState(seed).uniform(-pi, pi, 16)`` and gradient via
    ``qml.grad(cost_fn, argnums=0)``. The ONLY variable is ``circuit_fn`` — the
    noise-free lightning circuit (freeze-and-evaluate baseline) or a trainable
    noisy density-matrix circuit (noise-aware). Using the SAME seed for both the
    noise-free and the noise-aware runs of a seed gives them an identical init.
    """
    rng = np.random.RandomState(seed)
    init_weights = rng.uniform(-np.pi, np.pi, size=NUM_QUBITS * (ANSATZ_REPS + 1))

    X_pnp = pnp.array(X_train_scaled, requires_grad=False)
    y_pnp = pnp.array(y_train_scaled, requires_grad=False)

    # qml.grad(..., argnums=0) differentiates w.r.t. the weights (first arg of
    # the wrapped cost); circuit_fn is captured, not differentiated.
    grad_fn = qml.grad(
        lambda w, X, y: cost_fn(circuit_fn, w, X, y), argnums=0
    )

    result = minimize(
        fun=lambda w: float(cost_fn(
            circuit_fn, pnp.array(w, requires_grad=True), X_pnp, y_pnp
        )),
        x0=init_weights,
        method="L-BFGS-B",
        jac=lambda w: np.array(grad_fn(
            pnp.array(w, requires_grad=True), X_pnp, y_pnp
        )),
        options={"maxiter": maxiter},
    )

    return result.x.reshape(WEIGHT_SHAPE)


def predict_raw_original_scale(circuit_fn, weights, X_scaled, scaler_y):
    """Predict on scaled features; inverse-transform; return UNCLIPPED preds.

    Non-finite outputs (possible under heavy noise) are replaced with 0 before
    inverse scaling, matching the published noise harness. This is the raw
    original-scale prediction BEFORE the clip>=0 step, so the linear calibration
    control can be fit on an undistorted (purely linear) relationship.
    """
    y_pred_scaled = np.array([float(circuit_fn(weights, x)) for x in X_scaled])
    n_bad = int(np.sum(~np.isfinite(y_pred_scaled)))
    if n_bad > 0:
        print_flush(
            f"    WARNING: {n_bad}/{len(y_pred_scaled)} non-finite "
            f"predictions — replacing with 0"
        )
        y_pred_scaled = np.where(np.isfinite(y_pred_scaled), y_pred_scaled, 0.0)
    return scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()


def predict_original_scale(circuit_fn, weights, X_scaled, scaler_y):
    """Predict with ``circuit_fn`` on scaled features; inverse-transform; clip>=0.

    Identical to the published noise harness (inverse-transform then clip to
    non-negative power). Thin wrapper over ``predict_raw_original_scale``.
    """
    return np.clip(
        predict_raw_original_scale(circuit_fn, weights, X_scaled, scaler_y),
        0, None,
    )


def fit_linear_calibration(pred_train_raw, y_train):
    """Fit a slope+intercept linear map from noisy train predictions to targets.

    This is the cheap calibration control: it captures how much of any noise-
    aware gain is merely a global rescale correcting the deterministic
    depolarizing contraction (which shrinks the expectation value toward 0),
    obtainable WITHOUT retraining. Fit on TRAINING predictions only — no test
    leakage — via ordinary least squares.

    Returns:
        (slope, intercept) floats.
    """
    slope, intercept = np.polyfit(
        np.asarray(pred_train_raw, dtype=float),
        np.asarray(y_train, dtype=float),
        deg=1,
    )
    return float(slope), float(intercept)


def apply_linear_calibration(pred_raw, slope, intercept):
    """Apply the fitted linear calibration, then clip>=0 (same post-processing)."""
    return np.clip(slope * np.asarray(pred_raw, dtype=float) + intercept, 0, None)


# --- Scaling (identical to the rerun's final-training phase) ---

def scale_data(X_train, y_train, X_test):
    """MinMaxScaler on X and y; fit on full train, transform test."""
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(
        np.array(y_train).reshape(-1, 1)
    ).ravel()
    X_test_scaled = scaler_X.transform(X_test)
    return X_train_scaled, y_train_scaled, X_test_scaled, scaler_X, scaler_y


# --- Per-seed procedure ---

def _row(seed, p, mode, test_metrics, train_r2):
    """Assemble one per-seed CSV row dict for (seed, noise_p, mode)."""
    gap = (train_r2 - test_metrics["r2"]) if train_r2 is not None else None
    return {
        "seed": seed,
        "noise_p": p,
        "mode": mode,
        "train_size": config.TRAIN_SIZE,
        "test_r2": test_metrics["r2"],
        "train_r2": train_r2,
        "generalization_gap": gap,
        "test_mse": test_metrics["mse"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
    }


def run_seed(seed, only_p=None):
    """Run the full Q5 per-seed procedure (or a single level if only_p given)."""
    levels = config.NOISE_AWARE_LEVELS if only_p is None else [only_p]
    maxiter = config.QNN_CONFIG["maxiter"]

    print_flush(f"\n{'='*64}")
    print_flush("Q5 noise-aware training vs freeze-and-evaluate — reverse_linear")
    print_flush(f"  Seed: {seed}  Train size: {config.TRAIN_SIZE}")
    print_flush(f"  Levels: {levels}  (noise-aware diff: "
                f"{config.NOISE_AWARE_DIFF_METHOD})")
    print_flush(f"{'='*64}")

    start = time.time()
    X_train, y_train, X_test, y_test = load_data(seed, config.TRAIN_SIZE)
    print_flush(f"Loaded data: {len(X_train)} train, {len(X_test)} test")
    X_train_scaled, y_train_scaled, X_test_scaled, _, scaler_y = scale_data(
        X_train, y_train, X_test
    )

    # === Step 1: noise-free training (lightning.qubit, adjoint) ===
    print_flush("\n[1] Noise-free training (lightning.qubit, adjoint)...")
    t0 = time.time()
    weights_clean = train_model(
        circuit, X_train_scaled, y_train_scaled, maxiter, seed
    )
    print_flush(f"    trained in {time.time() - t0:.1f}s")

    y_train_pred_clean = predict_original_scale(
        circuit, weights_clean, X_train_scaled, scaler_y
    )
    noise_free_train_r2 = compute_metrics(y_train, y_train_pred_clean)["r2"]
    y_test_pred_clean = predict_original_scale(
        circuit, weights_clean, X_test_scaled, scaler_y
    )
    noise_free_test_r2 = compute_metrics(y_test, y_test_pred_clean)["r2"]
    print_flush(
        f"    noise-free test R2 = {noise_free_test_r2:.4f}  "
        f"(expected ~{config.EXPECTED_NOISE_FREE_R2:.3f}, "
        f"|d|={abs(noise_free_test_r2 - config.EXPECTED_NOISE_FREE_R2):.4f}; "
        f"per-seed — the strict gate is on the 10-seed mean at aggregation)"
    )
    print_flush(f"    noise-free train R2 = {noise_free_train_r2:.4f}")

    rows = []

    # === Step 2: FREEZE-AND-EVALUATE + CALIBRATED control (weights_clean frozen) ===
    # Both curves reuse the SAME frozen noise-free weights and the SAME noisy
    # evaluation path; the calibrated control additionally fits a single linear
    # rescale on TRAIN predictions and applies it to test (no retraining, no
    # test leakage).
    print_flush("\n[2] Freeze-and-evaluate + calibrated control "
                "(frozen noise-free weights):")
    for p in levels:
        eval_fn = make_noisy_circuit(p)
        t0 = time.time()
        # Raw (unclipped) noisy predictions on train and test at this p.
        train_pred_raw = predict_raw_original_scale(
            eval_fn, weights_clean, X_train_scaled, scaler_y)
        test_pred_raw = predict_raw_original_scale(
            eval_fn, weights_clean, X_test_scaled, scaler_y)

        # --- freeze_eval: the paper's protocol (clip>=0, no calibration) ---
        y_pred_fe = np.clip(test_pred_raw, 0, None)
        tm_fe = compute_metrics(y_test, y_pred_fe)
        # train_r2 is the noise-free training fit (same at every p); the gap then
        # shows the noise-induced test degradation vs that fit.
        rows.append(_row(seed, p, "freeze_eval", tm_fe, noise_free_train_r2))

        # --- freeze_eval_calibrated: fit slope+intercept on TRAIN preds only ---
        slope, intercept = fit_linear_calibration(train_pred_raw, y_train)
        y_train_cal = apply_linear_calibration(train_pred_raw, slope, intercept)
        y_test_cal = apply_linear_calibration(test_pred_raw, slope, intercept)
        cal_train_r2 = compute_metrics(y_train, y_train_cal)["r2"]
        tm_cal = compute_metrics(y_test, y_test_cal)
        rows.append(_row(seed, p, "freeze_eval_calibrated", tm_cal, cal_train_r2))

        exp = config.EXPECTED_FREEZE_EVAL_R2.get(p)
        exp_str = (f"  (published mean ~{exp:.3f}, "
                   f"|d|={abs(tm_fe['r2'] - exp):.3f})") if exp is not None else ""
        print_flush(
            f"    p={p:<6}  freeze_eval R2={tm_fe['r2']:.4f}  "
            f"calibrated R2={tm_cal['r2']:.4f}  "
            f"(slope={slope:.3f}, intercept={intercept:.1f})  "
            f"t={time.time() - t0:.1f}s{exp_str}")

    # === Step 3: NOISE-AWARE curve (one fresh model trained per level) ===
    print_flush("\n[3] Noise-aware curve (train under noise, eval at same p):")
    for p in levels:
        train_fn = make_trainable_noisy_circuit(p)
        t0 = time.time()
        weights_p = train_model(
            train_fn, X_train_scaled, y_train_scaled, maxiter, seed
        )
        train_time = time.time() - t0

        eval_fn = make_noisy_circuit(p)   # same published eval path as freeze_eval
        y_train_pred = predict_original_scale(eval_fn, weights_p, X_train_scaled,
                                              scaler_y)
        train_r2 = compute_metrics(y_train, y_train_pred)["r2"]
        y_test_pred = predict_original_scale(eval_fn, weights_p, X_test_scaled,
                                             scaler_y)
        tm = compute_metrics(y_test, y_test_pred)
        rows.append(_row(seed, p, "noise_aware", tm, train_r2))
        print_flush(
            f"    p={p:<6}  test R2={tm['r2']:.4f}  train R2={train_r2:.4f}  "
            f"gap={train_r2 - tm['r2']:.4f}  train_t={train_time:.1f}s"
        )

    # === Save ===
    payload = {
        "seed": seed,
        "train_size": config.TRAIN_SIZE,
        "architecture": config.ENTANGLEMENT,
        "noise_aware_diff_method": config.NOISE_AWARE_DIFF_METHOD,
        "noise_free_train_r2": noise_free_train_r2,
        "noise_free_test_r2": noise_free_test_r2,
        "expected_noise_free_r2": config.EXPECTED_NOISE_FREE_R2,
        "levels": levels,
        "rows": rows,
    }
    out_dir = Path(config.RESULTS_DIR) / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if only_p is None else f"_p{only_p}"
    out_path = out_dir / f"noise_aware_metrics{suffix}.json"
    save_metrics(payload, str(out_path))
    print_flush(f"\nSaved: {out_path}")
    print_flush(f"Total wall time: {time.time() - start:.1f}s")
    print_flush(f"{'='*64}")
    print_flush(f"Seed {seed} done.")
    print_flush(f"{'='*64}")


