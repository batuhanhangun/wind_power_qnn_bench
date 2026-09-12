"""Stage 1 — main benchmark (noise-free) on the reverse_linear QNN.

Gate-for-gate identical to the published main QNN harness EXCEPT the ansatz
entanglement, which is reverse_linear (imported from src.circuit). For each
(seed, train_size): 5-fold CV + final training on lightning.qubit with adjoint
diff, evaluation on the 893-sample test set, then write to
results/run_<RUN_ID>/seed_<seed>/:
    exp_<size>_metrics.json      test/train metrics + generalization gap (+ CV)
    exp_<size>_weights.npy       frozen trained weights, shape (4, 4)
    exp_<size>_predictions.csv   actual vs predicted on the test set
    exp_<size>_convergence.csv   L-BFGS-B loss history

The weights file is the handoff to Stage 2 (src/run_noise.py); because training
is deterministic (fixed seed, lightning.qubit + adjoint, L-BFGS-B), Stage 2 can
reload and re-derive these exact weights.

Usage:
    python -m scripts.run_qnn --seeds 42
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from qnnbench import config
from qnnbench.circuits import circuit, NUM_QUBITS, ANSATZ_REPS, WEIGHT_SHAPE
from qnnbench._io import (
    compute_metrics,
    load_data,
    print_flush,
    save_convergence,
    save_metrics,
    save_predictions,
)


# --- Cost / training (reverse_linear circuit, argnums=0 gradient) ---

def batch_predict(weights_flat, X):
    """Run the reverse_linear circuit on a batch of inputs."""
    weights_2d = weights_flat.reshape(WEIGHT_SHAPE)
    return pnp.array([circuit(weights_2d, x) for x in X])


def cost_fn(weights_flat, X, y):
    """MSE cost function for training."""
    preds = batch_predict(weights_flat, X)
    return pnp.mean((preds - y) ** 2)


def make_callback(loss_history, X_train, y_train):
    """scipy callback that records the loss at each L-BFGS-B iteration."""
    def callback(xk):
        loss_val = float(cost_fn(
            pnp.array(xk, requires_grad=False), X_train, y_train
        ))
        loss_history.append({
            "iteration": len(loss_history) + 1,
            "loss": loss_val,
        })

    return callback


def train_model(X_train_scaled, y_train_scaled, maxiter, seed,
                loss_history=None):
    """Train the reverse_linear QNN with scipy L-BFGS-B (deterministic).

    Weight init: np.random.RandomState(seed).uniform(-pi, pi, size=16).
    Gradient:    qml.grad(cost_fn, argnums=0).
    """
    rng = np.random.RandomState(seed)
    init_weights = rng.uniform(-np.pi, np.pi, size=NUM_QUBITS * (ANSATZ_REPS + 1))

    X_pnp = pnp.array(X_train_scaled, requires_grad=False)
    y_pnp = pnp.array(y_train_scaled, requires_grad=False)

    grad_fn = qml.grad(cost_fn, argnums=0)

    cb = None
    if loss_history is not None:
        cb = make_callback(loss_history, X_pnp, y_pnp)

    result = minimize(
        fun=lambda w: float(cost_fn(
            pnp.array(w, requires_grad=True), X_pnp, y_pnp
        )),
        x0=init_weights,
        method="L-BFGS-B",
        jac=lambda w: np.array(grad_fn(
            pnp.array(w, requires_grad=True), X_pnp, y_pnp
        )),
        options={"maxiter": maxiter},
        callback=cb,
    )

    return result.x.reshape(WEIGHT_SHAPE)


def predict_original_scale(weights, X_scaled, scaler_y):
    """Predict on scaled features and inverse-transform; clip wind power >= 0."""
    y_pred_scaled = np.array([float(circuit(weights, x)) for x in X_scaled])
    y_pred = scaler_y.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).ravel()
    return np.clip(y_pred, 0, None)


# --- Scaling / CV ---

def scale_data(X_train, y_train, X_val, y_val):
    """MinMaxScaler on X and y; fit on train, transform val."""
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(
        np.array(y_train).reshape(-1, 1)
    ).ravel()

    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(
        np.array(y_val).reshape(-1, 1)
    ).ravel()

    return (X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
            scaler_X, scaler_y)


def run_cross_validation(X_train, y_train, seed):
    """5-fold CV; returns mean/std of r2, rmse, mae (per-fold init seed+fold)."""
    kfold = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)

    r2_scores, rmse_scores, mae_scores = [], [], []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
        print_flush(f"  CV Fold {fold_idx}/{config.CV_FOLDS}...")

        X_train_fold = X_train.iloc[train_idx]
        y_train_fold = y_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_val_fold = y_train.iloc[val_idx]

        X_tr_sc, y_tr_sc, X_val_sc, _, scaler_X, scaler_y = scale_data(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold
        )

        fold_start = time.time()
        weights = train_model(X_tr_sc, y_tr_sc,
                              maxiter=config.QNN_CONFIG["maxiter"],
                              seed=seed + fold_idx)
        fold_time = time.time() - fold_start

        y_val_pred = predict_original_scale(weights, X_val_sc, scaler_y)
        metrics = compute_metrics(y_val_fold, y_val_pred)

        r2_scores.append(metrics["r2"])
        rmse_scores.append(metrics["rmse"])
        mae_scores.append(metrics["mae"])

        print_flush(f"    R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.2f}, "
                    f"Time={fold_time:.1f}s")

    return {
        "cv_mean_r2": float(np.mean(r2_scores)),
        "cv_std_r2": float(np.std(r2_scores)),
        "cv_mean_rmse": float(np.mean(rmse_scores)),
        "cv_std_rmse": float(np.std(rmse_scores)),
        "cv_mean_mae": float(np.mean(mae_scores)),
        "cv_std_mae": float(np.std(mae_scores)),
    }


# --- Experiment runner ---

def run_experiment(train_size, seed):
    """Run one Stage 1 (seed, train_size) experiment and write all outputs."""
    print_flush(f"\n{'='*60}")
    print_flush(f"Stage 1 (main, noise-free) — reverse_linear QNN")
    print_flush(f"  Train size: {train_size}")
    print_flush(f"  Seed: {seed}")
    print_flush(f"  Entanglement: {config.ENTANGLEMENT} "
                f"({config.CNOT_COUNT} CNOTs), maxiter="
                f"{config.QNN_CONFIG['maxiter']}")
    print_flush(f"{'='*60}")

    start_time = time.time()

    X_train, y_train, X_test, y_test = load_data(seed, train_size)
    print_flush(f"Loaded data: {len(X_train)} train, {len(X_test)} test samples")

    # === Phase 1: Cross-validation ===
    print_flush(f"\nPhase 1: {config.CV_FOLDS}-fold Cross-Validation...")
    cv_metrics = run_cross_validation(X_train, y_train, seed)
    print_flush(f"CV Results: R2={cv_metrics['cv_mean_r2']:.4f} "
                f"+/- {cv_metrics['cv_std_r2']:.4f}")

    # === Phase 2: Final training with convergence tracking ===
    print_flush("\nPhase 2: Final training on full dataset...")

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(
        np.array(y_train).reshape(-1, 1)
    ).ravel()
    X_test_scaled = scaler_X.transform(X_test)

    loss_history = []
    train_start = time.time()
    optimal_weights = train_model(
        X_train_scaled, y_train_scaled,
        maxiter=config.QNN_CONFIG["maxiter"],
        seed=seed,
        loss_history=loss_history,
    )
    train_time = time.time() - train_start
    print_flush(f"Training completed in {train_time:.1f}s")

    y_train_pred = predict_original_scale(optimal_weights, X_train_scaled, scaler_y)
    train_metrics = compute_metrics(y_train, y_train_pred)
    print_flush(f"Train R2: {train_metrics['r2']:.4f}")

    y_test_pred = predict_original_scale(optimal_weights, X_test_scaled, scaler_y)
    test_metrics = compute_metrics(y_test, y_test_pred)
    print_flush(f"Test R2: {test_metrics['r2']:.4f}")

    gen_gap = train_metrics["r2"] - test_metrics["r2"]
    print_flush(f"Generalization gap (train_r2 - test_r2): {gen_gap:.4f}")

    wall_time = time.time() - start_time
    print_flush(f"Total wall time: {wall_time:.1f} seconds")

    # Output dir: results/run_<RUN_ID>/seed_<seed>/
    output_dir = Path(config.RESULTS_DIR) / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Metrics JSON
    metrics_dict = {
        "model": "qnn",
        "architecture": config.ENTANGLEMENT,
        "train_size": train_size,
        "seed": seed,
        "train_r2": train_metrics["r2"],
        "train_mse": train_metrics["mse"],
        "train_rmse": train_metrics["rmse"],
        "train_mae": train_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "test_mse": test_metrics["mse"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "generalization_gap": gen_gap,
        **cv_metrics,
        "config": config.QNN_CONFIG,
        "cnot_count": config.CNOT_COUNT,
        "wall_time_seconds": wall_time,
        "training_time_seconds": train_time,
    }
    metrics_path = output_dir / f"exp_{train_size}_metrics.json"
    save_metrics(metrics_dict, str(metrics_path))
    print_flush(f"Saved metrics: {metrics_path}")

    # Frozen weights (handoff to Stage 2), shape (4, 4)
    weights_path = output_dir / f"exp_{train_size}_weights.npy"
    np.save(str(weights_path), np.asarray(optimal_weights))
    print_flush(f"Saved weights: {weights_path}")

    # Predictions on ORIGINAL scale
    predictions_path = output_dir / f"exp_{train_size}_predictions.csv"
    save_predictions(y_test, y_test_pred, str(predictions_path))
    print_flush(f"Saved predictions: {predictions_path}")

    # Convergence history
    if loss_history:
        convergence_path = output_dir / f"exp_{train_size}_convergence.csv"
        save_convergence(loss_history, str(convergence_path))
        print_flush(f"Saved convergence: {convergence_path} "
                    f"({len(loss_history)} points)")
    else:
        print_flush("WARNING: No convergence data captured from optimizer")

    print_flush(f"\n{'='*60}")
    print_flush(f"Stage 1 experiment completed successfully!")
    print_flush(f"{'='*60}")


