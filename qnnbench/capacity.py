"""QNN capacity ablation — sweep ansatz repetitions at N=800 (noise-free).

For each (reps, seed): load that seed's train_800.csv + test_set.csv, MinMaxScale
X and y, train the reverse_linear QNN at the given repetition count on
lightning.qubit (adjoint diff, L-BFGS-B maxiter 50), and record train/test R2,
the generalization gap (train_r2 - test_r2), and test MSE/RMSE/MAE.

The entanglement topology (reverse_linear) and qubit width (4) are held fixed;
only ``reps`` varies, so parameter count = 4*(reps+1) is the sole capacity knob.

Writes one metrics JSON per (reps, seed) to results/run_<RUN_ID>/seed_<seed>/:
    exp_reps<reps>_metrics.json

Training is gate-for-gate identical to the deployed reverse_linear harness at
reps=3 (weight init np.random.RandomState(seed).uniform(-pi, pi, size=4*(reps+1)),
qml.grad(cost_fn, argnums=0)), so reps=3 reproduces the deployed N=800 result.

Usage:
    python -m scripts.run_capacity --seeds 42
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
    make_train_circuit,
    weight_shape,
    num_params,
    num_cnots,
    NUM_QUBITS,
)
from qnnbench._io import compute_metrics, load_data, print_flush, save_metrics


# --- Cost / training (reverse_linear circuit, argnums=0 gradient) ---

def make_cost(reps):
    """Build (circuit, shape, cost_fn) closures bound to a repetition count."""
    circuit = make_train_circuit(reps)
    shape = weight_shape(reps)

    def batch_predict(weights_flat, X):
        weights_2d = weights_flat.reshape(shape)
        return pnp.array([circuit(weights_2d, x) for x in X])

    def cost_fn(weights_flat, X, y):
        preds = batch_predict(weights_flat, X)
        return pnp.mean((preds - y) ** 2)

    return circuit, shape, cost_fn


def train_model(X_train_scaled, y_train_scaled, reps, seed, maxiter):
    """Train the reverse_linear QNN at ``reps`` with scipy L-BFGS-B (deterministic).

    Weight init: np.random.RandomState(seed).uniform(-pi, pi, size=4*(reps+1)).
    Gradient:    qml.grad(cost_fn, argnums=0).
    """
    circuit, shape, cost_fn = make_cost(reps)

    rng = np.random.RandomState(seed)
    init_weights = rng.uniform(-np.pi, np.pi, size=NUM_QUBITS * (reps + 1))

    X_pnp = pnp.array(X_train_scaled, requires_grad=False)
    y_pnp = pnp.array(y_train_scaled, requires_grad=False)

    grad_fn = qml.grad(cost_fn, argnums=0)

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
    )

    return result.x.reshape(shape), circuit


def predict_original_scale(circuit, weights, X_scaled, scaler_y):
    """Predict on scaled features and inverse-transform; clip wind power >= 0."""
    y_pred_scaled = np.array([float(circuit(weights, x)) for x in X_scaled])
    y_pred = scaler_y.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).ravel()
    return np.clip(y_pred, 0, None)


# --- Experiment runner ---

def run_experiment(reps, seed):
    """Run one (reps, seed) capacity cell at N=800 and write its metrics JSON."""
    train_size = config.TRAIN_SIZE
    np_ = num_params(reps)
    nc = num_cnots(reps)

    print_flush(f"\n{'='*60}")
    print_flush(f"Capacity ablation — reverse_linear QNN")
    print_flush(f"  reps={reps}  ({np_} params, {nc} CNOTs)")
    print_flush(f"  Train size: {train_size}  |  Seed: {seed}")
    print_flush(f"  Entanglement: {config.ENTANGLEMENT} (fixed), "
                f"maxiter={config.QNN_CONFIG['maxiter']}")
    print_flush(f"{'='*60}")

    start_time = time.time()

    X_train, y_train, X_test, y_test = load_data(seed, train_size)
    print_flush(f"Loaded data: {len(X_train)} train, {len(X_test)} test samples")

    # Scale: MinMaxScaler on X and y, fit on train, transform test.
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(
        np.array(y_train).reshape(-1, 1)
    ).ravel()
    X_test_scaled = scaler_X.transform(X_test)

    # Train (noise-free).
    train_start = time.time()
    weights, circuit = train_model(
        X_train_scaled, y_train_scaled,
        reps=reps, seed=seed,
        maxiter=config.QNN_CONFIG["maxiter"],
    )
    train_time = time.time() - train_start
    print_flush(f"Training completed in {train_time:.1f}s")

    y_train_pred = predict_original_scale(circuit, weights, X_train_scaled, scaler_y)
    train_metrics = compute_metrics(y_train, y_train_pred)
    print_flush(f"Train R2: {train_metrics['r2']:.4f}")

    y_test_pred = predict_original_scale(circuit, weights, X_test_scaled, scaler_y)
    test_metrics = compute_metrics(y_test, y_test_pred)
    print_flush(f"Test R2: {test_metrics['r2']:.4f}")

    gen_gap = train_metrics["r2"] - test_metrics["r2"]
    print_flush(f"Generalization gap (train_r2 - test_r2): {gen_gap:.4f}")

    wall_time = time.time() - start_time
    print_flush(f"Total wall time: {wall_time:.1f} seconds")

    # Output dir: results/run_<RUN_ID>/seed_<seed>/
    output_dir = Path(config.RESULTS_DIR) / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_dict = {
        "model": "qnn",
        "architecture": config.ENTANGLEMENT,
        "reps": reps,
        "num_params": np_,
        "num_cnots": nc,
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
        "entanglement": config.ENTANGLEMENT,
        "maxiter": config.QNN_CONFIG["maxiter"],
        "wall_time_seconds": wall_time,
        "training_time_seconds": train_time,
    }
    metrics_path = output_dir / f"exp_reps{reps}_metrics.json"
    save_metrics(metrics_dict, str(metrics_path))
    print_flush(f"Saved metrics: {metrics_path}")

    print_flush(f"\n{'='*60}")
    print_flush(f"Capacity cell (reps={reps}, seed={seed}) completed successfully!")
    print_flush(f"{'='*60}")


