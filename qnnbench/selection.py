"""Structure-selection runner: evaluate candidate QNN architectures per (config, seed).

For each (entanglement config, seed) on the data-scarce N=800 regime:
  * Phase 1 (decision basis): 5-fold cross-validation on the 800-sample training
    set, recording fold-train and fold-validation R2.
  * Phase 2 (transparency only): train once on the full train_800 and evaluate on
    the held-out 893-sample test set, recording test_r2 and test_gap.

The selection decision uses cross-validation ONLY. The test-set numbers are kept
for transparency and cross-environment reproduction checking and never enter the
decision (see ``src/aggregate_selection.py``).

Conventions are matched gate-for-gate and step-for-step to the reference pipeline
(``src/run_qnn.py``): MinMaxScaler on X and y, L-BFGS-B with maxiter from config,
weight init ``RandomState(seed).uniform(-pi, pi, size=16)``, per-fold init seed
``seed + fold_idx`` and final-train init seed ``seed``, predictions clipped to >= 0.

CLI:
    python -m scripts.run_selection --seeds 42
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

# Allow `import config` and `from src...` from the project root.
from qnnbench import config
from qnnbench._io import compute_metrics, load_data, print_flush, save_metrics
from qnnbench.config import CONFIGS
from qnnbench.circuits import NUM_PARAMS, WEIGHT_SHAPE, make_circuit

TRAIN_SIZE = 800
FEATURE_MAP = "ZFeatureMap"


# --- Training (mirrors src/run_qnn.py, generalized over the QNode) ---

def _make_cost_fn(circuit):
    """Build an MSE cost function bound to a given circuit QNode."""
    def batch_predict(weights_flat, X):
        weights_2d = weights_flat.reshape(WEIGHT_SHAPE)
        return pnp.array([circuit(weights_2d, x) for x in X])

    def cost_fn(weights_flat, X, y):
        preds = batch_predict(weights_flat, X)
        return pnp.mean((preds - y) ** 2)

    return cost_fn


def train_qnn(circuit, X_scaled, y_scaled, seed):
    """Train a QNN via scipy L-BFGS-B and return optimal weights (WEIGHT_SHAPE).

    Weight init is ``RandomState(seed).uniform(-pi, pi, size=16)``; the gradient
    uses ``qml.grad(cost_fn, argnums=0)`` (the non-deprecated form under 0.44).
    """
    rng = np.random.RandomState(seed)
    init_weights = rng.uniform(-np.pi, np.pi, size=NUM_PARAMS)
    assert init_weights.size == 16, "Each candidate must have exactly 16 parameters"

    cost_fn = _make_cost_fn(circuit)
    X_pnp = pnp.array(X_scaled, requires_grad=False)
    y_pnp = pnp.array(y_scaled, requires_grad=False)
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
        options={"maxiter": config.QNN_CONFIG["maxiter"]},
    )
    return result.x.reshape(WEIGHT_SHAPE)


def predict_original_scale(circuit, weights, X_scaled, scaler_y):
    """Predict on scaled features, inverse-transform to original scale, clip >= 0."""
    y_pred_scaled = np.array([float(circuit(weights, x)) for x in X_scaled])
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    return np.clip(y_pred, 0, None)  # Wind power cannot be negative


def _scale_fit(X_train, y_train):
    """Fit MinMaxScaler on X and y (train only) and return scaled arrays + scalers."""
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_train)
    y_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1, 1)).ravel()
    return X_scaled, y_scaled, scaler_X, scaler_y


# --- Per (config, seed) experiment ---

def run_config_seed(entanglement, seed):
    """Run the selection protocol for one (config, seed) and save a result JSON."""
    print_flush(f"\n{'='*60}")
    print_flush(f"Structure selection — config={entanglement}, seed={seed}")
    print_flush(f"{'='*60}")

    circuit = make_circuit(entanglement)
    assert NUM_PARAMS == 16, "Candidate must have exactly 16 parameters"

    start_time = time.time()
    X_train, y_train, X_test, y_test = load_data(seed, TRAIN_SIZE)
    print_flush(f"Loaded {len(X_train)} train / {len(X_test)} test samples")

    # === Phase 1: 5-fold cross-validation (the decision basis) ===
    print_flush(f"\nPhase 1: {config.CV_FOLDS}-fold cross-validation (decision basis)")
    kfold = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)
    fold_val_r2 = []
    fold_train_r2 = []
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

        # Scale on the fold-train only, then apply to the fold-validation.
        X_tr_s, y_tr_s, scaler_X, scaler_y = _scale_fit(X_tr, y_tr)
        X_va_s = scaler_X.transform(X_va)

        t0 = time.time()
        weights = train_qnn(circuit, X_tr_s, y_tr_s, seed=seed + fold_idx)
        y_tr_pred = predict_original_scale(circuit, weights, X_tr_s, scaler_y)
        y_va_pred = predict_original_scale(circuit, weights, X_va_s, scaler_y)
        r2_tr = compute_metrics(y_tr, y_tr_pred)["r2"]
        r2_va = compute_metrics(y_va, y_va_pred)["r2"]
        fold_train_r2.append(r2_tr)
        fold_val_r2.append(r2_va)
        print_flush(f"  fold {fold_idx}/{config.CV_FOLDS}: "
                    f"val R²={r2_va:.4f}, train R²={r2_tr:.4f} "
                    f"({time.time()-t0:.1f}s)")

    cv_val_r2 = float(np.mean(fold_val_r2))
    cv_train_r2 = float(np.mean(fold_train_r2))
    cv_gap = cv_train_r2 - cv_val_r2
    print_flush(f"CV: cv_val_r2={cv_val_r2:.4f}, cv_train_r2={cv_train_r2:.4f}, "
                f"cv_gap={cv_gap:.4f}")

    # === Phase 2: full-train + held-out test (transparency / reproduction only) ===
    print_flush("\nPhase 2: full train_800 -> held-out test "
                "(transparency only; NOT used for selection)")
    X_tr_s, y_tr_s, scaler_X, scaler_y = _scale_fit(X_train, y_train)
    X_te_s = scaler_X.transform(X_test)
    weights = train_qnn(circuit, X_tr_s, y_tr_s, seed=seed)
    y_tr_pred = predict_original_scale(circuit, weights, X_tr_s, scaler_y)
    y_te_pred = predict_original_scale(circuit, weights, X_te_s, scaler_y)
    train_r2 = compute_metrics(y_train, y_tr_pred)["r2"]
    test_r2 = compute_metrics(y_test, y_te_pred)["r2"]
    test_gap = train_r2 - test_r2
    print_flush(f"TEST: test_r2={test_r2:.4f}, train_r2={train_r2:.4f}, "
                f"test_gap={test_gap:.4f}")

    wall_time = time.time() - start_time
    result = {
        "config": entanglement,
        "feature_map": FEATURE_MAP,
        "entanglement": entanglement,
        "num_params": int(NUM_PARAMS),
        "seed": int(seed),
        "cv_val_r2": cv_val_r2,
        "cv_train_r2": cv_train_r2,
        "cv_gap": cv_gap,
        "test_r2": test_r2,
        "test_gap": test_gap,
        "train_r2": float(train_r2),
        "fold_val_r2": [float(v) for v in fold_val_r2],
        "fold_train_r2": [float(v) for v in fold_train_r2],
        "wall_time_seconds": wall_time,
    }

    out_dir = Path(config.RESULTS_DIR) / entanglement / f"seed_{seed}"
    out_path = out_dir / "selection_metrics.json"
    save_metrics(result, str(out_path))
    print_flush(f"Saved: {out_path}  (wall {wall_time:.1f}s)")
    return result


