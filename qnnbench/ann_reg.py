"""Runner for the production-grade, regularized ANN baseline (rerun).

Re-trains the regularized multilayer perceptron (sklearn MLPRegressor) tuned
with GridSearchCV, using the exact same pipeline, scaling, CV folds, seeds,
and splits as the original ANN_reg run. In addition to the metrics JSON, this
rerun saves per-run prediction files (Actual,Predicted — one row per test
sample) so the figure loader can read them.

Determinism: GridSearchCV runs with n_jobs=1 and the launcher (run_local.py /
deterministic and bit-identical to the original run.

Usage:
    python -m scripts.run_ann_reg --seeds 42
"""

import os
# Preserved from the verified run_local.py, before numpy/scikit-learn imports.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from qnnbench import config
from qnnbench._io import (
    compute_metrics,
    load_data,
    print_flush,
    save_metrics,
    save_predictions,
)


def build_estimator(seed: int) -> MLPRegressor:
    """Create the regularized MLPRegressor with fixed (non-grid) settings.

    Args:
        seed: Random seed for reproducible weight initialization and the
              internal early-stopping validation split.

    Returns:
        Unfitted MLPRegressor instance.
    """
    cfg = config.ANN_REG_CONFIG
    return MLPRegressor(
        solver=cfg["solver"],
        activation=cfg["activation"],
        early_stopping=cfg["early_stopping"],
        validation_fraction=cfg["validation_fraction"],
        n_iter_no_change=cfg["n_iter_no_change"],
        max_iter=cfg["max_iter"],
        random_state=seed,
    )


def count_parameters(mlp: MLPRegressor) -> int:
    """Trainable parameter count of a fitted MLP.

    Sum of the sizes of all weight matrices (coefs_) and bias vectors
    (intercepts_).

    Args:
        mlp: Fitted MLPRegressor.

    Returns:
        Total number of trainable parameters.
    """
    n_weights = sum(int(w.size) for w in mlp.coefs_)
    n_biases = sum(int(b.size) for b in mlp.intercepts_)
    return n_weights + n_biases


def run_experiment(train_size: int, seed: int) -> dict:
    """Run a single regularized-ANN experiment for one (seed, train_size).

    Args:
        train_size: Number of training samples (800, 1600, 2400, 3200).
        seed: Random seed.

    Returns:
        Dict of metrics (also saved to disk as JSON).
    """
    print_flush(f"\n{'='*60}")
    print_flush(f"Running {config.MODEL_NAME} experiment")
    print_flush(f"  Train size: {train_size}")
    print_flush(f"  Seed: {seed}")
    print_flush(f"{'='*60}")

    start_time = time.time()

    # Load data (same splits as the main benchmark)
    X_train, y_train, X_test, y_test = load_data(seed, train_size)
    print_flush(f"Loaded data: {len(X_train)} train, {len(X_test)} test samples")

    # Pipeline: scaling inside the pipeline prevents CV leakage
    pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", build_estimator(seed)),
    ])

    # Cross-validation: identical to the classical harness
    kfold = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)

    n_combos = (
        len(config.ANN_REG_CONFIG["param_grid"]["model__hidden_layer_sizes"])
        * len(config.ANN_REG_CONFIG["param_grid"]["model__alpha"])
        * len(config.ANN_REG_CONFIG["param_grid"]["model__learning_rate_init"])
    )
    print_flush(
        f"Starting GridSearchCV: {n_combos} combinations x "
        f"{config.CV_FOLDS}-fold CV (scoring=R2)..."
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=config.ANN_REG_CONFIG["param_grid"],
        cv=kfold,
        scoring="r2",
        refit=True,
        n_jobs=1,  # single process for platform-deterministic reproduction
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_  # already refit on full train set
    best_params = grid_search.best_params_
    best_index = grid_search.best_index_
    cv_mean_r2 = float(grid_search.cv_results_["mean_test_score"][best_index])
    cv_std_r2 = float(grid_search.cv_results_["std_test_score"][best_index])

    print_flush(f"\nBest parameters: {best_params}")
    print_flush(f"CV R²: {cv_mean_r2:.4f} ± {cv_std_r2:.4f}")

    # Trainable parameter count of the selected network
    n_params = count_parameters(best_model.named_steps["model"])
    print_flush(f"Trainable parameters: {n_params}")

    # Evaluate on training set (clip: wind power cannot be negative)
    y_train_pred = np.clip(best_model.predict(X_train), 0, None)
    train_metrics = compute_metrics(y_train, y_train_pred)
    print_flush(f"Train R²: {train_metrics['r2']:.4f}")

    # Evaluate on the 893-sample held-out test set
    y_test_pred = np.clip(best_model.predict(X_test), 0, None)
    test_metrics = compute_metrics(y_test, y_test_pred)
    print_flush(f"Test R²: {test_metrics['r2']:.4f}")

    generalization_gap = train_metrics["r2"] - test_metrics["r2"]
    print_flush(f"Generalization gap (train - test R²): {generalization_gap:.4f}")

    wall_time = time.time() - start_time
    print_flush(f"Total wall time: {wall_time:.1f} seconds")

    # Stringify best_params for clean CSV serialization (tuples -> str)
    best_params_str = {k: str(v) for k, v in best_params.items()}

    metrics_dict = {
        "model": config.MODEL_NAME,
        "seed": seed,
        "train_size": train_size,
        "test_r2": test_metrics["r2"],
        "train_r2": train_metrics["r2"],
        "generalization_gap": generalization_gap,
        "test_mse": test_metrics["mse"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "n_params": n_params,
        "best_params": best_params_str,
        "cv_mean_r2": cv_mean_r2,
        "cv_std_r2": cv_std_r2,
        "wall_time_seconds": wall_time,
    }

    output_dir = Path(config.RESULTS_DIR) / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"exp_{train_size}_metrics.json"
    save_metrics(metrics_dict, str(metrics_path))
    print_flush(f"Saved metrics: {metrics_path}")

    # Per-run predictions (the reason for this rerun): Actual,Predicted header,
    # one row per test sample, matching the classical harness's files.
    predictions_path = output_dir / f"exp_{train_size}_predictions.csv"
    save_predictions(y_test, y_test_pred, str(predictions_path))
    print_flush(f"Saved predictions: {predictions_path}")

    print_flush(f"\n{'='*60}")
    print_flush("Experiment completed successfully!")
    print_flush(f"{'='*60}")

    return metrics_dict


