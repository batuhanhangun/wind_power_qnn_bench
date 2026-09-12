"""Unified runner for all classical machine learning models.

Supports: ANN, SVR, Random Forest, Decision Tree, XGBoost

Usage:
    python -m scripts.run_classical --models rf --sizes 800 --seeds 42
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Add parent directory to path for imports
from qnnbench import config
from qnnbench._io import (
    compute_metrics,
    load_data,
    print_flush,
    save_convergence,
    save_metrics,
    save_predictions,
)


def get_model_and_param_grid(model_name: str, seed: int):
    """Get model instance and parameter grid based on model name.

    Args:
        model_name: One of 'ann', 'svr', 'rf', 'dtr', 'xgboost'
        seed: Random seed for reproducibility

    Returns:
        Tuple of (model_instance, param_grid)
    """
    if model_name == "ann":
        model = MLPRegressor(
            hidden_layer_sizes=config.ANN_CONFIG["hidden_layer_sizes"],
            random_state=seed,
            max_iter=config.ANN_CONFIG["max_iter"],
            early_stopping=config.ANN_CONFIG["early_stopping"],
            validation_fraction=config.ANN_CONFIG["validation_fraction"],
        )
        param_grid = config.ANN_CONFIG["param_grid"]

    elif model_name == "svr":
        model = SVR()
        param_grid = config.SVR_PARAM_GRID

    elif model_name == "rf":
        model = RandomForestRegressor(random_state=seed)
        param_grid = config.RF_PARAM_GRID

    elif model_name == "dtr":
        model = DecisionTreeRegressor(random_state=seed)
        param_grid = config.DTR_PARAM_GRID

    elif model_name == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Run: pip install xgboost")
        model = XGBRegressor(random_state=seed, objective="reg:squarederror")
        param_grid = config.XGBOOST_PARAM_GRID

    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from {config.CLASSICAL_MODELS}")

    return model, param_grid


def run_experiment(model_name: str, train_size: int, seed: int):
    """Run a single experiment for a classical model.

    Args:
        model_name: Name of the model to run
        train_size: Number of training samples
        seed: Random seed
    """
    print_flush(f"\n{'='*60}")
    print_flush(f"Running {model_name.upper()} experiment")
    print_flush(f"  Train size: {train_size}")
    print_flush(f"  Seed: {seed}")
    print_flush(f"{'='*60}")

    start_time = time.time()

    # Load data
    X_train, y_train, X_test, y_test = load_data(seed, train_size)
    print_flush(f"Loaded data: {len(X_train)} train, {len(X_test)} test samples")

    # Get model and param grid
    model, param_grid = get_model_and_param_grid(model_name, seed)

    # Build pipeline
    pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", model),
    ])

    # Set up cross-validation
    kfold = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)

    # Set up grid search
    scoring = {
        "neg_rmse": "neg_root_mean_squared_error",
        "neg_mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    print_flush(f"Starting GridSearchCV with {config.CV_FOLDS}-fold CV...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=kfold,
        scoring=scoring,
        refit="neg_rmse",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_index = grid_search.best_index_

    # Extract CV metrics for best configuration
    cv_mean_r2 = grid_search.cv_results_["mean_test_r2"][best_index]
    cv_std_r2 = grid_search.cv_results_["std_test_r2"][best_index]
    cv_mean_rmse = -grid_search.cv_results_["mean_test_neg_rmse"][best_index]
    cv_std_rmse = grid_search.cv_results_["std_test_neg_rmse"][best_index]
    cv_mean_mae = -grid_search.cv_results_["mean_test_neg_mae"][best_index]
    cv_std_mae = grid_search.cv_results_["std_test_neg_mae"][best_index]

    print_flush(f"\nBest parameters: {best_params}")
    print_flush(f"CV R²: {cv_mean_r2:.4f} ± {cv_std_r2:.4f}")

    # Evaluate on training set
    y_train_pred = best_model.predict(X_train)
    y_train_pred = np.clip(y_train_pred, 0, None)  # Wind power cannot be negative
    train_metrics = compute_metrics(y_train, y_train_pred)
    print_flush(f"Train R²: {train_metrics['r2']:.4f}")

    # Evaluate on test set
    y_test_pred = best_model.predict(X_test)
    y_test_pred = np.clip(y_test_pred, 0, None)  # Wind power cannot be negative
    test_metrics = compute_metrics(y_test, y_test_pred)
    print_flush(f"Test R²: {test_metrics['r2']:.4f}")

    wall_time = time.time() - start_time
    print_flush(f"Total wall time: {wall_time:.1f} seconds")

    # Prepare output directory
    output_dir = Path(config.RESULTS_DIR) / model_name / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_dict = {
        "model": model_name,
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
        "cv_mean_r2": float(cv_mean_r2),
        "cv_std_r2": float(cv_std_r2),
        "cv_mean_rmse": float(cv_mean_rmse),
        "cv_std_rmse": float(cv_std_rmse),
        "cv_mean_mae": float(cv_mean_mae),
        "cv_std_mae": float(cv_std_mae),
        "best_params": best_params,
        "wall_time_seconds": wall_time,
    }

    metrics_path = output_dir / f"exp_{train_size}_metrics.json"
    save_metrics(metrics_dict, str(metrics_path))
    print_flush(f"Saved metrics: {metrics_path}")

    # Save predictions
    predictions_path = output_dir / f"exp_{train_size}_predictions.csv"
    save_predictions(y_test, y_test_pred, str(predictions_path))
    print_flush(f"Saved predictions: {predictions_path}")

    # Save convergence data for ANN only
    if model_name == "ann":
        try:
            mlp = best_model.named_steps["model"]
            if hasattr(mlp, "loss_curve_") and mlp.loss_curve_:
                loss_history = [
                    {"epoch": i + 1, "loss": loss}
                    for i, loss in enumerate(mlp.loss_curve_)
                ]
                convergence_path = output_dir / f"exp_{train_size}_convergence.csv"
                save_convergence(loss_history, str(convergence_path))
                print_flush(f"Saved convergence: {convergence_path}")
            else:
                print_flush("WARNING: No loss curve available for ANN")
        except Exception as e:
            print_flush(f"WARNING: Could not extract ANN loss curve: {e}")

    print_flush(f"\n{'='*60}")
    print_flush(f"Experiment completed successfully!")
    print_flush(f"{'='*60}")


