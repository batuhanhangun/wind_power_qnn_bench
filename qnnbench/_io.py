"""Shared utilities for the QNN wind power benchmarking pipeline."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add parent directory to path for imports
from qnnbench import config


def load_data(seed: int, train_size: int):
    """Load training and test data for a specific seed and train size.

    Args:
        seed: Random seed used for data split
        train_size: Number of training samples (800, 1600, 2400, or 3200)

    Returns:
        Tuple of (X_train, y_train, X_test, y_test) as pandas objects
    """
    data_dir = config.DATA_DIR / f"seed_{seed}"

    train_path = data_dir / f"train_{train_size}.csv"
    test_path = data_dir / "test_set.csv"

    train_df = pd.read_csv(train_path, delimiter=config.CSV_DELIMITER)
    test_df = pd.read_csv(test_path, delimiter=config.CSV_DELIMITER)

    X_train = train_df.drop(config.TARGET_COLUMN, axis=1)
    y_train = train_df[config.TARGET_COLUMN]
    X_test = test_df.drop(config.TARGET_COLUMN, axis=1)
    y_test = test_df[config.TARGET_COLUMN]

    return X_train, y_train, X_test, y_test


def compute_metrics(y_true, y_pred) -> dict:
    """Compute regression metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dictionary with keys: r2, rmse, mae
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def save_metrics(metrics_dict: dict, path: str):
    """Save metrics dictionary to JSON file.

    Args:
        metrics_dict: Dictionary of metrics to save
        path: Output file path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics_dict, f, indent=2)


def save_predictions(y_true, y_pred, path: str):
    """Save actual vs predicted values to CSV.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        path: Output file path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame({
        "Actual": np.array(y_true).flatten(),
        "Predicted": np.array(y_pred).flatten()
    })
    df.to_csv(path, index=False)


def save_convergence(loss_history: list, path: str):
    """Save loss history to CSV for convergence analysis.

    Args:
        loss_history: List of dicts with 'iteration' and 'loss' keys,
                     or list of dicts with 'epoch' and 'loss' keys
        path: Output file path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not loss_history:
        # Handle empty history
        df = pd.DataFrame(columns=["iteration", "loss"])
    else:
        # Check if it's epoch-based (ANN) or iteration-based (QNN)
        if "epoch" in loss_history[0]:
            df = pd.DataFrame(loss_history)
        else:
            df = pd.DataFrame(loss_history)

    df.to_csv(path, index=False)


def print_flush(msg: str):
    """Print message and flush stdout for buffering.

    Args:
        msg: Message to print
    """
    print(msg)
    sys.stdout.flush()
