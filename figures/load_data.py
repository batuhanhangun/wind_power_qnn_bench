"""Loaders for all figure inputs, with hard row-count assertions."""
from pathlib import Path

import pandas as pd

from qnnbench.config import REFERENCE_DIR as DATA, SEEDS, TRAIN_SIZES as SIZES


def _report(name, df):
    print(f"{name}: shape={df.shape} columns={list(df.columns)}")


def load_qnn():
    df = pd.read_csv(DATA / "qnn" / "qnn_per_seed.csv")
    _report("qnn_per_seed", df)
    expected = len(SEEDS) * len(SIZES)
    assert len(df) == expected, f"qnn_per_seed expected {expected} rows, got {len(df)}"
    df["model"] = "QNN"
    return df


def load_classical():
    df = pd.read_csv(DATA / "classical" / "classical_per_seed.csv")
    _report("classical_per_seed", df)
    expected = 5 * len(SEEDS) * len(SIZES)
    assert len(df) == expected, f"classical_per_seed expected {expected} rows, got {len(df)}"
    assert set(df["model"]) == {"ANN", "SVR", "DTR", "RF", "XGBoost"}, set(df["model"])
    return df


def load_ann_reg():
    df = pd.read_csv(DATA / "ann_reg" / "ann_reg_classical_per_seed.csv")
    _report("ann_reg_classical_per_seed", df)
    expected = len(SEEDS) * len(SIZES)
    assert len(df) == expected, f"ann_reg expected {expected} rows, got {len(df)}"
    df["model"] = "ANN-Reg"
    return df


def load_all_per_seed():
    """All seven models, unified schema, display labels applied."""
    df = pd.concat([load_qnn(), load_classical(), load_ann_reg()], ignore_index=True)
    return df[["model", "seed", "train_size", "test_r2", "train_r2", "generalization_gap"]]


def load_noise_summary():
    df = pd.read_csv(DATA / "noise" / "noise_summary.csv")
    _report("noise_summary", df)
    assert len(df) == 6, f"noise_summary expected 6 rows, got {len(df)}"
    return df.sort_values("noise_p").reset_index(drop=True)


# Prediction folders for Fig 4 (N=3200). Header verified: "Actual,Predicted".
_CLASSICAL_RUN = DATA / "predictions"
PRED_DIRS = {
    "QNN": DATA / "predictions" / "qnn",
    "ANN": _CLASSICAL_RUN / "ann",
    "ANN-Reg": DATA / "predictions" / "ann_reg",
    "SVR": _CLASSICAL_RUN / "svr",
    "DTR": _CLASSICAL_RUN / "dtr",
    "XGBoost": _CLASSICAL_RUN / "xgboost",
    "RF": _CLASSICAL_RUN / "rf",
}


def load_predictions_3200():
    """Pooled Actual/Predicted at N=3200 per model. Returns (dict, missing list)."""
    pooled, missing = {}, []
    for model, base in PRED_DIRS.items():
        frames = []
        for seed in SEEDS:
            f = base / f"seed_{seed}" / "exp_3200_predictions.csv"
            if not f.exists():
                missing.append(str(f))
                frames = []
                break
            d = pd.read_csv(f)
            assert {"Actual", "Predicted"} <= set(d.columns), f"{f}: {list(d.columns)}"
            frames.append(d)
        if frames:
            pooled[model] = pd.concat(frames, ignore_index=True)
            print(f"predictions {model}: {len(pooled[model])} pooled points")
    return pooled, missing
