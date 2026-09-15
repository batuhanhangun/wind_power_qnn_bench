"""Published constants and grids copied from the verified source projects."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATASET_PATH = DATA_DIR / "raw" / "total_dataset.xlsx"
REFERENCE_DIR = ROOT / "results" / "reference"
PUBLISHED_TABLES = REFERENCE_DIR / "published_tables.json"
RUN_ID = "local"
RESULTS_DIR = ROOT / "results" / "runs" / "qnn"
AGGREGATED_DIR = RESULTS_DIR
TRAIN_SIZES = [800, 1600, 2400, 3200]
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
TEST_SIZE = 893
TARGET_COLUMN = "Power"
CSV_DELIMITER = ";"
ANN_CONFIG = {
    "hidden_layer_sizes": (3,),  # 4*3+3 + 3*1+1 = 19 parameters
    "max_iter": 1000,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "param_grid": {
        "model__activation": ["relu", "tanh"],
        "model__alpha": [0.0001, 0.001, 0.01],
        "model__learning_rate_init": [0.001, 0.01],
        "model__solver": ["adam"],
    },
}
SVR_PARAM_GRID = {
    "model__kernel": ["rbf", "linear"],
    "model__C": [0.1, 1, 10, 100],
    "model__gamma": ["scale", "auto"],
}
RF_PARAM_GRID = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
}
XGBOOST_PARAM_GRID = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [3, 5],
}
DTR_PARAM_GRID = {
    "model__max_depth": [None, 5, 10, 20],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
}
CV_FOLDS = 5
CLASSICAL_MODELS = ["ann", "svr", "rf", "dtr", "xgboost"]
ALL_MODELS = ["qnn"] + CLASSICAL_MODELS
ENTANGLEMENT = "reverse_linear"
ENTANGLEMENT_PAIRS = [(2, 3), (1, 2), (0, 1)]
CNOT_COUNT = len(ENTANGLEMENT_PAIRS) * 3
NOISE_LEVELS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
TRAIN_SIZE = 800
REPS_GRID = [1, 2, 3, 4, 5]
DEPLOYED_REPS = 3
EXPECTED_N800_TEST_R2 = 0.903
NOISE_AWARE_LEVELS = [0.001, 0.005, 0.01, 0.02]
ENVELOPE_MAX_P = 0.005
EDGE_P = 0.01
STRESS_P = 0.02
NOISE_AWARE_DIFF_METHOD = "backprop"
EXPECTED_NOISE_FREE_R2 = 0.903
EXPECTED_FREEZE_EVAL_R2 = {
    0.001: 0.893,
    0.005: 0.814,
    0.01: 0.659,
    0.02: 0.299,
}
FIDELITY_TOL = 0.01
ANN_REG_CONFIG = {
    "solver": "adam",
    "activation": "relu",
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 10,
    "max_iter": 500,
    "param_grid": {
        "model__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
        "model__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
        "model__learning_rate_init": [1e-3, 1e-2],
    },
}
ANN_REG_SELECTION_CHOICES = ["r2", "neg_rmse"]
ANN_REG_SELECTION_DEFAULT = "neg_rmse"
# Model-selection criterion for the ANN-Reg grid search. "neg_rmse" is the
# default and matches the classical harness (multi-metric scoring dict with
# refit="neg_rmse"); "r2" reproduces the earlier published ANN-Reg run.
ANN_REG_SELECTION = ANN_REG_SELECTION_DEFAULT
CONSTRAINED_ANN_SIGMA = 0.294
SEED_COLLAPSE_THRESHOLD = 0.25
REPRODUCTION_ATOL = 1e-6
REPRODUCTION_ATOL_LOOSE = 1e-3
MODEL_NAME = "ANN_reg"
QNN_CONFIG = {
    "framework": "pennylane",
    "feature_map": "ZFeatureMap",       # H + RZ(2x) per qubit
    "feature_map_reps": 1,
    "ansatz": "RealAmplitudes",         # RY + CNOT entanglement
    "ansatz_reps": 3,
    "entanglement": ENTANGLEMENT,       # reverse_linear
    "optimizer": "L_BFGS_B",
    "maxiter": 50,
    "num_qubits": 4,
    "num_parameters": 16,               # 4 * (3+1) = 16
    "cnot_count": CNOT_COUNT,           # 9 (vs circular's 12)
    "device_train": "lightning.qubit",  # noise-free training + p=0 eval
    "device_eval": "default.mixed",     # noisy (density-matrix) evaluation
    "diff_method": "adjoint",
}
QNN_CONFIG["device"] = QNN_CONFIG["device_train"]
ARCHITECTURE_PAIRS = {
    "circular": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "full": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
    "reverse_linear": [(2, 3), (1, 2), (0, 1)],
}
CONFIGS = ["circular", "full", "reverse_linear"]
REFERENCE_CSV = REFERENCE_DIR / "ann_reg" / "ann_reg_classical_per_seed.csv"
