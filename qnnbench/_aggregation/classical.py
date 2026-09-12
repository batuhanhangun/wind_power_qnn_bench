"""Multi-seed result aggregation and statistical tests.

- Aggregate mean ± std across seeds
- Run Wilcoxon signed-rank tests for statistical significance

Aggregation is called automatically by the corresponding scripts.run_* entry point.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Add parent directory to path for imports
from qnnbench import config


def collect_results():
    """Collect all results from the results directory.

    Returns:
        List of dictionaries containing results from all experiments
    """
    results = []
    results_dir = Path(config.RESULTS_DIR)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return results

    for model in config.ALL_MODELS:
        model_dir = results_dir / model

        if not model_dir.exists():
            continue

        for seed_dir in model_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue

            seed = int(seed_dir.name.split("_")[1])

            for metrics_file in seed_dir.glob("exp_*_metrics.json"):
                try:
                    train_size = int(metrics_file.stem.split("_")[1])

                    with open(metrics_file, "r") as f:
                        metrics = json.load(f)

                    results.append({
                        "model": model,
                        "seed": seed,
                        "train_size": train_size,
                        "test_r2": metrics.get("test_r2"),
                        "test_mse": metrics.get("test_mse"),
                        "test_rmse": metrics.get("test_rmse"),
                        "test_mae": metrics.get("test_mae"),
                        "train_r2": metrics.get("train_r2"),
                        "train_mse": metrics.get("train_mse"),
                        "train_rmse": metrics.get("train_rmse"),
                        "train_mae": metrics.get("train_mae"),
                        "cv_mean_r2": metrics.get("cv_mean_r2"),
                        "cv_std_r2": metrics.get("cv_std_r2"),
                    })
                except Exception as e:
                    print(f"Warning: Could not parse {metrics_file}: {e}")

    return results


def _ci_bounds(values, alpha=0.05):
    """Return (t-based CI half-width, normal CI half-width) for a 1-D array.

    Args:
        values: pandas Series or numpy array of per-seed metric values
        alpha: Significance level (default 0.05 → 95 % CI)

    Returns:
        Tuple (t_hw, norm_hw) — half-widths; add/subtract from the mean to
        obtain lower/upper bounds.
    """
    n = len(values)
    s = np.std(values, ddof=1)          # sample std
    se = s / np.sqrt(n)                 # standard error
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    norm_crit = stats.norm.ppf(1 - alpha / 2)   # ≈ 1.96 for alpha=0.05
    return float(t_crit * se), float(norm_crit * se)


def aggregate_by_model_and_size(results):
    """Aggregate results by model and train_size across seeds.

    Computes mean, std, t-based 95 % CI, and normal-approximation 95 % CI
    for every metric.

    Args:
        results: List of result dictionaries

    Returns:
        DataFrame with aggregated statistics
    """
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Metrics to aggregate (test + train variants)
    metrics = ["r2", "mse", "rmse", "mae"]

    grouped = df.groupby(["model", "train_size"])

    aggregated = []
    for (model, train_size), group in grouped:
        n_seeds = len(group)
        seeds_used = sorted(group["seed"].tolist())

        agg_row = {
            "model": model,
            "train_size": train_size,
            "n_seeds": n_seeds,
            "seeds": str(seeds_used),
        }

        for split in ("test", "train"):
            for m in metrics:
                col = f"{split}_{m}"
                if col not in group.columns:
                    continue
                vals = group[col].dropna()
                mean_ = float(vals.mean())
                std_ = float(vals.std(ddof=1))
                t_hw, norm_hw = _ci_bounds(vals)
                agg_row[f"mean_{col}"] = mean_
                agg_row[f"std_{col}"] = std_
                agg_row[f"ci95t_lo_{col}"] = mean_ - t_hw
                agg_row[f"ci95t_hi_{col}"] = mean_ + t_hw
                agg_row[f"ci95n_lo_{col}"] = mean_ - norm_hw
                agg_row[f"ci95n_hi_{col}"] = mean_ + norm_hw

        # Generalization gap (train_r2 − test_r2)
        if "train_r2" in group.columns and "test_r2" in group.columns:
            gaps = group["train_r2"] - group["test_r2"]
            gap_mean = float(gaps.mean())
            gap_std = float(gaps.std(ddof=1))
            gap_t_hw, gap_norm_hw = _ci_bounds(gaps)
            agg_row["mean_gap_r2"] = gap_mean
            agg_row["std_gap_r2"] = gap_std
            agg_row["ci95t_lo_gap_r2"] = gap_mean - gap_t_hw
            agg_row["ci95t_hi_gap_r2"] = gap_mean + gap_t_hw
            agg_row["ci95n_lo_gap_r2"] = gap_mean - gap_norm_hw
            agg_row["ci95n_hi_gap_r2"] = gap_mean + gap_norm_hw

        aggregated.append(agg_row)

    return pd.DataFrame(aggregated)


def run_wilcoxon_tests(results, train_sizes=None):
    """Run Wilcoxon signed-rank tests comparing QNN to other models.

    Runs tests at all specified training sizes for completeness.

    Args:
        results: List of result dictionaries
        train_sizes: List of training sizes to test (default: all from config)

    Returns:
        List of test result dictionaries
    """
    if train_sizes is None:
        train_sizes = config.TRAIN_SIZES

    df = pd.DataFrame(results)
    test_results = []

    comparison_models = ["ann", "rf", "xgboost", "svr", "dtr"]

    for train_size in train_sizes:
        df_size = df[df["train_size"] == train_size]

        # Get QNN results
        qnn_results = df_size[df_size["model"] == "qnn"].set_index("seed")["test_r2"]

        if len(qnn_results) < 5:
            print(f"Warning: Only {len(qnn_results)} QNN results at train_size={train_size}")

        for other_model in comparison_models:
            other_results = df_size[df_size["model"] == other_model].set_index("seed")["test_r2"]

            # Find common seeds
            common_seeds = qnn_results.index.intersection(other_results.index)

            if len(common_seeds) < 5:
                test_results.append({
                    "train_size": train_size,
                    "comparison": f"QNN vs {other_model.upper()}",
                    "n_pairs": len(common_seeds),
                    "statistic": None,
                    "p_value": None,
                    "note": f"Insufficient paired samples ({len(common_seeds)} < 5)",
                })
                continue

            # Extract paired values
            qnn_vals = qnn_results.loc[common_seeds].values
            other_vals = other_results.loc[common_seeds].values

            # Run Wilcoxon test (two-sided)
            try:
                statistic, p_value = stats.wilcoxon(qnn_vals, other_vals, alternative="two-sided")
                test_results.append({
                    "train_size": train_size,
                    "comparison": f"QNN vs {other_model.upper()}",
                    "n_pairs": len(common_seeds),
                    "qnn_mean": float(np.mean(qnn_vals)),
                    "other_mean": float(np.mean(other_vals)),
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "significant_0.05": p_value < 0.05,
                    "significant_0.01": p_value < 0.01,
                })
            except Exception as e:
                test_results.append({
                    "train_size": train_size,
                    "comparison": f"QNN vs {other_model.upper()}",
                    "n_pairs": len(common_seeds),
                    "statistic": None,
                    "p_value": None,
                    "note": f"Test failed: {e}",
                })

    return test_results


def generate_latex_table(aggregated_df):
    """Generate one LaTeX table per training size for the paper.

    Each table has models as rows and test metrics (R², RMSE, MAE, MSE) as
    columns, formatted as mean $\\pm$ std.  This is the standard layout in
    comparative ML/QML papers.

    Args:
        aggregated_df: Aggregated results DataFrame from aggregate_by_model_and_size

    Returns:
        String containing all four tables, separated by blank lines.
        Each table is wrapped in a table environment with a caption.
    """
    if aggregated_df.empty:
        return "% No data available for LaTeX table"

    # (column header, mean key, std key)
    metric_cols = [
        ("$R^2$",       "mean_test_r2",   "std_test_r2"),
        ("RMSE",        "mean_test_rmse",  "std_test_rmse"),
        ("MAE",         "mean_test_mae",   "std_test_mae"),
        ("MSE",         "mean_test_mse",   "std_test_mse"),
    ]
    n_metrics = len(metric_cols)
    col_spec = "l" + "c" * n_metrics

    table_blocks = []

    for train_size in config.TRAIN_SIZES:
        size_df = aggregated_df[aggregated_df["train_size"] == train_size]

        header_cells = " & ".join(h for h, _, _ in metric_cols)

        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{Test set performance (mean $\\pm$ std, $n={size_df['n_seeds'].max() if not size_df.empty else 10}$ seeds) — training size {train_size}}}",
            f"\\label{{tab:results_{train_size}}}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
            f"Model & {header_cells} \\\\",
            "\\midrule",
        ]

        for model in config.ALL_MODELS:
            row = size_df[size_df["model"] == model]
            cells = [model.upper()]
            for _, mean_key, std_key in metric_cols:
                if len(row) > 0 and mean_key in row.columns:
                    mean_ = row[mean_key].values[0]
                    std_ = row[std_key].values[0]
                    cells.append(f"{mean_:.4f} $\\pm$ {std_:.4f}")
                else:
                    cells.append("--")
            lines.append(" & ".join(cells) + " \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        table_blocks.append("\n".join(lines))

    return "\n\n".join(table_blocks)


def collect_hyperparameters():
    """Collect best hyperparameters from all models for presentation.

    Reads metrics JSON files, extracts hyperparameters, strips sklearn
    pipeline prefixes, and creates a unified table.

    Returns:
        Tuple of (hyperparams_df, hyperparams_latex) where hyperparams_df
        is a DataFrame with one row per model and hyperparams_latex is
        a LaTeX table string.
    """
    results_dir = Path(config.RESULTS_DIR)
    if not results_dir.exists():
        return pd.DataFrame(), "% No hyperparameter data"

    # Collect hyperparams per model (use largest train size, first seed found)
    model_params = {}

    for model in config.ALL_MODELS:
        model_dir = results_dir / model
        if not model_dir.exists():
            continue

        # Find a representative metrics file (largest train_size available)
        for train_size in reversed(config.TRAIN_SIZES):
            for seed in config.SEEDS:
                metrics_file = model_dir / f"seed_{seed}" / f"exp_{train_size}_metrics.json"
                if metrics_file.exists():
                    with open(metrics_file, "r") as f:
                        metrics = json.load(f)

                    if model == "qnn":
                        # QNN stores params under "config"
                        raw = metrics.get("config", {})
                        params = {k: v for k, v in raw.items()}
                    else:
                        # Classical models store under "best_params" with model__ prefix
                        raw = metrics.get("best_params", {})
                        params = {
                            k.replace("model__", ""): v
                            for k, v in raw.items()
                        }

                    model_params[model] = params
                    break
            if model in model_params:
                break

    if not model_params:
        return pd.DataFrame(), "% No hyperparameter data"

    # Build CSV table: one row per model, columns = all unique param names
    rows = []
    for model in config.ALL_MODELS:
        if model in model_params:
            row = {"model": model.upper()}
            row.update(model_params[model])
            rows.append(row)

    df = pd.DataFrame(rows)

    # Build LaTeX table (model | param1 | param2 | ...)
    lines = [
        "% Auto-generated hyperparameter table",
        "% One representative configuration per model (best from GridSearchCV)",
        "\\begin{tabular}{l l}",
        "\\toprule",
        "Model & Hyperparameters \\\\",
        "\\midrule",
    ]

    for model in config.ALL_MODELS:
        if model not in model_params:
            continue
        params = model_params[model]
        # Format as key=value pairs
        param_strs = []
        for k, v in sorted(params.items()):
            if v is None:
                param_strs.append(f"{k}=None")
            elif isinstance(v, float):
                param_strs.append(f"{k}={v:g}")
            else:
                param_strs.append(f"{k}={v}")
        params_text = ", ".join(param_strs)
        # Escape underscores for LaTeX
        params_text_tex = params_text.replace("_", "\\_")
        model_tex = model.upper().replace("_", "\\_")
        lines.append(f"{model_tex} & {params_text_tex} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
    ])

    latex = "\n".join(lines)
    return df, latex


def save_results(aggregated_df, test_results, latex_table, hp_df=None, hp_latex=None):
    """Save all aggregated results to files.

    Args:
        aggregated_df: Aggregated results DataFrame
        test_results: List of Wilcoxon test results
        latex_table: LaTeX table string
        hp_df: Hyperparameter DataFrame (optional)
        hp_latex: Hyperparameter LaTeX table (optional)
    """
    output_dir = Path(config.AGGREGATED_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full summary CSV (includes mean, std, and both CI columns)
    csv_path = output_dir / "summary_table.csv"
    aggregated_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Save a focused CI table (human-readable, one row per model × size × metric)
    ci_rows = []
    metrics = ["r2", "mse", "rmse", "mae"]
    for _, row in aggregated_df.iterrows():
        for split in ("test", "train"):
            for m in metrics:
                col = f"{split}_{m}"
                mean_key = f"mean_{col}"
                if mean_key not in aggregated_df.columns:
                    continue
                ci_rows.append({
                    "model": row["model"],
                    "train_size": row["train_size"],
                    "split": split,
                    "metric": m,
                    "n_seeds": row["n_seeds"],
                    "mean": row.get(mean_key),
                    "std": row.get(f"std_{col}"),
                    "ci95_t_lo": row.get(f"ci95t_lo_{col}"),
                    "ci95_t_hi": row.get(f"ci95t_hi_{col}"),
                    "ci95_norm_lo": row.get(f"ci95n_lo_{col}"),
                    "ci95_norm_hi": row.get(f"ci95n_hi_{col}"),
                })
    if ci_rows:
        ci_path = output_dir / "confidence_intervals.csv"
        pd.DataFrame(ci_rows).to_csv(ci_path, index=False)
        print(f"Saved: {ci_path}")

    # Save LaTeX table
    latex_path = output_dir / "latex_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex_table)
    print(f"Saved: {latex_path}")

    # Save statistical tests as TXT (human-readable)
    tests_path = output_dir / "statistical_tests.txt"
    with open(tests_path, "w") as f:
        f.write("Wilcoxon Signed-Rank Tests (Two-sided)\n")
        f.write("=" * 60 + "\n\n")

        # Note about sample size limitation
        n_seeds = len(config.SEEDS)
        f.write(f"NOTE: With n={n_seeds} paired samples, the minimum achievable\n")
        f.write(f"two-sided Wilcoxon p-value is {2 / (2**n_seeds):.4f}. Significance\n")
        f.write(f"at alpha=0.05 requires n >= 6 seeds.\n\n")

        current_size = None
        for test in test_results:
            if test.get("train_size") != current_size:
                current_size = test["train_size"]
                f.write("=" * 60 + "\n")
                f.write(f"Train size = {current_size} samples\n")
                f.write("=" * 60 + "\n\n")

            f.write(f"{test['comparison']}\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Number of paired samples: {test['n_pairs']}\n")

            if test.get("statistic") is not None:
                f.write(f"  QNN mean R²: {test['qnn_mean']:.4f}\n")
                f.write(f"  Other mean R²: {test['other_mean']:.4f}\n")
                f.write(f"  Test statistic: {test['statistic']:.4f}\n")
                f.write(f"  P-value: {test['p_value']:.6f}\n")
                f.write(f"  Significant at alpha=0.05: {test['significant_0.05']}\n")
                f.write(f"  Significant at alpha=0.01: {test['significant_0.01']}\n")
            else:
                f.write(f"  Note: {test.get('note', 'Test not performed')}\n")

            f.write("\n")

    print(f"Saved: {tests_path}")

    # Save statistical tests as JSON (machine-readable)
    tests_json_path = output_dir / "statistical_tests.json"
    with open(tests_json_path, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"Saved: {tests_json_path}")

    # Save hyperparameter tables
    if hp_df is not None and not hp_df.empty:
        hp_csv_path = output_dir / "hyperparameter_table.csv"
        hp_df.to_csv(hp_csv_path, index=False)
        print(f"Saved: {hp_csv_path}")

    if hp_latex:
        hp_tex_path = output_dir / "hyperparameter_table.tex"
        with open(hp_tex_path, "w") as f:
            f.write(hp_latex)
        print(f"Saved: {hp_tex_path}")


def print_summary(aggregated_df, test_results):
    """Print summary to stdout.

    Args:
        aggregated_df: Aggregated results DataFrame
        test_results: List of Wilcoxon test results
    """
    print("\n" + "=" * 70)
    print("AGGREGATION SUMMARY")
    print("=" * 70)

    if aggregated_df.empty:
        print("No results found to aggregate.")
        return

    print("\nTest metrics (mean ± std) at each training size:\n")

    for train_size in config.TRAIN_SIZES:
        size_df = aggregated_df[aggregated_df["train_size"] == train_size]
        print(f"Train size: {train_size}")
        print(f"  {'Model':<10}  {'R²':>16}  {'RMSE':>16}  {'MAE':>16}  {'MSE':>16}")
        print(f"  {'-'*10}  {'-'*16}  {'-'*16}  {'-'*16}  {'-'*16}")
        for _, row in size_df.iterrows():
            model = row["model"].upper()
            n = int(row["n_seeds"])

            def fmt(mean_key, std_key):
                m = row.get(mean_key)
                s = row.get(std_key)
                if m is None or pd.isna(m):
                    return f"{'--':>16}"
                return f"{m:.4f} ±{s:.4f}"

            r2   = fmt("mean_test_r2",   "std_test_r2")
            rmse = fmt("mean_test_rmse",  "std_test_rmse")
            mae  = fmt("mean_test_mae",   "std_test_mae")
            mse  = fmt("mean_test_mse",   "std_test_mse")
            print(f"  {model:<10}  {r2}  {rmse}  {mae}  {mse}  (n={n})")
        print()

    print("  95 % CIs (t-based and normal) saved to confidence_intervals.csv")

    # Print best model at 3200
    best_3200 = aggregated_df[aggregated_df["train_size"] == 3200].sort_values(
        "mean_test_r2", ascending=False
    )
    if not best_3200.empty:
        best = best_3200.iloc[0]
        print(f"Best model at 3200 samples: {best['model'].upper()}")
        print(f"  R² = {best['mean_test_r2']:.4f} ± {best['std_test_r2']:.4f}")

    # Print Wilcoxon test summary
    print("\n" + "-" * 70)
    print("STATISTICAL TESTS (Wilcoxon signed-rank, two-sided)")
    print("-" * 70)

    n_seeds = len(config.SEEDS)
    print(f"\n  NOTE: With n={n_seeds} seeds, min p-value = {2 / (2**n_seeds):.4f}")

    current_size = None
    for test in test_results:
        if test.get("train_size") != current_size:
            current_size = test["train_size"]
            print(f"\n  --- Train size = {current_size} ---")

        if test.get("p_value") is not None:
            sig = "*" if test["significant_0.05"] else ""
            sig += "*" if test["significant_0.01"] else ""
            print(f"  {test['comparison']}: p={test['p_value']:.4f} {sig}")
        else:
            print(f"  {test['comparison']}: {test.get('note', 'N/A')}")

    print("\n  * = significant at alpha=0.05, ** = significant at alpha=0.01")
    print("=" * 70)


def check_missing_seeds(results):
    """Check which seeds are missing for each model/size combination.

    Args:
        results: List of result dictionaries
    """
    if not results:
        print("No results found. Run experiments first.")
        return

    df = pd.DataFrame(results)

    print("\nData Coverage Check:")
    print("-" * 50)

    all_seeds = set(config.SEEDS)

    for model in config.ALL_MODELS:
        for train_size in config.TRAIN_SIZES:
            present = df[(df["model"] == model) & (df["train_size"] == train_size)]
            present_seeds = set(present["seed"].tolist())
            missing = all_seeds - present_seeds

            if missing:
                print(f"  {model.upper()} @ {train_size}: missing seeds {sorted(missing)}")

    if not any(
        set(config.SEEDS) - set(
            df[(df["model"] == m) & (df["train_size"] == s)]["seed"].tolist()
        )
        for m in config.ALL_MODELS
        for s in config.TRAIN_SIZES
    ):
        print("  All experiments complete!")


