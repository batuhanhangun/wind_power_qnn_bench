# QNN wind-power benchmark

Code and reference results for the study “Benchmarking Quantum Neural Networks for Wind Power Forecasting: Generalization Stability and Noise Robustness Under Data-Scarce Conditions,” by Batuhan Hangun, Oguz Altun, Onder Eyecioglu, and Tahir Cetin Akinci. It includes the data splits, classical and quantum model experiments, statistical tests, and scripts for all five figures. The experiments run locally on a CPU; no quantum hardware is required.

## Download and install

Local reproduction was tested with **Python 3.12.10 on Windows**. Install Python and Git, then clone the repository:

```text
git clone https://github.com/batuhanhangun/wind_power_qnn_bench.git
cd wind_power_qnn_bench
```

Alternatively, download the repository ZIP from GitHub, extract it, and open a terminal in the extracted folder.

Create and activate a virtual environment. On Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the pinned dependencies and the local package:

```text
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run all commands below from the repository root with this environment active. The data and reference results are included. Keep the pinned package versions for numerical reproduction, especially SciPy's Wilcoxon calculations and ANN-Reg training. Environment details are recorded in [PROVENANCE.md](results/reference/PROVENANCE.md).

## Verify the published results

Check the supplied reference results without retraining:

```text
python -m scripts.verify_results
python -m pytest -m "not slow" --tb=short
```

The verifier compares the seven published result tables, stored as `results/reference/published_tables.json`, against the reference data at the precision each value is printed to. Expected result: **184/184 numeric cells PASS; 0 FAIL**. The fast test suite also checks circuits, data, statistics, and figure inputs.

To include fresh QNN and ANN-Reg training:

```text
python -m pytest
```

The full suite passed all 26 tests in approximately 20 minutes on the machine described below. The two training gates compare against reference results with an absolute tolerance of `1e-6`.

## Run the experiments

Commands below run all ten seeds (42–51). For a shorter run, add `--seeds 42`. Classical, QNN, noise, and ANN-Reg experiments use training sizes 800, 1600, 2400, and 3200 by default; add `--sizes 800` to run just that size. Architecture selection, capacity, and noise-aware training use N=800.

For example, start with one classical seed and one training size:

```text
python -m scripts.run_classical --seeds 42 --sizes 800 --out results/runs/classical_trial
```

Choose a new, empty `--out` directory for each run. Scripts print the output location; training scripts also save elapsed time and completion status in `runtime.json`. New outputs go under `results/runs/`.

### Classical models

```text
python -m scripts.run_classical --out results/runs/classical
```

Trains SVR, DTR, RF, XGBoost, and ANN. Writes per-run metrics and predictions, `classical_per_seed.csv`, and `summary_table.csv`.

### QNN

```text
python -m scripts.run_qnn --out results/runs/qnn
```

Runs five-fold cross-validation and final training for the reverse-linear QNN. Writes metrics, predictions, convergence records, frozen weights, `qnn_per_seed.csv`, and `qnn_performance_summary.csv`.

### Freeze-and-evaluate noise

```text
python -m scripts.run_noise --qnn-results results/runs/qnn --out results/runs/noise
```

Run QNN training first, using the same seeds and sizes. This command loads its weights and metrics, checks deterministic retraining, and evaluates six noise levels. Writes `noise_per_run.csv` and `noise_summary.csv`, plus per-run noise metrics.

### Architecture selection

```text
python -m scripts.run_selection --out results/runs/selection
```

Compares circular, full, and reverse-linear circuits at N=800 with five-fold cross-validation. Writes `selection_per_seed.csv`, `selection_summary.csv`, and `selection_decision.txt`. A single-seed run reports the aggregate decision as unavailable because an across-seed standard deviation requires multiple seeds.

### Capacity ablation

```text
python -m scripts.run_capacity --out results/runs/capacity
```

Evaluates one through five circuit repetitions at N=800. Writes `capacity_per_seed.csv` and `capacity_summary.csv`.

### ANN-Reg

```text
python -m scripts.run_ann_reg --out results/runs/ann_reg
```

Runs the regularized ANN grid search and checks the results against the reference rows. Writes predictions, `ann_reg_per_seed.csv`, `ann_reg_summary.csv`, and `reproduction_report.txt`. A failed reproduction check exits with an error.

### Noise-aware training

```text
python -m scripts.run_noise_aware --out results/runs/noise_aware
```

Runs freeze-and-evaluate, noise-aware training, and affine calibration at four noise levels and N=800. Writes `noise_aware_per_seed.csv` and `noise_aware_summary.csv`. Noise-aware training uses a density-matrix simulator and takes several hours per seed.

### Statistics

```text
python -m scripts.run_stats --out results/runs/stats
```

Computes paired Wilcoxon tests and derived statistics from the supplied references. Writes `stats_summary.txt` and `table_wilcoxon.tex`. To analyze your own results, add `--reference results/runs` when using the individual output paths above, or `--reference results/runs/all` after `run_all`. Match `--seeds` and `--sizes` to the runs being analyzed.

### Figures

```text
python -m scripts.make_figures --out results/runs/figures
```

Generates all five figures from the supplied references as PDF and PNG files. To use your own results, add `--reference results/runs` (or `results/runs/all`) and the matching `--seeds` and `--sizes`. Figure 4 requires N=3200 predictions.

### Run everything

```text
python -m scripts.run_all --seeds 42 --out results/runs/all
```

Runs every experiment in dependency order, then computes statistics and generates figures from those new results. Remove `--seeds 42` for the full ten-seed study. Even one seed takes many hours; use the individual commands for smaller checks.

### Measured runtimes

Measurements below used seed 42 with the full settings for each experiment on a Windows desktop with an AMD Ryzen 7 5800X3D and 32 GB RAM. Some measurements overlapped with other training or validation processes. Laptop runtimes have not been measured. The ten-seed column is a linear extrapolation, not a measured full run; actual time depends on the machine.

| Experiment | One seed, measured | Ten seeds, extrapolated |
|---|---:|---:|
| Classical | 70.6 seconds | 11.8 minutes |
| QNN | 5.24 hours | 52.4 hours |
| Noise (including retraining check) | 51.0 minutes | 8.5 hours |
| Architecture selection | 1.65 hours | 16.5 hours |
| Capacity | 30.7 minutes | 5.12 hours |
| ANN-Reg | 12.2 minutes | 122 minutes |
| Noise-aware | 4.78 hours | 47.8 hours |

Statistics took 1.4 seconds for seed 42; full-population statistics were not timed separately. All five figures took 6.7 seconds using the full reference population. Measurement records are in [runtime_measurements.json](runtime_measurements.json).

## Published tables and figures

Reference paths are relative to `results/reference/` unless noted.

| Published table or figure | Script | Reference input |
|---|---|---|
| Table I: Basic descriptive statistics for the wind turbine dataset | `data/make_splits.py` loads the raw file; descriptive cells are outside the seven verified tables | `data/raw/total_dataset.xlsx` at repository root |
| Table II: QNN architecture-selection results at N=800 (`tab:arch_selection`) | `scripts/run_selection.py` | `selection/selection_per_seed.csv` |
| Table III: Hyperparameter search spaces (`tab:hyperparams`) | Classical, ANN-Reg and QNN scripts; grids in `qnnbench/config.py` | Configuration, not an aggregate result |
| Table IV: Comparative performance at N=3200 (`tab:performance_comparison`) | `scripts/run_classical.py`, `run_qnn.py`, `run_ann_reg.py` | `classical/metrics/`, `qnn/qnn_per_seed.csv`, `ann_reg/ann_reg_per_seed.csv` |
| Table V: Generalization gap per model (`tab:gen_gap`) | `scripts/run_stats.py`, `verify_results.py` | Classical, QNN and ANN-Reg per-seed CSVs |
| Table VI: QNN capacity ablation at N=800 (`tab:capacity`) | `scripts/run_capacity.py` | `capacity/capacity_per_seed.csv` |
| Table VII: Wilcoxon signed-rank test results (`tab:wilcoxon`) | `scripts/run_stats.py` | Classical, QNN and ANN-Reg per-seed CSVs |
| Table VIII: QNN performance under depolarizing noise (`tab:noise`) | `scripts/run_noise.py` | `noise/noise_per_run.csv`, `noise/noise_summary.csv` |
| Table IX: Noise-aware training versus freeze-and-evaluate and a two-parameter output calibration (`tab:noise_aware`) | `scripts/run_noise_aware.py`, `run_stats.py` | `noise_aware/noise_aware_per_seed.csv` |
| Figure 1: The 4-qubit QNN circuit used in this study (`fig:general_qnn`) | `figures/fig1_circuit.py` | Original circuit definition; circuit gates checked in tests |
| Figure 2: Learning dynamics across increasing training dataset sizes (`fig:saturation`) | `figures/fig2_saturation.py` | Classical, QNN and ANN-Reg per-seed CSVs |
| Figure 3: Generalization gap (`fig:generalization`) | `figures/fig3_gap.py` | Classical, QNN and ANN-Reg per-seed CSVs |
| Figure 4: Actual vs. predicted wind power output for all seven models at N=3200 (`fig:scatter`) | `figures/fig4_scatter.py` | `predictions/<model>/seed_<seed>/exp_3200_predictions.csv` and per-seed metric CSVs |
| Figure 5: QNN performance degradation under depolarizing noise | `figures/fig5_noise.py` | `noise/noise_summary.csv` |

## Repository layout

```text
data/                Raw dataset, fixed splits, and split generator
qnnbench/            Experiment implementations and configuration
scripts/             Command-line entry points
results/reference/   Supplied results and provenance
results/runs/        New local outputs (Git-ignored)
figures/             Figure scripts and reference PDFs
tests/               Structural and numerical reproduction checks
```

## Reproducibility notes

The fixed splits use seeds 42–51. To regenerate them from the included raw dataset:

```text
python -m data.make_splits
```

ANN-Reg sets `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1` before numerical-library imports and uses `n_jobs=1`. Its launcher handles these settings automatically.

The reverse-linear noisy circuit has exactly 42 depolarizing-channel insertions at three repetitions. Tests assert the count because channel placement affects the results. Noise-aware results use N=800 with ten seeds; the main noise table pools four sizes and ten seeds. A subset run therefore does not reproduce the full aggregates.

The classical grid search selects by negative RMSE; ANN-Reg selects by R². The architecture report identifies `full` as the CV accuracy leader, while the study uses the numerically equivalent `reverse_linear` circuit with fewer CNOTs. Origin, contents and checksums of the supplied results are documented in [PROVENANCE.md](results/reference/PROVENANCE.md).

## Citation

Citation metadata for this software is provided in [CITATION.cff](CITATION.cff).

## License

[MIT](LICENSE).
