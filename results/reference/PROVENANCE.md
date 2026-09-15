# Reference results — change log

The files in this directory are experiment outputs, copied verbatim from the runs that
produced them. `FILE_MANIFEST.json` records each file's SHA-256 digest, and
`tests/test_reference_integrity.py` checks every digest, so an accidental change to a
reference value fails the test suite.

This file records changes to the reference set after its initial publication.

## 2026-09-13 — ANN-Reg re-run under negative-RMSE selection

ANN-Reg was re-run for all ten seeds (42–51) and all four training sizes (800, 1600,
2400, 3200) with the grid search selecting by negative RMSE instead of R², so that every
classical model, ANN-Reg included, is tuned by the same criterion as
`qnnbench/classical.py`: multi-metric scoring `{"neg_rmse", "neg_mae", "r2"}` with
`refit="neg_rmse"`, recording R², RMSE and MAE for every candidate. Nothing else changed:
same parameter grid, fixed estimator settings, 5-fold CV, seeds, splits, single-thread
BLAS settings and `n_jobs=1`.

`--selection neg_rmse` is now the default in `qnnbench/ann_reg.py`; `--selection r2`
remains available and reproduces the earlier behaviour.

Source run: `results/runs/ann_reg_rmse/` (2026-09-12, `run_local`, 8518 s).

Scope of the change:

- **Only ANN-Reg changed.** No other experiment under `results/reference/` was re-run or
  edited.
- **2 of the 40 runs selected different hyperparameters:** seed 48 at N=3200 and seed 51
  at N=800. Seed 48's test R² is unchanged to four decimals (0.9355) but its predictions
  differ, so Figure 4 was regenerated; seed 51's test R² at N=800 falls from 0.9075 to
  0.8257. The remaining 38 runs selected the same hyperparameters as before.
- Files replaced: `ann_reg/ann_reg_per_seed.csv`, `ann_reg/ann_reg_summary.csv`,
  `ann_reg/ann_reg_classical_per_seed.csv`, `ann_reg/reproduction_report.txt`, and
  `predictions/ann_reg/seed_48/exp_3200_predictions.csv` (the only prediction file whose
  contents changed). Their digests in `FILE_MANIFEST.json` were updated accordingly.
- The reproduction gate now compares against this promoted reference. A fresh run with
  default settings (`python -m scripts.run_ann_reg --seeds 42`) reproduces it to 1e-6;
  this was verified on 2026-09-13.
- `results/reference/published_tables.json` was **not** updated. The manuscript still
  holds the R²-selection values, so `scripts/verify_results.py` fails on the ANN-Reg
  cells by design until the manuscript is revised and the published cells are updated to
  match.

`ann_reg/reproduction_report.txt` is the gate report from a ten-seed, four-size run with
default settings (`results/runs/ann_reg_default/`, 2026-09-14), which reproduces the
promoted CSVs exactly: all 40 rows match within 1e-6, and every column of
`ann_reg_per_seed.csv` is bit-identical, `best_params` included. Its `Reference:` line was
rewritten from the launcher's absolute temporary path to the repo-relative path of the
file actually compared against, so the artifact is not machine-specific; nothing else in
the report was edited.

## Original runs

| Experiment | Original run | Date | Contents |
|---|---|---|---|
| Classical | `run_20260326_083044` | 2026-03-26 | Aggregates, per-run metrics, reduced CSV, predictions |
| QNN | not retained | unknown | 40 per-seed rows, per-run metrics, N=3200 predictions |
| Noise | not retained | unknown | 240 per-run rows, six-level summary |
| ANN-Reg | `run_local` | 2026-09-12 (negative-RMSE selection; see above) | 40 per-seed rows, reproduction report |
| Selection | `run_20260624_132249` | 2026-06-24 | Per-seed CSV, summary, report |
| Capacity | `run_20260628_134952` | 2026-06-28 | Per-seed CSV, summary, report |
| Noise-aware | `run_20260907_120412` | 2026-09-07 | Per-seed CSV, summary, report |

The original experiments ran on CPython 3.11.0rc1 with the package versions pinned in
`requirements.txt` and Matplotlib 3.10.3. The ANN-Reg re-run above was executed on
CPython 3.12.10 on Windows.
