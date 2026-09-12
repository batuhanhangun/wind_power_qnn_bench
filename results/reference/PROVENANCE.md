# Reference results

The files in this directory are the original experiment outputs, copied verbatim from the
runs that produced the published numbers. Nothing here is regenerated or edited.
`FILE_MANIFEST.json` records each file's SHA-256 digest, and `tests/test_reference_integrity.py`
checks every digest, so an accidental change to a reference value fails the test suite.

| Experiment | Original run | Date | Contents |
|---|---|---|---|
| Classical | `run_20260326_083044` | 2026-03-26 | Aggregates, per-run metrics, reduced CSV, predictions |
| QNN | not retained | unknown | 40 per-seed rows, per-run metrics, N=3200 predictions |
| Noise | not retained | unknown | 240 per-run rows, six-level summary |
| ANN-Reg | `run_local` | 2026-09-10 (file timestamp) | 40 per-seed rows, reproduction report |
| Selection | `run_20260624_132249` | 2026-06-24 | Per-seed CSV, summary, report |
| Capacity | `run_20260628_134952` | 2026-06-28 | Per-seed CSV, summary, report |
| Noise-aware | `run_20260907_120412` | 2026-09-07 | Per-seed CSV, summary, report |

Run identifiers encode the run date; the timezone is not recorded. Dates marked "unknown"
come from copies whose original run directory was not retained.

`published_tables.json` holds the published table cells that `scripts/verify_manuscript.py`
compares against these results. It contains printed numbers only, no derivation.

The classical aggregate directory retains historical rows from an earlier circular-QNN
configuration. The verifier uses the five classical models and `qnn/qnn_per_seed.csv` for the
deployed reverse-linear QNN, and recomputes the paired tests in `qnnbench/stats.py`.

The raw dataset and license are also tracked in
`https://github.com/batuhanhangun/qnn_bench_for_wind_power`; the raw file here is byte-identical
to that repository's copy.

The original experiments ran on CPython 3.11.0rc1 with the package versions pinned in
`requirements.txt` and Matplotlib 3.10.3. Local reproduction of this repository was checked on
CPython 3.12.10 on Windows, where the fresh QNN and ANN-Reg training gates pass.
