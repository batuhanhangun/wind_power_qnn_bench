"""Single-seed reporting must not change the original ten-seed decision."""
import pandas as pd
from qnnbench.config import REFERENCE_DIR
from qnnbench._aggregation.selection import build_summary, decide


def test_full_selection_report_is_unchanged():
    root=REFERENCE_DIR/'selection'
    data=pd.read_csv(root/'selection_per_seed.csv')
    decision,report=decide(build_summary(data))
    assert decision=='full'
    assert report.strip()==(root/'selection_decision.txt').read_text(encoding='utf-8').strip()


def test_single_seed_has_no_stability_decision():
    data=pd.read_csv(REFERENCE_DIR/'selection/selection_per_seed.csv')
    summary=build_summary(data[data.seed==42])
    decision,report=decide(summary)
    assert decision is None
    assert 'require at least two seeds' in report
    assert summary.cv_val_r2_run_to_run_sigma.isna().all()
    assert len(summary)==3
