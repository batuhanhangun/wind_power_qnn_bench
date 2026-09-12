import numpy as np
from figures import load_data

def test_every_figure_input():
    all_rows=load_data.load_all_per_seed()
    assert len(all_rows)==280
    assert not all_rows.duplicated(['model','seed','train_size']).any()
    assert np.isfinite(all_rows[['test_r2','train_r2','generalization_gap']]).all().all()
    noise=load_data.load_noise_summary()
    assert len(noise)==6
    pooled,missing=load_data.load_predictions_3200()
    assert not missing, missing
    assert len(pooled)==7
    for frame in pooled.values():
        assert len(frame)==8930
        assert np.isfinite(frame[['Actual','Predicted']]).all().all()
