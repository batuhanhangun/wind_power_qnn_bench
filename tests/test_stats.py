import numpy as np
import pytest
from qnnbench.stats import benchmark_tests, resolution_floor, paired_test, load_benchmark, derived_statistics
from scripts.verify_results import verify, check_cell

def test_table_iv_pvalues():
    expected={'ANN':[1.52e-3,8.92e-2],'ANN-Reg':[1.41e-6,3.97e-1],
              'SVR':[1.82e-12,4.22e-2],'DTR':[1.82e-12,1.82e-12],
              'XGBoost':[1.82e-12,1.82e-12],'RF':[1.82e-12,1.82e-12]}
    result=benchmark_tests()
    assert len(result)==12 and set(result.n_pairs)=={40}
    for model,values in expected.items():
        rows=result[result.model==model]
        for value,wanted in zip(rows.p_value,values):
            assert f'{value:.2e}'==f'{wanted:.2e}'
    floor_rows=result[result.at_floor]
    assert len(floor_rows)==7
    np.testing.assert_array_equal(floor_rows.p_value, resolution_floor(40))

def test_pairing_rejects_duplicate_and_missing_rows():
    df=load_benchmark()
    q=df[df.model=='QNN']
    other=df[df.model=='ANN']
    expected=list(q[['seed','train_size']].itertuples(index=False,name=None))
    for invalid in [other.iloc[:-1],other.iloc[[0]*40]]:
        with pytest.raises(ValueError):
            paired_test(q,invalid,'test_r2',['seed','train_size'],expected)
    shuffled=paired_test(q.sample(frac=1,random_state=2),other.sample(frac=1,random_state=3),
                         'test_r2',['seed','train_size'],expected)
    assert shuffled['p_value']==benchmark_tests().iloc[0].p_value

def test_derived_statistics():
    stats=derived_statistics()
    assert stats['qnn_runs']==40
    assert 0<stats['qnn_negative_gap_runs']<40
    assert len(stats['collapsed_ann_prediction_means_kw'])==2

def test_printed_precision_is_not_double_rounded():
    assert check_cell('t','r',1,'0.006',0.00649).passed
    assert not check_cell('t','r',1,'0.007',0.00649).passed
    assert check_cell('t','r',1,r'1.82 \times 10^{-12}',2/2**40).passed
    assert check_cell('t','r',1,r'6.4 \times 10^{-2}',0.064453125).passed
    assert not check_cell('t','r',1,r'6.5 \times 10^{-2}',0.064453125).passed

def test_recovered_ten_seed_reference_statistics():
    from qnnbench.stats import noise_aware_tests, capacity_tests
    noise=noise_aware_tests()
    assert len(noise)==8 and set(noise.n_pairs)=={10}
    first=noise[(noise.noise_p==0.001)&(noise.control=='freeze_eval')].iloc[0]
    assert first.statistic==9 and first.p_value==0.064453125
    assert noise.at_floor.sum()==7
    capacity=capacity_tests().set_index('metric')
    assert set(capacity.n_pairs)=={10}
    assert capacity.loc['generalization_gap','p_value']==0.232421875
    assert capacity.loc['test_r2','p_value']==0.10546875

def test_ten_seed_capacity_pairing(monkeypatch):
    """Synthetic unit fixture: all nonzero differences point in one direction."""
    import pandas as pd
    from qnnbench import stats
    rows=[]
    for seed in range(42,52):
        for reps in [3,5]:
            offset=(seed-40)*0.001 if reps==5 else 0
            rows.append(dict(seed=seed,reps=reps,test_r2=0.8+offset,generalization_gap=0.03-offset))
    data=pd.DataFrame(rows).sample(frac=1,random_state=7)
    monkeypatch.setattr(stats.pd,'read_csv',lambda path:data)
    result=stats.capacity_tests()
    assert list(result.metric)==['generalization_gap','test_r2']
    assert set(result.n_pairs)=={10}
    assert (result.p_value==2/2**10).all()
    assert result.at_floor.all()

def test_ten_seed_noise_aware_pairing(monkeypatch):
    import pandas as pd
    from qnnbench import stats
    rows=[]
    for p in stats.NOISE_AWARE_LEVELS:
        for seed in range(42,52):
            for mode,shift in [('freeze_eval',-1),('noise_aware',0),('freeze_eval_calibrated',1)]:
                rows.append(dict(seed=seed,noise_p=p,mode=mode,test_r2=0.8+shift*(seed-40)*0.001))
    data=pd.DataFrame(rows).sample(frac=1,random_state=8)
    monkeypatch.setattr(stats.pd,'read_csv',lambda path:data)
    result=stats.noise_aware_tests()
    assert len(result)==8 and set(result.n_pairs)=={10}
    assert (result.p_value==2/2**10).all()
    assert result.at_floor.all()
    monkeypatch.setattr(stats.pd,'read_csv',lambda path:data.iloc[:-1])
    with pytest.raises(ValueError): stats.noise_aware_tests()

def test_published_numeric_cells():
    checks=verify()
    failed=[f'{c.table}/{c.row}/{c.column}: {c.printed} vs {c.actual}; {c.reason}'
            for c in checks if not c.passed]
    assert not failed, '\n'.join(failed)
