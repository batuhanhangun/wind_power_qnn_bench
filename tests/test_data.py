import pandas as pd
from qnnbench import config

def test_split_regeneration_matches_shipped_seed42(tmp_path,monkeypatch):
    from data.make_splits import generate_splits_for_seed
    original=config.DATA_DIR
    monkeypatch.setattr(config,'DATA_DIR',tmp_path)
    generate_splits_for_seed(42,verbose=False)
    for filename in ['test_set.csv']+[f'train_{n}.csv' for n in config.TRAIN_SIZES]:
        expected=pd.read_csv(original/'seed_42'/filename,sep=';')
        actual=pd.read_csv(tmp_path/'seed_42'/filename,sep=';')
        pd.testing.assert_frame_equal(actual,expected,check_exact=True)

def test_all_split_shapes_and_nested_training_prefixes():
    for seed in config.SEEDS:
        root=config.DATA_DIR/f'seed_{seed}'
        test=pd.read_csv(root/'test_set.csv',sep=';')
        full=pd.read_csv(root/'train_3200.csv',sep=';')
        assert test.shape==(893,5) and full.shape==(3200,5)
        for size in config.TRAIN_SIZES:
            subset=pd.read_csv(root/f'train_{size}.csv',sep=';')
            pd.testing.assert_frame_equal(subset,full.iloc[:size],check_exact=True)
