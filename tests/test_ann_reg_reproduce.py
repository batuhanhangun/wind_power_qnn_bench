import json
import subprocess
import sys
import pandas as pd
import pytest
from qnnbench import config

@pytest.mark.slow
def test_fresh_ann_reg_rows(tmp_path):
    out=tmp_path/'ann_reg'
    command=[sys.executable,'-m','scripts.run_ann_reg','--seeds','42','--out',str(out)]
    result=subprocess.run(command,cwd=config.ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    assert result.returncode==0,result.stdout[-10000:]
    fresh=pd.read_csv(out/'ann_reg_per_seed.csv')
    ref=pd.read_csv(config.REFERENCE_DIR/'ann_reg/ann_reg_per_seed.csv')
    paired=fresh.merge(ref[ref.seed==42],on=['seed','train_size'],validate='one_to_one',suffixes=('_new','_ref'))
    assert len(paired)==4
    for metric in ['test_r2','train_r2','generalization_gap']:
        assert (abs(paired[metric+'_new']-paired[metric+'_ref'])<=1e-6).all()
