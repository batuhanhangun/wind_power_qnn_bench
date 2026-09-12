import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from qnnbench import config

@pytest.mark.slow
def test_seed42_n800_fresh_training():
    from qnnbench.qnn import train_model, scale_data, predict_original_scale
    from qnnbench.data import load_data
    from qnnbench.metrics import compute_metrics
    X,y,Xt,yt=load_data(42,800)
    Xs,ys,Xts,_,_,scaler=scale_data(X,y,Xt,yt)
    weights=train_model(Xs,ys,maxiter=config.QNN_CONFIG['maxiter'],seed=42)
    result=compute_metrics(yt,predict_original_scale(weights,Xts,scaler))['r2']
    ref=pd.read_csv(config.REFERENCE_DIR/'qnn/qnn_per_seed.csv')
    expected=ref[(ref.seed==42)&(ref.train_size==800)].test_r2.item()
    assert abs(result-expected)<=1e-6, (result,expected)
