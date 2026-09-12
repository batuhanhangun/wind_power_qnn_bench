"""Paired statistical tests; no experiment training is implemented here."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from qnnbench.config import REFERENCE_DIR, SEEDS, TRAIN_SIZES, NOISE_AWARE_LEVELS

MODELS = ['ANN', 'ANN-Reg', 'SVR', 'DTR', 'XGBoost', 'RF']
KEYS = ['seed', 'train_size']

def resolution_floor(n):
    """Smallest two-sided exact p-value for n nonzero, untied differences."""
    return 2 / 2**n

def validate_keys(df, keys, expected):
    if df[keys].isna().any().any() or df.duplicated(keys).any():
        raise ValueError(f'Missing or duplicate pairing keys: {keys}')
    actual = set(df[keys].itertuples(index=False, name=None))
    if actual != set(expected):
        raise ValueError(f'Pair coverage mismatch: expected {len(expected)}, got {len(actual)}; '
                         f'missing={set(expected)-actual}, extra={actual-set(expected)}')

def paired_test(x, y, metric, keys, expected):
    validate_keys(x, keys, expected)
    validate_keys(y, keys, expected)
    pairs = x[keys+[metric]].merge(y[keys+[metric]], on=keys, validate='one_to_one',
                                 suffixes=('_x','_y')).sort_values(keys)
    assert len(pairs) == len(expected)
    a,b = pairs[metric+'_x'].to_numpy(), pairs[metric+'_y'].to_numpy()
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite values in paired test')
    result = wilcoxon(a, b, alternative='two-sided', method='auto')
    floor = resolution_floor(len(pairs))
    return {'metric': metric, 'n_pairs': len(pairs), 'x_mean': float(a.mean()),
            'y_mean': float(b.mean()), 'statistic': float(result.statistic),
            'p_value': float(result.pvalue), 'resolution_floor': floor,
            'at_floor': bool(np.isclose(result.pvalue, floor, rtol=1e-12, atol=0))}

def load_benchmark(root=REFERENCE_DIR):
    root = Path(root)
    frames = []
    for name, filename, model in [('qnn','qnn_per_seed.csv','QNN'),
                                 ('classical','classical_per_seed.csv',None),
                                 ('ann_reg','ann_reg_classical_per_seed.csv','ANN-Reg')]:
        df = pd.read_csv(root/name/filename)
        if model: df['model'] = model
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def benchmark_tests(root=REFERENCE_DIR, seeds=SEEDS, sizes=TRAIN_SIZES):
    df = load_benchmark(root)
    df = df[df.seed.isin(seeds) & df.train_size.isin(sizes)]
    expected = [(s,n) for s in seeds for n in sizes]
    rows = []
    for model in MODELS:
        for metric in ['test_r2','generalization_gap']:
            row = paired_test(df[df.model=='QNN'],df[df.model==model],metric,KEYS,expected)
            row['model'] = model
            rows.append(row)
    return pd.DataFrame(rows)

def noise_aware_tests(root=REFERENCE_DIR, seeds=SEEDS):
    df = pd.read_csv(Path(root)/'noise_aware/noise_aware_per_seed.csv')
    df = df[df.seed.isin(seeds)]
    rows=[]
    for p in NOISE_AWARE_LEVELS:
        g=df[df.noise_p==p]
        for control in ['freeze_eval','freeze_eval_calibrated']:
            row=paired_test(g[g['mode']=='noise_aware'],g[g['mode']==control],
                            'test_r2',['seed'],[(s,) for s in seeds])
            row.update(noise_p=p,control=control)
            rows.append(row)
    return pd.DataFrame(rows)

def capacity_tests(root=REFERENCE_DIR, seeds=SEEDS):
    df=pd.read_csv(Path(root)/'capacity/capacity_per_seed.csv')
    df=df[df.seed.isin(seeds)]
    return pd.DataFrame([paired_test(df[df.reps==3],df[df.reps==5],metric,['seed'],
                                    [(s,) for s in seeds])
                         for metric in ['generalization_gap','test_r2']])

def derived_statistics(root=REFERENCE_DIR, seeds=SEEDS, sizes=TRAIN_SIZES):
    root=Path(root)
    df=load_benchmark(root)
    df=df[df.seed.isin(seeds)&df.train_size.isin(sizes)]
    q=df[df.model=='QNN']
    validate_keys(q,KEYS,[(s,n) for s in seeds for n in sizes])
    gap=q.generalization_gap
    collapsed=df[(df.model=='ANN') & (df.train_size==3200) & (df.test_r2<0.25)]
    if set(seeds)==set(SEEDS) and 3200 in sizes and len(collapsed)!=2:
        raise ValueError(f'Expected two collapsed ANN seeds; found {len(collapsed)}')
    levels={}
    for seed in collapsed.seed:
        path=root/f'predictions/ann/seed_{seed}/exp_3200_predictions.csv'
        if not path.exists(): path=root/f'classical/ann/seed_{seed}/exp_3200_predictions.csv'
        pred=pd.read_csv(path)
        if len(pred)!=893: raise ValueError('Incomplete prediction file')
        levels[int(seed)]=float(pred.Predicted.mean())
    ratios={m: float(g.generalization_gap.std(ddof=1)/g.generalization_gap.mean())
            for m,g in df.groupby('model')}
    return {'qnn_negative_gap_runs':int((gap<0).sum()),'qnn_runs':len(q),
            'gap_std_over_mean':ratios,'collapsed_ann_prediction_means_kw':levels}

def latex_wilcoxon(table):
    lines=[r'\begin{tabular}{llrrrl}',r'Model & Metric & QNN mean & Classical mean & $p$ & Sig. \\',r'\hline']
    for r in table.itertuples(index=False):
        mantissa,exponent=f'{r.p_value:.2e}'.split('e')
        p=rf'${mantissa} \times 10^{{{int(exponent)}}}$'
        metric=r'$R^2$' if r.metric=='test_r2' else r'$\Delta R^2$'
        sig='**' if r.p_value<0.01 else '*' if r.p_value<0.05 else 'ns'
        lines.append(f'{r.model} & {metric} & {r.x_mean:.3f} & {r.y_mean:.3f} & {p} & {sig} '+r'\\')
    return '\n'.join(lines+[r'\end{tabular}'])+'\n'
