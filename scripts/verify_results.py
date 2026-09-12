"""Compare every published result-table cell with the shipped original records.

The published cells are stored in `results/reference/published_tables.json`, so the
verifier needs no LaTeX source. One may still be passed explicitly.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from qnnbench import config
from qnnbench.stats import benchmark_tests, load_benchmark, noise_aware_tests, validate_keys

LABELS=['tab:arch_selection','tab:performance_comparison','tab:gen_gap','tab:wilcoxon',
        'tab:noise','tab:capacity','tab:noise_aware']

@dataclass
class Cell:
    table: str
    row: str
    column: int
    printed: str
    actual: str
    passed: bool
    reason: str = ''

def clean(text):
    return re.sub(r'\\(?:textbf|mathbf)\{([^{}]*)\}',r'\1',text).replace('$','').strip()

def published_tables(path=None):
    """Load the published cells: a frozen JSON table dump, or a LaTeX source."""
    path=Path(path or config.PUBLISHED_TABLES)
    found=json.loads(path.read_text(encoding='utf-8')) if path.suffix=='.json' else latex_tables(path)
    return validate_tables(found)

def latex_tables(path):
    text=Path(path).read_text(encoding='utf-8')
    text=re.sub(r'(?<!\\)%[^\n]*','',text)
    found={}
    for block in re.findall(r'\\begin\{table\*?\}.*?\\end\{table\*?\}',text,re.S):
        label=re.search(r'\\label\{([^}]+)\}',block)
        if not label or label[1] not in LABELS: continue
        body=re.search(r'\\begin\{tabular\}\{[^\n]+\}(.*?)\\end\{tabular\}',block,re.S)[1]
        rows=[]
        for line in body.splitlines():
            if '&' not in line or not line.rstrip().endswith('\\'): continue
            cells=[clean(c) for c in line.rstrip()[:-2].split('&')]
            rows.append(cells)
        found[label[1]]=rows[1:] # header is never treated as evidence
    return found

def validate_tables(found):
    if set(found)!=set(LABELS): raise ValueError(f'Missing published tables: {set(LABELS)-set(found)}')
    expected_counts=dict(zip(LABELS,[3,7,7,12,6,5,4]))
    for label,rows in found.items():
        if len(rows)!=expected_counts[label]:
            raise ValueError(f'Could not extract every row of {label}: {len(rows)} rows')
    return found

def numeric_tokens(cell):
    cell=clean(cell)
    match=re.fullmatch(r'([+-]?\d+(?:\.\d+)?)\s*\\times\s*10\^\{([+-]?\d+)\}',cell)
    if match:
        decimals=len(match[1].partition('.')[2])
        return [(float(match[1])*10**int(match[2]),decimals,int(match[2]))]
    if '\\pm' in cell:
        return sum((numeric_tokens(v.strip()) for v in cell.split('\\pm')),[])
    if re.fullmatch(r'[+-]?\d+(?:\.\d+)?',cell):
        return [(float(cell),len(cell.partition('.')[2]),None)]
    if re.fullmatch(r'\d+\.\d+ \(baseline\)',cell): return numeric_tokens(cell.split()[0])
    return []

def check_cell(table,row,column,printed,actual,reason=''):
    tokens=numeric_tokens(printed)
    values=actual if isinstance(actual,(list,tuple)) else [actual]
    if reason or len(values)!=len(tokens):
        return Cell(table,row,column,printed,'MISSING',False,reason or 'Cell arity mismatch')
    rendered=[]
    ok=True
    for (expected,precision,exponent),value in zip(tokens,values):
        if value is None or not np.isfinite(value):
            rendered.append('MISSING'); ok=False; continue
        # Compare at exactly the precision printed in this cell, including scientific notation.
        scale=1 if exponent is None else 10**exponent
        want=f'{expected/scale:.{precision}f}'
        got=f'{value/scale:.{precision}f}'
        ok &= want==got
        rendered.append(got if exponent is None else f'{got}e{exponent}')
    return Cell(table,row,column,printed,' +/- '.join(rendered),bool(ok))

def mean_std(series):
    return [float(series.mean()),float(series.std(ddof=1))]

def table_values(label,root):
    """Recompute from per-run data; no published number enters this function."""
    if label in ['tab:performance_comparison','tab:gen_gap','tab:wilcoxon']:
        df=load_benchmark(root)
        for _,g in df.groupby('model'):
            validate_keys(g,['seed','train_size'],[(s,n) for s in config.SEEDS for n in config.TRAIN_SIZES])
        if label=='tab:gen_gap':
            means=df.groupby('model').generalization_gap.mean()
            ranks=means.rank(method='min').astype(int)
            return {m:{1:mean_std(g.generalization_gap),2:len(g),3:int(ranks[m])} for m,g in df.groupby('model')}
        if label=='tab:wilcoxon':
            result={}
            for r in benchmark_tests(root).itertuples(index=False):
                result[(r.model,r.metric)]={2:r.x_mean,3:r.y_mean,4:r.p_value}
            return result
        frames=[pd.read_csv(root/'qnn/qnn_per_seed.csv').assign(model='QNN'),
                pd.read_csv(root/'ann_reg/ann_reg_per_seed.csv').assign(model='ANN-Reg')]
        records=[json.loads(p.read_text(encoding='utf-8')) for p in (root/'classical/metrics').glob('*/seed_*/exp_*_metrics.json')]
        classical=pd.DataFrame(records)
        classical['model']=classical.model.replace({'ann':'ANN','svr':'SVR','rf':'RF','dtr':'DTR','xgboost':'XGBoost'})
        frames.append(classical)
        full=pd.concat(frames,ignore_index=True)
        output={}
        for m,g in full[full.train_size==3200].groupby('model'):
            validate_keys(g,['seed'],[(s,) for s in config.SEEDS])
            output[m]={c:mean_std(g[metric]) for c,metric in enumerate(['test_r2','test_mse','test_rmse','test_mae'],1)}
        return output
    if label=='tab:noise':
        df=pd.read_csv(root/'noise/noise_per_run.csv')
        baseline=df[df.noise_p==0].test_r2.mean()
        out={}
        for p,g in df.groupby('noise_p'):
            validate_keys(g,['seed','train_size'],[(s,n) for s in config.SEEDS for n in config.TRAIN_SIZES])
            out[float(p)]={0:p,**{c:mean_std(g[m]) for c,m in enumerate(['test_r2','test_mse','test_rmse','test_mae'],1)},
                           5:(g.test_r2.mean()-baseline)/baseline*100}
        if set(out)!=set(config.NOISE_LEVELS): raise ValueError('Noise-level coverage mismatch')
        return out
    if label=='tab:arch_selection':
        df=pd.read_csv(root/'selection/selection_per_seed.csv')
        out={}
        for name,g in df.groupby('config'):
            validate_keys(g,['seed'],[(s,) for s in config.SEEDS])
            out[name]={2:len(config.ARCHITECTURE_PAIRS[name])*config.QNN_CONFIG['ansatz_reps'],
                       3:mean_std(g.cv_val_r2),4:g.cv_gap.mean(),5:g.cv_val_r2.std(ddof=1),6:g.test_r2.mean()}
        return out
    if label=='tab:capacity':
        df=pd.read_csv(root/'capacity/capacity_per_seed.csv')
        out={}
        for reps,g in df.groupby('reps'):
            validate_keys(g,['seed'],[(s,) for s in config.SEEDS])
            out[int(reps)]={0:reps,1:4*(reps+1),2:3*reps,3:mean_std(g.test_r2),4:mean_std(g.generalization_gap)}
        return out
    if label=='tab:noise_aware':
        df=pd.read_csv(root/'noise_aware/noise_aware_per_seed.csv')
        tests=noise_aware_tests(root)
        out={}
        for p,g in df.groupby('noise_p'):
            row={0:p}
            for c,mode in enumerate(['freeze_eval','noise_aware','freeze_eval_calibrated'],1):
                group=g[g['mode']==mode]
                validate_keys(group,['seed'],[(s,) for s in config.SEEDS])
                row[c]=mean_std(group.test_r2)
            for c,control in [(4,'freeze_eval'),(5,'freeze_eval_calibrated')]:
                row[c]=tests[(tests.noise_p==p)&(tests.control==control)].p_value.item()
            out[float(p)]=row
        return out
    raise ValueError(label)

def verify(published=None,root=config.REFERENCE_DIR):
    tables=published_tables(published)
    checks=[]
    for label,rows in tables.items():
        try:
            values=table_values(label,Path(root)); reason=''
        except (FileNotFoundError,ValueError,KeyError) as exc:
            values={}; reason=str(exc)
        previous=''
        for row in rows:
            name=row[0] or previous
            previous=name
            if label=='tab:wilcoxon': key=(name,'test_r2' if 'Delta' not in row[1] else 'generalization_gap')
            elif label=='tab:arch_selection': key=name.lower().replace(' ','_')
            elif label=='tab:capacity': key=int(name)
            elif label in ['tab:noise','tab:noise_aware']: key=float(name.split()[0])
            else: key=name
            for c,printed in enumerate(row):
                if not numeric_tokens(printed): continue
                missing=reason or ('' if key in values and c in values[key] else f'Missing computed row/cell {key}, {c}')
                checks.append(check_cell(label,str(key),c,printed,values.get(key,{}).get(c),missing))
    return checks

def format_report(checks):
    lines=['Published numeric-cell verification (sample standard deviation, ddof=1)',
           'Table | Row | Column | Published | Recomputed | Result']
    for c in checks:
        lines.append(f'{c.table} | {c.row} | {c.column} | {c.printed} | {c.actual} | '+('PASS' if c.passed else 'FAIL'))
    reasons=sorted({c.reason for c in checks if c.reason})
    lines.extend(['','Missing or invalid evidence:']+reasons if reasons else [])
    passed=sum(c.passed for c in checks)
    lines.append(f'\n{passed}/{len(checks)} numeric cells PASS; {len(checks)-passed} FAIL')
    return '\n'.join(lines)+'\n'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--published',type=Path,default=config.PUBLISHED_TABLES,
                        help='Frozen published-cell JSON, or a LaTeX source to parse instead.')
    parser.add_argument('--reference',type=Path,default=config.REFERENCE_DIR)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    checks=verify(args.published,args.reference)
    report=format_report(checks)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(report,encoding='utf-8')
    return 0 if checks and all(c.passed for c in checks) else 1

if __name__=='__main__':
    raise SystemExit(main())
