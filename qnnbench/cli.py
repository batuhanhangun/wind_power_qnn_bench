"""Local entry points around the original experiment functions."""
import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import time
from qnnbench import config

def prepare_threads():
    # Copied from ann_regularized_baseline/run_local.py, before scientific imports.
    os.environ['OMP_NUM_THREADS']='1'
    os.environ['MKL_NUM_THREADS']='1'
    os.environ['OPENBLAS_NUM_THREADS']='1'

def parser_for(name):
    parser=argparse.ArgumentParser(description=f'Run the published {name} experiment locally.')
    parser.add_argument('--seeds',nargs='+',type=int,choices=config.SEEDS,default=config.SEEDS)
    if name in ['classical','qnn','noise','ann_reg']:
        parser.add_argument('--sizes',nargs='+',type=int,choices=config.TRAIN_SIZES,default=config.TRAIN_SIZES)
    parser.add_argument('--out',type=Path,default=config.ROOT/'results/runs'/name)
    if name=='classical':
        parser.add_argument('--models',nargs='+',choices=config.CLASSICAL_MODELS,default=config.CLASSICAL_MODELS)
    if name=='noise':
        parser.add_argument('--qnn-results',type=Path,default=config.ROOT/'results/runs/qnn')
    return parser

def aggregate(name, out, seeds, sizes):
    import pandas as pd
    module=importlib.import_module('qnnbench._aggregation.'+('qnn' if name=='noise' else name))
    if name=='classical':
        rows=module.collect_results()
        df=pd.DataFrame(rows)
        df['generalization_gap']=df.train_r2-df.test_r2
        df['model']=df.model.replace({'ann':'ANN','svr':'SVR','dtr':'DTR','rf':'RF','xgboost':'XGBoost'})
        df.to_csv(out/'classical_per_seed.csv',index=False)
        module.aggregate_by_model_and_size(rows).to_csv(out/'summary_table.csv',index=False)
    elif name=='qnn':
        rows=module.collect_stage1()
        pd.DataFrame(rows).to_csv(out/'qnn_per_seed.csv',index=False)
        module.build_performance_summary(rows).to_csv(out/'qnn_performance_summary.csv',index=False)
    elif name=='noise':
        rows,handoff=module.collect_stage2()
        expected=len(seeds)*len(sizes)*len(config.NOISE_LEVELS)
        if len(rows)!=expected: raise ValueError(f'Expected {expected} noise rows; found {len(rows)}')
        pd.DataFrame(rows).to_csv(out/'noise_per_run.csv',index=False)
        summary,_=module.build_noise_summary(rows)
        summary.to_csv(out/'noise_summary.csv',index=False)
    elif name=='selection':
        df=pd.DataFrame(module.collect_results())
        df.to_csv(out/'selection_per_seed.csv',index=False)
        summary=module.build_summary(df)
        summary.to_csv(out/'selection_summary.csv',index=False)
        _,lines=module.decide(summary)
        (out/'selection_decision.txt').write_text(lines+'\n',encoding='utf-8')
    elif name=='capacity':
        df=pd.DataFrame(module.collect_rows())
        df.to_csv(out/'capacity_per_seed.csv',index=False)
        summary=module.build_summary(df)
        summary.to_csv(out/'capacity_summary.csv',index=False)
        module.write_report(out/'capacity_report.txt',summary,df)
    elif name=='noise_aware':
        rows,noise_free=module.collect()
        df=pd.DataFrame(rows)
        df.to_csv(out/'noise_aware_per_seed.csv',index=False)
        summary=module.build_summary(df)
        summary.to_csv(out/'noise_aware_summary.csv',index=False)
        head=module.paired_head_to_head(df)
        # Original fidelity gates are defined on the ten-seed mean only.
        if len(seeds)==10:
            module.write_report(out/'noise_aware_report.txt',summary,head,noise_free)
            ok_nf,_,_=module.gate_noise_free(noise_free)
            ok_fe,_=module.gate_fidelity(head)
            if not (ok_nf and ok_fe): raise RuntimeError('Original noise-aware fidelity gate failed')
        else:
            (out/'noise_aware_report.txt').write_text(
                'Explicit seed subset: ten-seed mean fidelity anchors do not apply.\n',encoding='utf-8')
    elif name=='ann_reg':
        from qnnbench.stats import validate_keys
        df=module.collect_results()
        expected=[(s,n) for s in seeds for n in sizes]
        validate_keys(df,['seed','train_size'],expected)
        ref=pd.read_csv(config.REFERENCE_CSV)
        ref=ref[ref.seed.isin(seeds)&ref.train_size.isin(sizes)]
        # The original gate reads a CSV; restrict its I/O input to the explicitly requested cells.
        selected=out/'selected_reference.csv'
        ref.to_csv(selected,index=False)
        original=config.REFERENCE_CSV
        try:
            config.REFERENCE_CSV=selected
            passed,lines=module.run_reproduction_gate(df)
        finally:
            config.REFERENCE_CSV=original
        (out/'reproduction_report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        if not passed: raise RuntimeError('ANN-Reg reproduction gate failed; inspect reproduction_report.txt')
        df.to_csv(out/'ann_reg_per_seed.csv',index=False)
        df[['model','seed','train_size','test_r2','train_r2','generalization_gap']].to_csv(out/'ann_reg_classical_per_seed.csv',index=False)
        module.build_summary(df).to_csv(out/'ann_reg_summary.csv',index=False)

def run(name, argv=None):
    prepare_threads()
    args=parser_for(name).parse_args(argv)
    if len(args.seeds)!=len(set(args.seeds)): raise ValueError('Duplicate seeds')
    sizes=getattr(args,'sizes',[800])
    if len(sizes)!=len(set(sizes)): raise ValueError('Duplicate sizes')
    out=args.out.resolve()
    if out==config.REFERENCE_DIR or config.REFERENCE_DIR in out.parents:
        raise ValueError('Experiment outputs must not overwrite shipped references')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {out}; choose a fresh --out')
    out.mkdir(parents=True,exist_ok=True)
    config.RESULTS_DIR=out
    config.AGGREGATED_DIR=out
    config.SEEDS=list(args.seeds)
    config.TRAIN_SIZES=list(sizes)
    if name=='noise':
        # Preserve the verified handoff logic; stage its two input files into --out.
        import shutil
        for s in args.seeds:
            for n in sizes:
                for suffix in ['weights.npy','metrics.json']:
                    relative=Path(f'seed_{s}/exp_{n}_{suffix}')
                    (out/relative).parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(args.qnn_results/relative,out/relative)
    module=importlib.import_module('qnnbench.'+name)
    started=time.perf_counter()
    record={'experiment':name,'seeds':args.seeds,'sizes':sizes,'python':platform.python_version(),
            'platform':platform.platform(),'processor':platform.processor(),'status':'running'}
    try:
        for seed in args.seeds:
            if name=='selection':
                for topology in config.CONFIGS: module.run_config_seed(topology,seed)
            elif name=='capacity':
                for reps in config.REPS_GRID: module.run_experiment(reps,seed)
            elif name=='noise_aware': module.run_seed(seed)
            else:
                for size in sizes:
                    if name=='classical':
                        for model in args.models: module.run_experiment(model,size,seed)
                    else: module.run_experiment(size,seed)
        aggregate(name,out,args.seeds,sizes)
        record['status']='passed'
    except BaseException as exc:
        record.update(status='failed',error=str(exc))
        raise
    finally:
        record['elapsed_seconds']=time.perf_counter()-started
        (out/'runtime.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        print(f'Wrote {name} outputs to {out} ({record["elapsed_seconds"]:.1f} seconds; {record["status"]})',flush=True)
