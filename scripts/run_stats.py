"""Write paired statistics from reference results or a matching run tree."""
import argparse
import json
from pathlib import Path
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qnnbench import config
from qnnbench.stats import benchmark_tests, noise_aware_tests, capacity_tests, derived_statistics, latex_wilcoxon

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',type=int,nargs='+',choices=config.SEEDS,default=config.SEEDS)
    parser.add_argument('--sizes',type=int,nargs='+',choices=config.TRAIN_SIZES,default=config.TRAIN_SIZES)
    parser.add_argument('--reference',type=Path,default=config.REFERENCE_DIR)
    parser.add_argument('--out',type=Path,default=config.ROOT/'results/runs/stats')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    benchmark=benchmark_tests(args.reference,args.seeds,args.sizes)
    lines=['Two-sided Wilcoxon, method="auto"; paired by seed and training size.',
           'The full benchmark has n=40; floor 2/2^40 = 1.8189894035458565e-12.',
           'Noise-aware/capacity have n=10; floor 2/2^10 = 0.001953125.',
           f'Requested seeds: {args.seeds}; sizes: {args.sizes}',benchmark.to_string(index=False)]
    failed=False
    for name,fn in [('Noise-aware',noise_aware_tests),('Capacity',capacity_tests)]:
        try: lines.extend(['',name,fn(args.reference,args.seeds).to_string(index=False)])
        except FileNotFoundError as exc:
            failed=True; lines.append(f'\nMISSING {name}: {exc.filename}')
    lines.extend(['','Derived (requested seed/size population):',json.dumps(derived_statistics(args.reference,args.seeds,args.sizes),indent=2)])
    (args.out/'stats_summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (args.out/'table_wilcoxon.tex').write_text(latex_wilcoxon(benchmark),encoding='utf-8')
    print(f'Wrote statistics to {args.out.resolve()}')
    return int(failed)

if __name__=='__main__':
    raise SystemExit(main())
