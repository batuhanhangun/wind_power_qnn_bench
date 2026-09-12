"""Run complete local experiments in dependency order, preserving requested seeds."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qnnbench import config

ORDER=['classical','qnn','noise','selection','capacity','ann_reg','noise_aware','stats']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',type=int,nargs='+',choices=config.SEEDS,default=config.SEEDS)
    parser.add_argument('--sizes',type=int,nargs='+',choices=config.TRAIN_SIZES,default=config.TRAIN_SIZES)
    parser.add_argument('--out',type=Path,default=config.ROOT/'results/runs/all')
    args=parser.parse_args()
    timing_path=config.ROOT/'runtime_measurements.json'
    measured=json.loads(timing_path.read_text()) if timing_path.exists() else {}
    total=0
    missing=[]
    for name in ORDER[:-1]:
        entry=measured.get(name)
        if not entry or entry.get('status')!='passed':
            missing.append(name); print(f'{name}: runtime not measured',flush=True)
        else:
            seconds=entry['elapsed_seconds']*len(args.seeds)
            total+=seconds
            print(f'{name}: {seconds/3600:.2f} hours, extrapolated from one seed on the documented machine',flush=True)
    if missing:
        print('Total runtime: not measured; no defensible complete estimate is available. '
              'The full density-matrix noise-aware training is included.',flush=True)
    else: print(f'Total estimated training runtime: {total/3600:.2f} hours.',flush=True)
    if 3200 not in args.sizes:
        parser.error('run_all includes Figure 4 and requires size 3200; use individual experiment scripts for other subsets')
    if args.out.exists() and any(args.out.iterdir()): raise FileExistsError('Choose an empty --out directory')
    args.out.mkdir(parents=True,exist_ok=True)
    seeds=['--seeds',*map(str,args.seeds)]
    sizes=['--sizes',*map(str,args.sizes)]
    for name in ORDER:
        command=[sys.executable,'-m','scripts.run_'+name,*seeds,'--out',str(args.out/name)]
        if name in ['classical','qnn','noise','ann_reg','stats']: command+=sizes
        if name=='noise': command+=['--qnn-results',str(args.out/'qnn')]
        if name=='stats': command+=['--reference',str(args.out)]
        print(' '.join(command),flush=True)
        subprocess.run(command,cwd=config.ROOT,check=True)
    subprocess.run([sys.executable,'-m','scripts.make_figures','--reference',str(args.out),
                    '--out',str(args.out/'figures'),*seeds,*sizes],cwd=config.ROOT,check=True)
    print(f'Wrote all experiments and figures to {args.out.resolve()}')

if __name__=='__main__':
    main()
