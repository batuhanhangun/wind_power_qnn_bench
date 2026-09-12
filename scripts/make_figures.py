"""Check inputs and invoke the five original figure scripts."""
import argparse
from pathlib import Path
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'figures')
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(42,52)),choices=list(range(42,52)))
    parser.add_argument('--sizes',type=int,nargs='+',default=[800,1600,2400,3200],choices=[800,1600,2400,3200])
    args=parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    from figures import style,load_data
    load_data.SEEDS=args.seeds
    load_data.SIZES=args.sizes
    if args.reference:
        load_data.DATA=args.reference.resolve()
        names={'QNN':'qnn','ANN':'ann','ANN-Reg':'ann_reg','SVR':'svr','DTR':'dtr','XGBoost':'xgboost','RF':'rf'}
        load_data.PRED_DIRS={k:load_data.DATA/'predictions'/v for k,v in names.items()}
        if not (load_data.DATA/'predictions').exists():
            load_data.PRED_DIRS={k:load_data.DATA/(v if v in ['qnn','ann_reg'] else 'classical/'+v) for k,v in names.items()}
    style.OUTPUT_DIR=args.out.resolve()
    style.OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    load_data.load_all_per_seed()
    load_data.load_noise_summary()
    _,missing=load_data.load_predictions_3200()
    if missing: raise FileNotFoundError('\n'.join(missing))
    from figures import fig1_circuit,fig2_saturation,fig3_gap,fig4_scatter,fig5_noise
    for module in [fig1_circuit,fig2_saturation,fig3_gap,fig4_scatter,fig5_noise]:
        module.main()
    print(f'Wrote all five figures to {style.OUTPUT_DIR}')

if __name__=='__main__':
    main()
