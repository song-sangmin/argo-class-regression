"""Nested CV hyperparameter tuning for one algorithm + tuning grid.

Batch version of dask-hyperparameter-tuning.ipynb. Instead of connecting to the
scheduler from launch-dask-conda-cpu.sh, it starts a LocalCluster on the node
Slurm allocated (64 workers x 2 threads by default, same as the interactive runs).

Usage (normally called by nested_cv.sbatch):
    python run_dask_nested_cv.py RFR rfr_full
    python run_dask_nested_cv.py XGB xgb_full --run-name crusoe_XGB_platformCV_fullgrid10x5
"""
import argparse
import os
from datetime import datetime

import pandas as pd
from dask.distributed import LocalCluster, Client

import procedures_inner
from tuning_grids import TUNING_GRIDS

# ==== Standard run parameters =======
n_clusters = 6 # SO classes
sea_ice_clusters = [3, 5]

feat_list = ['CT', 'SA', 'log_mld', 'adt', 'sea_ice_avg_7d',
             'wind_speed', 'atmos_pres_atm', 'max_solar_rad'] # 8 features
var_predict = 'delta_pco2'

# Combined training/validation data from SOCAT and BGC-Argo
default_data = '/global/u1/s/sangsong/dask-cv-src/combined_trainval_acc20260928_seaiceclusters35.csv'
# Folder to save results to
default_savepath = '/pscratch/sd/s/sangsong/dask_cv/results/'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('algorithm', choices=['RFR', 'ERT', 'XGB'])
    parser.add_argument('grid', choices=sorted(TUNING_GRIDS),
                        help='Name of tuning grid in tuning_grids.py')
    parser.add_argument('--run-name', default=None,
                        help='Output file prefix (default: crusoe_<ALGORITHM>_platformCV_<grid>)')
    parser.add_argument('--n-outer', type=int, default=10, help='Outer folds (error estimation)')
    parser.add_argument('--n-inner', type=int, default=5, help='Inner folds (tuning)')
    parser.add_argument('--cv-type', default='platform', choices=['platform', 'kmeans'])
    parser.add_argument('--scoring', default='median_abs_error', help='Inner fold scoring error')
    parser.add_argument('--n-workers', type=int, default=64)
    parser.add_argument('--threads-per-worker', type=int, default=2)
    parser.add_argument('--data', default=default_data)
    parser.add_argument('--savepath', default=default_savepath)
    return parser.parse_args()


def main():
    args = parse_args()
    tuning_grid = TUNING_GRIDS[args.grid]
    run_name = args.run_name or f'crusoe_{args.algorithm}_platformCV_{args.grid}'
    savepath = os.path.join(args.savepath, '') # outfile is appended directly to savepath
    os.makedirs(savepath, exist_ok=True)

    trainval_data = pd.read_csv(args.data, index_col=0) # for platformID
    print(f'Loaded {len(trainval_data)} samples from {args.data}')
    print(f'Algorithm: {args.algorithm}, grid: {args.grid} = {tuning_grid}')
    print(f'Folds: {args.n_outer} outer x {args.n_inner} inner, run name: {run_name}')

    with LocalCluster(n_workers=args.n_workers,
                      threads_per_worker=args.threads_per_worker,
                      dashboard_address=None) as cluster, Client(cluster) as client:
        print(client)
        print('Starting dask training: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))
        procedures_inner.run_dask_nested_cv(client, trainval_data,
                                            n_clusters, sea_ice_clusters,
                                            feat_list, var_predict,
                                            args.n_outer, args.n_inner,
                                            args.algorithm, tuning_grid,
                                            args.cv_type,
                                            args.scoring,
                                            savepath,
                                            run_name)
        print('Finished dask cross validation at: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))


if __name__ == "__main__":
    main()
