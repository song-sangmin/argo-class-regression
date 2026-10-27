"""
Nested CV hyperparameter tuning for one algorithm + tuning grid.

Batch version of dask-hyperparameter-tuning.ipynb. 
64 workers x 2 threads by default.

Usage (normally called by submit_nested_cv.sbatch):
    python run_dask_nested_cv.py RFR rfr_full
    python run_dask_nested_cv.py XGB xgb_full --run-name crusoe_XGB_platformCV_fullgrid10x5
"""
import argparse
import os
from datetime import datetime

import pandas as pd
from dask.distributed import LocalCluster, Client

from argoclare import pipeline

import procedures_cv
from tuning_grids import TUNING_GRIDS

# ==== Standard run parameters =======
sea_ice_clusters = [5, 6] # default, change with --sea-ice-clusters (or from reg_info with --config)

feat_list = ['CT', 'SA', 'log_mld', 'adt', 'sea_ice',
             'wind_speed', 'atmos_pres_atm', 'max_solar_rad'] # 8 features, leave some out with --drop-features
var_predict = 'delta_pco2'

# Combined training/validation data from SOCAT and BGC-Argo
default_data = '/global/cfs/cdirs/m4839/sangsong/github-repos/argo-class-regression/scripts/model-optimization/x_localdata/combined_trainval_pc8_gmm6_dbar500_seaiceclusters56_acc20261016.csv'
# Folder to save results to
default_savepath = '/pscratch/sd/s/sangsong/argoclare_runs/results/'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('algorithm', choices=['RFR', 'ERT', 'XGB'])
    parser.add_argument('grid', choices=sorted(TUNING_GRIDS),
                        help='Name of tuning grid in tuning_grids.py')
    parser.add_argument('--run-name', default=None,
                        help='Output file prefix (default: crusoe_<ALGORITHM>_<cv_type>CV_grid_<grid>)')
    parser.add_argument('--n-outer', type=int, default=None,
                        help='Outer folds (error estimation). Default: [regression] n_outer with --config, else 10')
    parser.add_argument('--n-inner', type=int, default=None,
                        help='Inner folds (tuning). Default: [regression] n_inner with --config, else 5')
    parser.add_argument('--cv-type', default=None, choices=['platform', 'profile', 'kmeans'],
                        help='Fold splits: whole floats/cruises (platform), whole profiles (profile), or '
                             'k-means regions (kmeans). Default: [regression] cv_type with --config, else platform')
    parser.add_argument('--scoring', default='median_abs_error', help='Inner fold scoring error')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for platform fold splits; use the same seed to compare algorithms on identical folds')
    parser.add_argument('--n-workers', type=int, default=None,
                        help='Dask LocalCluster workers (default: chosen by dask from available CPUs)')
    parser.add_argument('--threads-per-worker', type=int, default=None,
                        help='Threads per dask worker (default: chosen by dask from available CPUs)')
    parser.add_argument('--data', default=None, help='Combined trainval csv (default: [files] reg_trainval with --config)')
    parser.add_argument('--sea-ice-clusters', type=int, nargs='+', default=None,
                        help='Clusters trained with sea_ice as a feature (from regression-input-prep.py)')
    parser.add_argument('--drop-features', nargs='+', default=[],
                        help='Features in feat_list to leave out (e.g. adt for synthetic data)')
    parser.add_argument('--savepath', default=None, help='Results folder (default: [files] model_results with --config)')
    pipeline.add_model_args(parser)
    args = parser.parse_args()
    pipeline.fill_model_args(args)
    args.data = args.data or default_data
    args.savepath = args.savepath or default_savepath
    if args.sea_ice_clusters is None:
        args.sea_ice_clusters = sea_ice_clusters
    args.cv_type = args.cv_type or 'platform'
    args.n_outer = args.n_outer or 10
    args.n_inner = args.n_inner or 5
    return args


def main():
    args = parse_args()
    tuning_grid = TUNING_GRIDS[args.grid]
    run_name = args.run_name or f'crusoe_{args.algorithm}_{args.cv_type}CV_grid_{args.grid}'
    savepath = os.path.join(args.savepath, '') # outfile is appended directly to savepath
    os.makedirs(savepath, exist_ok=True)
    run_feat_list = [x for x in (getattr(args, 'features', None) or feat_list) if x not in args.drop_features]
    target = getattr(args, 'target', None) or var_predict

    trainval_data = pd.read_csv(args.data, index_col=0) # for platformID
    if args.cv_type == 'profile' and 'profid' not in trainval_data.columns:
        raise SystemExit(f'--cv-type profile needs a profid column in {args.data}: rerun '
                         'regression-input-prep.py with [regression] cv_type = "profile"')
    n_clusters = int(trainval_data.columns.str.fullmatch(r'cluster\d+_prob').sum()) # GMM classes
    print(f'Loaded {len(trainval_data)} samples from {args.data} ({n_clusters} clusters)')
    print(f'Algorithm: {args.algorithm}, grid: {args.grid} = {tuning_grid}')
    print(f'Folds: {args.n_outer} outer x {args.n_inner} inner, seed: {args.seed}, run name: {run_name}')
    print(f'Features: {run_feat_list}, target: {target}, sea ice clusters: {args.sea_ice_clusters}')

    with LocalCluster(dashboard_address=None) as cluster, Client(cluster) as client:
        print(client)
        print('Starting dask training: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))
        procedures_cv.run_dask_nested_cv(client, trainval_data,
                                            n_clusters, args.sea_ice_clusters,
                                            run_feat_list, target,
                                            args.n_outer, args.n_inner,
                                            args.algorithm, tuning_grid,
                                            args.cv_type,
                                            args.scoring,
                                            savepath,
                                            run_name,
                                            seed=args.seed)
        print('Finished dask cross validation at: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))


if __name__ == "__main__":
    main()
