"""Train one final clustered model and apply it to coreArgo data.

For each cluster, tunes hyperparameters on --n-folds folds of the trainval data
using the algorithm and grid set in CLUSTER_MODELS below, fits calibration
coefficients from out-of-fold predictions, then fits a final model on all of
that cluster's data. The 6 class models are applied to the application data
(weighted by cluster probabilities) to add cluster{k}_pred, weighted_pred,
cluster{k}_calpred and lincal_pred columns.

Starts a LocalCluster on the node Slurm allocated, same as run_dask_nested_cv.py.

Usage:
    python run_model_application.py
    python run_model_application.py --run-name crusoe_application_mixed --n-folds 5
"""
import argparse
import os
from datetime import datetime

import pandas as pd
from dask.distributed import LocalCluster, Client

import procedures_application
from tuning_grids import TUNING_GRIDS

# ==== Standard run parameters =======
n_clusters = 6 # SO classes
sea_ice_clusters = [3, 5]

feat_list = ['CT', 'SA', 'log_mld', 'adt', 'sea_ice',
             'wind_speed', 'atmos_pres_atm', 'max_solar_rad'] # 8 features
var_predict = 'delta_pco2'

# ==== Algorithm and tuning grid (name in tuning_grids.py) for each cluster =======
CLUSTER_MODELS = {
    1: ('RFR', 'rfr_full'),
    2: ('RFR', 'rfr_full'),
    3: ('RFR', 'rfr_full'),
    4: ('RFR', 'rfr_full'),
    5: ('RFR', 'rfr_full'),
    6: ('RFR', 'rfr_full'),
}

# Combined training/validation data from SOCAT and BGC-Argo
default_data = '/global/u1/s/sangsong/dask-cv-src/combined_trainval_acc20260928_seaiceclusters35.csv'
# coreArgo data to apply the final model to
default_application_data = '/global/u1/s/sangsong/dask-cv-src/coreArgo_application.csv'
# Folder to save results to
default_savepath = '/pscratch/sd/s/sangsong/dask_cv/results/'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-name', default='crusoe_clusteredModel_application',
                        help='Output file prefix')
    parser.add_argument('--n-folds', type=int, default=5, help='Folds for hyperparameter tuning')
    parser.add_argument('--cv-type', default='platform', choices=['platform', 'kmeans'])
    parser.add_argument('--scoring', default='median_abs_error', help='Tuning fold scoring error')
    parser.add_argument('--n-workers', type=int, default=64)
    parser.add_argument('--threads-per-worker', type=int, default=2)
    parser.add_argument('--data', default=default_data)
    parser.add_argument('--application-data', default=default_application_data)
    parser.add_argument('--savepath', default=default_savepath)
    parser.add_argument('--save-models', action='store_true',
                        help='Pickle the fitted final models with joblib (off by default)')
    return parser.parse_args()


def main():
    args = parse_args()
    savepath = os.path.join(args.savepath, '') # outfile is appended directly to savepath
    os.makedirs(savepath, exist_ok=True)

    # Check cluster setup before loading data
    missing_clusters = set(range(1, n_clusters + 1)) - set(CLUSTER_MODELS)
    if missing_clusters:
        raise ValueError(f'CLUSTER_MODELS is missing clusters {sorted(missing_clusters)}')
    for classnum, (algorithm, grid) in CLUSTER_MODELS.items():
        if algorithm not in ('RFR', 'ERT', 'XGB'):
            raise ValueError(f'Cluster {classnum}: unknown algorithm {algorithm}')
        if grid not in TUNING_GRIDS:
            raise ValueError(f'Cluster {classnum}: unknown grid {grid}, choose from {sorted(TUNING_GRIDS)}')
    cluster_models = {classnum: (algorithm, TUNING_GRIDS[grid])
                      for classnum, (algorithm, grid) in CLUSTER_MODELS.items()}

    trainval_data = pd.read_csv(args.data, index_col=0) # for platformID
    application_data = pd.read_csv(args.application_data, index_col=0)
    print(f'Loaded {len(trainval_data)} training samples from {args.data}')
    print(f'Loaded {len(application_data)} application samples from {args.application_data}')

    required_cols = feat_list + ['cluster' + str(k) + '_prob' for k in range(1, n_clusters + 1)]
    missing_cols = [x for x in required_cols if x not in application_data.columns]
    if missing_cols:
        raise ValueError(f'Application data is missing columns {missing_cols}')

    for classnum, (algorithm, grid) in CLUSTER_MODELS.items():
        print(f'Cluster {classnum}: {algorithm}, grid: {grid} = {TUNING_GRIDS[grid]}')
    print(f'Tuning folds: {args.n_folds}, run name: {args.run_name}')

    with LocalCluster(n_workers=args.n_workers,
                      threads_per_worker=args.threads_per_worker,
                      dashboard_address=None) as cluster, Client(cluster) as client:
        print(client)
        print('Starting dask training: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))
        procedures_application.run_model_application(client, trainval_data, application_data,
                                                     n_clusters, sea_ice_clusters,
                                                     feat_list, var_predict,
                                                     args.n_folds,
                                                     cluster_models,
                                                     args.cv_type,
                                                     args.scoring,
                                                     savepath,
                                                     args.run_name,
                                                     save_models=args.save_models)
        print('Finished model application at: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))


if __name__ == "__main__":
    main()
