"""Train one final clustered model and apply it to coreArgo data.

For each cluster, tunes hyperparameters on --n-folds folds of the trainval data
using the algorithm and grid set in CLUSTER_MODELS below (or in a JSON file from
compare_algorithms.py, via --cluster-models), fits calibration
coefficients from out-of-fold predictions, then fits a final model on all of
that cluster's data. The 6 class models are applied to the application data
(weighted by cluster probabilities) to add cluster{k}_pred, weighted_pred,
cluster{k}_calpred and lincal_pred columns.

Starts a LocalCluster on the node Slurm allocated, same as run_dask_nested_cv.py.

Usage:
    python run_model_application.py
    python run_model_application.py --run-name crusoe_application_mixed --n-folds 5
    python run_model_application.py --cluster-models /pscratch/sd/s/sangsong/argoclare_runs/results/cmp20261015_bestAlgorithms.json
"""
import argparse
import json
import os
from datetime import datetime

import pandas as pd
from dask.distributed import LocalCluster, Client

import procedures_application
from tuning_grids import TUNING_GRIDS

# ==== Standard run parameters =======
n_clusters = 6 # SO classes
sea_ice_clusters = [5, 6]

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
default_data = '/global/cfs/cdirs/m4839/sangsong/github-repos/argo-class-regression/scripts/model-optimization/x_localdata/combined_trainval_pc8_gmm6_dbar500_seaiceclusters56_acc20261016.csv'
# coreArgo data to apply the final model to
default_application_data = '/global/cfs/cdirs/m4839/sangsong/github-repos/argo-class-regression/scripts/model-optimization/x_localdata/coreArgo_application_pc8_gmm6_dbar500_acc20261016.csv'
# Folder to save results to
default_savepath = '/pscratch/sd/s/sangsong/argoclare_runs/results/'


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
    parser.add_argument('--cluster-models', default=None,
                        help='JSON file of {cluster: [algorithm, grid]} from compare_algorithms.py '
                             '(default: CLUSTER_MODELS in this script)')
    parser.add_argument('--save-models', action='store_true',
                        help='Pickle the fitted final models with joblib (off by default)')
    return parser.parse_args()


def main():
    args = parse_args()
    savepath = os.path.join(args.savepath, '') # outfile is appended directly to savepath
    os.makedirs(savepath, exist_ok=True)

    # Algorithm and grid names for each cluster, from JSON if given
    if args.cluster_models:
        with open(args.cluster_models) as file:
            cluster_model_names = {int(classnum): tuple(choice) for classnum, choice in json.load(file).items()}
        cluster_models_source = os.path.abspath(args.cluster_models)
    else:
        cluster_model_names = CLUSTER_MODELS
        cluster_models_source = 'CLUSTER_MODELS in run_model_application.py'
    print(f'Cluster models from: {cluster_models_source}')

    # Check cluster setup before loading data
    missing_clusters = set(range(1, n_clusters + 1)) - set(cluster_model_names)
    if missing_clusters:
        raise ValueError(f'Cluster models are missing clusters {sorted(missing_clusters)}')
    for classnum, (algorithm, grid) in cluster_model_names.items():
        if algorithm not in ('RFR', 'ERT', 'XGB'):
            raise ValueError(f'Cluster {classnum}: unknown algorithm {algorithm}')
        if grid not in TUNING_GRIDS:
            raise ValueError(f'Cluster {classnum}: unknown grid {grid}, choose from {sorted(TUNING_GRIDS)}')
    cluster_models = {classnum: (algorithm, TUNING_GRIDS[grid])
                      for classnum, (algorithm, grid) in cluster_model_names.items()}

    trainval_data = pd.read_csv(args.data, index_col=0) # for platformID
    application_data = pd.read_csv(args.application_data, index_col=0)
    print(f'Loaded {len(trainval_data)} training samples from {args.data}')
    print(f'Loaded {len(application_data)} application samples from {args.application_data}')

    required_cols = feat_list + ['cluster' + str(k) + '_prob' for k in range(1, n_clusters + 1)]
    missing_cols = [x for x in required_cols if x not in application_data.columns]
    if missing_cols:
        raise ValueError(f'Application data is missing columns {missing_cols}')

    for classnum, (algorithm, grid) in sorted(cluster_model_names.items()):
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
                                                     save_models=args.save_models,
                                                     cluster_models_source=cluster_models_source)
        print('Finished model application at: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))


if __name__ == "__main__":
    main()
