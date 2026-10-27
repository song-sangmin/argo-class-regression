"""
Train one final clustered model and apply it to coreArgo data.

For each cluster, tunes hyperparameters on --n-folds.
Fits calibration coefficients from out-of-fold predictions before
fitting a final model on all of that cluster's data. 

The 6 class models are applied (weighted by cluster probabilities) 
to add cluster{k}_pred, weighted_pred, cluster{k}_calpred and lincal_pred.

With a test set (--test-data, or [files] reg_test with --config), the same final models are
evaluated on it: <run name>_testPredictions and _testMetrics (RMSE, MAE and bias, overall and
by cluster, uncalibrated and calibrated). The test set is not used for tuning or calibration.

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

from argoclare import pipeline

import procedures_application
from tuning_grids import TUNING_GRIDS

# ==== Standard run parameters =======
sea_ice_clusters = [5, 6] # default, change with --sea-ice-clusters (or from reg_info with --config)

feat_list = ['CT', 'SA', 'log_mld', 'adt', 'sea_ice',
             'wind_speed', 'atmos_pres_atm', 'max_solar_rad'] # 8 features, leave some out with --drop-features
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
    parser.add_argument('--n-folds', type=int, default=None,
                        help='Folds for hyperparameter tuning. Default: [regression] n_folds with --config, else 5')
    parser.add_argument('--cv-type', default=None, choices=['platform', 'profile', 'kmeans'],
                        help='Fold splits: whole floats/cruises (platform), whole profiles (profile), or '
                             'k-means regions (kmeans). Default: [regression] cv_type with --config, else platform')
    parser.add_argument('--scoring', default='median_abs_error', help='Tuning fold scoring error')
    parser.add_argument('--n-workers', type=int, default=64)
    parser.add_argument('--threads-per-worker', type=int, default=2)
    parser.add_argument('--data', default=None, help='Combined trainval csv (default: [files] reg_trainval with --config)')
    parser.add_argument('--sea-ice-clusters', type=int, nargs='+', default=None,
                        help='Clusters trained with sea_ice as a feature (from regression-input-prep.py)')
    parser.add_argument('--drop-features', nargs='+', default=[],
                        help='Features in feat_list to leave out (e.g. adt for synthetic data)')
    parser.add_argument('--application-data', default=None,
                        help='coreArgo application csv (default: [files] reg_application with --config)')
    parser.add_argument('--test-data', default=None,
                        help='Held-out test csv to evaluate the final models on (default: [files] reg_test with '
                             '--config, if regression-input-prep.py made it)')
    parser.add_argument('--no-test', action='store_true', help='Skip the test set, even if [files] reg_test exists')
    parser.add_argument('--savepath', default=None, help='Results folder (default: [files] model_results with --config)')
    parser.add_argument('--cluster-models', default=None,
                        help='JSON file of {cluster: [algorithm, grid]} from compare_algorithms.py '
                             '(default: CLUSTER_MODELS in this script)')
    parser.add_argument('--save-models', action='store_true',
                        help='Pickle the fitted final models with joblib (off by default)')
    pipeline.add_model_args(parser)
    args = parser.parse_args()
    pipeline.fill_model_args(args)
    args.data = args.data or default_data
    args.application_data = args.application_data or default_application_data
    args.savepath = args.savepath or default_savepath
    if args.sea_ice_clusters is None:
        args.sea_ice_clusters = sea_ice_clusters
    args.cv_type = args.cv_type or 'platform'
    args.n_folds = args.n_folds or 5
    return args


def main():
    args = parse_args()
    savepath = os.path.join(args.savepath, '') # outfile is appended directly to savepath
    os.makedirs(savepath, exist_ok=True)
    run_feat_list = [x for x in (getattr(args, 'features', None) or feat_list) if x not in args.drop_features]
    target = getattr(args, 'target', None) or var_predict

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
    n_clusters = int(pd.read_csv(args.data, index_col=0, nrows=0).columns.str.fullmatch(r'cluster\d+_prob').sum())
    if not args.cluster_models: # CLUSTER_MODELS for the clusters in the data; RFR rfr_full for any it does not list
        cluster_model_names = {k: CLUSTER_MODELS.get(k, ('RFR', 'rfr_full')) for k in range(1, n_clusters + 1)}
    cluster_model_names = {k: v for k, v in cluster_model_names.items() if k <= n_clusters}
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
    if args.cv_type == 'profile' and 'profid' not in trainval_data.columns:
        raise SystemExit(f'--cv-type profile needs a profid column in {args.data}: rerun '
                         'regression-input-prep.py with [regression] cv_type = "profile"')
    application_data = pd.read_csv(args.application_data, index_col=0)
    print(f'Loaded {len(trainval_data)} training samples from {args.data}')
    print(f'Loaded {len(application_data)} application samples from {args.application_data}')
    test_data = None
    if args.test_data and not args.no_test:
        test_data = pd.read_csv(args.test_data, index_col=0)
        print(f'Loaded {len(test_data)} test samples from {args.test_data} (not used for tuning or calibration)')
        test_cols = run_feat_list + [target] + ['cluster' + str(k) + '_prob' for k in range(1, n_clusters + 1)]
        missing_cols = [x for x in test_cols if x not in test_data.columns]
        if missing_cols:
            raise ValueError(f'Test data is missing columns {missing_cols}')
    else:
        print('No test set' + (' (--no-test)' if args.no_test else ''))

    required_cols = run_feat_list + ['cluster' + str(k) + '_prob' for k in range(1, n_clusters + 1)]
    missing_cols = [x for x in required_cols if x not in application_data.columns]
    if missing_cols:
        raise ValueError(f'Application data is missing columns {missing_cols}')

    for classnum, (algorithm, grid) in sorted(cluster_model_names.items()):
        print(f'Cluster {classnum}: {algorithm}, grid: {grid} = {TUNING_GRIDS[grid]}')
    print(f'Tuning folds: {args.n_folds}, run name: {args.run_name}')
    print(f'Features: {run_feat_list}, target: {target}, sea ice clusters: {args.sea_ice_clusters}')

    with LocalCluster(n_workers=args.n_workers,
                      threads_per_worker=args.threads_per_worker,
                      dashboard_address=None) as cluster, Client(cluster) as client:
        print(client)
        print('Starting dask training: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))
        procedures_application.run_model_application(client, trainval_data, application_data,
                                                     n_clusters, args.sea_ice_clusters,
                                                     run_feat_list, target,
                                                     args.n_folds,
                                                     cluster_models,
                                                     args.cv_type,
                                                     args.scoring,
                                                     savepath,
                                                     args.run_name,
                                                     save_models=args.save_models,
                                                     cluster_models_source=cluster_models_source,
                                                     test_data=test_data)
        print('Finished model application at: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))


if __name__ == "__main__":
    main()
