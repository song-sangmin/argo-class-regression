"""Create the regression inputs for the Perlmutter nested CV (model-optimization/): the combined
training/validation csv, and optionally the matching application csv.

Reads the P2 combined trainval parquet (clustering/apply_P2_clustering.py, or the synthetic
apply_P2_synthetic_clustering.py) and:
    1. Renames the class columns to what model-optimization expects: class_assign -> cluster,
       class{k}_prob -> cluster{k}_prob; adds platform_indexer ('wmoid' for floats,
       'cruiseid' for SOCAT), alongside platform and platform_id
    2. Finds the sea ice clusters: clusters with more than --min-ice-profiles float samples
       with sea_ice > --ice-threshold (as in the draft). These are the clusters that keep
       sea_ice as a feature in model-optimization (sea_ice_clusters in run_dask_nested_cv.py and
       run_model_application.py); the file name gets a tag such as seaiceclusters56.
    3. Drops rows missing any feature or the target (--target, default delta_pco2) and exact
       duplicates; prints counts by cluster and platform
Writes <outpath>/combined_trainval_<pcm_params>_seaiceclusters<..>_acc<datetag>.csv, indexed by
sample_key (platform_id + sample time to the minute; unique), the format read by
run_dask_nested_cv.py (pd.read_csv(..., index_col=0)).
With --application <P2 core application parquet>, also writes
<outpath>/coreArgo_application_<pcm_params>_acc<datetag>.csv with the same renames, for
run_model_application.py --application-data (rows with missing features are dropped).

Usage:
    python regression-input-prep.py                                   # real P2 trainval (importer)
    python regression-input-prep.py --application <P2_coreArgo_application...parquet>
    python regression-input-prep.py --trainval <synthetic P2 trainval> --drop-features adt \\
        --application <P2_synArgo_core_application...parquet> --tag synthetic
"""
import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import importer # noqa: E402

default_features = ['CT', 'SA', 'log_mld', 'adt', 'sea_ice', 'wind_speed', 'atmos_pres_atm', 'max_solar_rad']
float_platforms = ['bgcArgo', 'synBgcArgo']
keep_cols = ['platform', 'platform_id', 'platform_indexer', 'datetime', 'year', 'month', 'day', 'linear_time',
             'latitude', 'longitude', 'CT', 'SA', 'mld', 'log_mld', 'adt', 'sea_ice', 'wind_speed', 'max_solar_rad',
             'atmos_pres_Pa', 'atmos_pres_atm', 'atmos_co2_ppm', 'vapor_pres_atm', 'pco2_ocean', 'pco2_atmos',
             'delta_pco2', 'cluster']


def rename_classes(df):
    """class_assign -> cluster, class{k}_prob -> cluster{k}_prob; returns df and the prob columns."""
    df = df.rename(columns={'class_assign': 'cluster'})
    df = df.rename(columns={c: 'cluster' + c[5:] for c in df.columns if c.startswith('class') and c.endswith('_prob')})
    prob_cols = sorted([c for c in df.columns if c.startswith('cluster') and c.endswith('_prob')],
                       key=lambda c: int(c[7:-5]))
    return df, prob_cols


def add_sample_key(df):
    """Unique row key: platform_id + sample time to the minute (+ a counter if ever repeated)."""
    key = df['platform_id'].astype(str) + '_' + pd.to_datetime(df['datetime']).dt.strftime('%Y%m%d%H%M')
    dup = key.groupby(key).cumcount()
    df.index = np.where(dup > 0, key + '_' + dup.astype(str), key)
    df.index.name = 'sample_key'
    return df


def find_sea_ice_clusters(df, n_clusters, threshold, min_profiles):
    """Clusters with more than min_profiles float samples with sea_ice > threshold."""
    floats = df[df['platform'].isin(float_platforms)]
    clusters = []
    print('\ncluster  float_samples  with_sea_ice  proportion')
    for k in range(1, n_clusters + 1):
        cdata = floats[floats['cluster'] == k]
        n_ice = int((cdata['sea_ice'] > threshold).sum())
        print(f'{k:7d}  {len(cdata):13d}  {n_ice:12d}  {n_ice / max(len(cdata), 1):10.3f}')
        if n_ice > min_profiles:
            clusters.append(k)
    print('Sea ice clusters:', clusters)
    return clusters


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--trainval', default=importer.path_to('P2', 'trainval'), help='P2 combined trainval parquet')
    parser.add_argument('--application', default=None, help='P2 core application parquet (optional)')
    parser.add_argument('--features', nargs='+', default=default_features, help='Regression features')
    parser.add_argument('--drop-features', nargs='+', default=[], help='Features to leave out (e.g. adt for synthetic)')
    parser.add_argument('--target', default='delta_pco2')
    parser.add_argument('--ice-threshold', type=float, default=0.1, help='sea_ice above this counts as an ice sample')
    parser.add_argument('--min-ice-profiles', type=int, default=10, help='Ice samples needed to call a cluster a sea ice cluster')
    parser.add_argument('--tag', default='', help='Extra tag for the file names (e.g. synthetic)')
    parser.add_argument('--outpath', default=importer.local_working_folder + 'regression-inputs/')
    args = parser.parse_args()

    feat_list = [f for f in args.features if f not in args.drop_features]
    trainval = pd.read_parquet(args.trainval)
    pcm_params = trainval.attrs.get('pcm_params', 'pcm')
    trainval, prob_cols = rename_classes(trainval)
    n_clusters = len(prob_cols)
    trainval['platform_indexer'] = np.where(trainval['platform'].isin(float_platforms), 'wmoid', 'cruiseid')
    print(f'{os.path.basename(args.trainval)}: {len(trainval)} samples, {n_clusters} clusters ({pcm_params})')
    print('Features:', feat_list, '| target:', args.target)

    sea_ice_clusters = find_sea_ice_clusters(trainval, n_clusters, args.ice_threshold, args.min_ice_profiles)

    n = len(trainval)
    cols = [c for c in keep_cols if c in trainval.columns] + prob_cols
    trainval = trainval.dropna(subset=feat_list + [args.target])[cols].drop_duplicates()
    print(f'\nDropped {n - len(trainval)} samples missing a feature or {args.target} (or duplicated); {len(trainval)} left')
    trainval = add_sample_key(trainval)
    counts = trainval.groupby(['cluster', 'platform']).size().unstack(fill_value=0)
    print(f'\nSamples by cluster and platform:\n{counts.to_string()}')

    os.makedirs(args.outpath, exist_ok=True)
    datetag = datetime.now().strftime('%Y%m%d')
    tag = f'_{args.tag}' if args.tag else ''
    ice_tag = 'seaiceclusters' + ''.join(str(k) for k in sea_ice_clusters)
    savefile = os.path.join(args.outpath, f'combined_trainval{tag}_{pcm_params}_{ice_tag}_acc{datetag}.csv')
    trainval.to_csv(savefile)
    print(f'\nSaved {len(trainval)} samples to {savefile}')
    print(f'Set in model-optimization/run_*.py: sea_ice_clusters = {sea_ice_clusters}; feat_list = {feat_list}')

    if args.application:
        app = pd.read_parquet(args.application)
        app, app_probs = rename_classes(app)
        assert app_probs == prob_cols, 'application and trainval have different clusters'
        app['platform_indexer'] = 'wmoid'
        n = len(app)
        cols = ['profid'] + [c for c in keep_cols if c in app.columns] + prob_cols
        app = app.dropna(subset=feat_list)[cols].set_index('profid')
        appfile = os.path.join(args.outpath, f'coreArgo_application{tag}_{pcm_params}_acc{datetag}.csv')
        app.to_csv(appfile)
        print(f'Saved {len(app)} application profiles ({n - len(app)} dropped for missing features) to {appfile}')


if __name__ == '__main__':
    main()
