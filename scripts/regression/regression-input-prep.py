"""Create the regression inputs for model-optimization/: the combined training/validation csv
and the application csv, from the P2 datasets of one clustering.

Reads the [regression] train P2 datasets (CRUSOE: soccom, socat; from clustering/apply_P2_classes.py)
and the application P2 dataset (core), and:
    1. Combines the train datasets (in the order given) and splits them by year into
       trainval_years and test_years, written as [files] P2_trainval and P2_test (parquet,
       class columns as in P2)
    2. Application set: with exclude_training, drops application profiles that are float
       training samples (the own profid of a bgcArgo/synBgcArgo sample in the trainval or
       test set), so it does not overlap the training data; written as [files] P2_application
    3. Renames the class columns to what model-optimization expects: class_assign -> cluster,
       class{k}_prob -> cluster{k}_prob; adds platform_indexer ('wmoid' for floats,
       'cruiseid' for SOCAT), alongside platform and platform_id
    4. Finds the sea ice clusters: clusters with more than min_ice_profiles float samples
       with sea_ice > ice_threshold (as in the draft). These are the clusters that keep
       sea_ice as a feature in model-optimization (none without sea_ice in P1); they are saved in [files] reg_info
       (json, read by run_dask_nested_cv.py --config) and in {ice_tag} (e.g. seaiceclusters56),
       which [files] reg_trainval may use in its name.
    5. Drops trainval rows missing any feature or the target and exact duplicates, and
       application rows missing a feature; with target_qc_flags, also drops trainval rows whose
       <target>_qc flag is not one of them (e.g. oxygen, which L1 keeps without QC filtering);
       prints counts by cluster and platform. The csvs keep the columns model-optimization
       needs, plus the features and target, and with [regression] cv_type = "profile" a profid
       column for model-optimization's whole-profile folds.
Writes [files] reg_trainval (indexed by sample_key: platform_id + sample time to the minute;
unique), the format read by run_dask_nested_cv.py (pd.read_csv(..., index_col=0)), with
test_years the same for [files] reg_test (run_model_application.py tests the final models on it),
and [files] reg_application (indexed by profid) for run_model_application.py --application-data.
With --from-splits, steps 1-2 are skipped: the csvs are made from the existing [files] P2_trainval,
P2_test and P2_application, and csvs that already exist are kept (e.g. to add a missing reg_test).

Usage:
    python regression-input-prep.py --config ../../configs/southernOcean_CRUSOE.toml
    python regression-input-prep.py --config ../../configs/southernOcean_CRUSOE.toml --n-gmm 7
    python regression-input-prep.py --config ../../configs/synthetic_CRUSOE.toml
    python regression-input-prep.py --config ../../configs/southernOcean_CRUSOE.toml --from-splits
"""
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from argoclare import pipeline

float_platforms = ['bgcArgo', 'synBgcArgo']
class_cols = ['class_assign', 'class_profid', 'class_source', 'class_dt_days']
keep_cols = ['platform', 'platform_id', 'platform_indexer', 'datetime', 'year', 'month', 'day', 'linear_time',
             'latitude', 'longitude', 'CT', 'SA', 'mld', 'log_mld', 'adt', 'sea_ice', 'wind_speed', 'max_solar_rad',
             'atmos_pres_Pa', 'atmos_pres_atm', 'atmos_co2_ppm', 'vapor_pres_atm', 'pco2_ocean', 'pco2_atmos',
             'delta_pco2', 'cluster']


def save_parquet(df, path, attrs):
    """Save as clustering/p2_classes.save_parquet() does (text columns as strings, '' for missing)."""
    df = df.reset_index(drop=True).copy()
    for c in [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c].dtype)]:
        df[c] = df[c].where(df[c].notna(), '').astype(str)
    df.columns = [str(c).replace(' ', '_') for c in df.columns]
    df.attrs = json.loads(json.dumps(attrs, default=str))
    df.to_parquet(path, index=False)


def prob_columns(df, prefix='class'):
    return sorted([c for c in df.columns if c.startswith(prefix) and c.endswith('_prob')],
                  key=lambda c: int(c[len(prefix):-5]))


def reorder(df, lead):
    lead = [c for c in lead if c in df.columns]
    return df[lead + [c for c in df.columns if c not in lead]]


def with_columns(cols, extra):
    """cols, then any of extra not already in it."""
    return cols + [c for c in dict.fromkeys(extra) if c not in cols]


def rename_classes(df):
    """class_assign -> cluster, class{k}_prob -> cluster{k}_prob; returns df and the prob columns."""
    df = df.rename(columns={'class_assign': 'cluster'})
    df = df.rename(columns={c: 'cluster' + c[5:] for c in df.columns if c.startswith('class') and c.endswith('_prob')})
    return df, prob_columns(df, 'cluster')


def add_sample_key(df):
    """Unique row key: platform_id + sample time to the minute (+ a counter if ever repeated)."""
    key = df['platform_id'].astype(str) + '_' + pd.to_datetime(df['datetime']).dt.strftime('%Y%m%d%H%M')
    dup = key.groupby(key).cumcount()
    df.index = np.where(dup > 0, key + '_' + dup.astype(str), key)
    df.index.name = 'sample_key'
    return df


def regression_table(df, cfg, feat_list, target, label):
    """A P2 split (trainval or test) as the model-optimization csv: class columns renamed,
    platform_indexer (and profid with cv_type = "profile"), target QC flags applied, rows missing
    a feature or the target and duplicates dropped, indexed by sample_key."""
    df, prob_cols = rename_classes(df)
    df['platform_indexer'] = np.where(df['platform'].isin(float_platforms), 'wmoid', 'cruiseid')
    group_cols = []
    if cfg.get('cv_type') == 'profile':
        # model-optimization folds of whole profiles group by profid; samples without one
        # (ship data, SOCCOM samples without a matched profile) are their own group
        pid = df['profid'].astype(str) if 'profid' in df.columns else pd.Series('', index=df.index)
        df['profid'] = pid.where(~pid.isin(['', 'nan', 'None']), df['sample_id'].astype(str))
        group_cols = ['profid']
    n = len(df)
    if cfg.get('target_qc_flags'):
        qc = df[f'{target}_qc'].astype(str).str.replace(r'\.0$', '', regex=True)
        bad = ~qc.isin([str(x) for x in cfg['target_qc_flags']])
        print(f'\n{label}: dropping {int(bad.sum())} samples with {target}_qc not in {cfg["target_qc_flags"]}')
        df = df[~bad]
        n = len(df)
    cols = with_columns([c for c in keep_cols if c in df.columns], feat_list + [target] + group_cols) + prob_cols
    df = df.dropna(subset=feat_list + [target])[cols].drop_duplicates()
    print(f'\n{label}: dropped {n - len(df)} samples missing a feature or {target} (or duplicated); {len(df)} left')
    df = add_sample_key(df)
    counts = df.groupby(['cluster', 'platform']).size().unstack(fill_value=0)
    print(f'\n{label} samples by cluster and platform:\n{counts.to_string()}')
    return df, prob_cols


def check_columns(df, cols, label):
    """Stop if the features or target are not in a P2 dataset (P1 did not compute them)."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f'{label} has no {", ".join(missing)}: add {"it" if len(missing) == 1 else "them"} to [P1] features (or '
                         '[regression] features), then rerun P1, apply_P2_classes.py and this script')


def find_sea_ice_clusters(df, n_clusters, threshold, min_profiles):
    """Clusters with more than min_profiles float samples with sea_ice > threshold."""
    if 'sea_ice' not in df.columns:
        print('\nSea ice clusters: none (no sea_ice in the P2 data)')
        return []
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
    parser = pipeline.add_pcm_args(pipeline.parser(__doc__))
    parser.add_argument('--from-splits', action='store_true',
                        help='Make the csvs from the existing [files] P2_trainval, P2_test and P2_application '
                             'instead of the train P2 datasets; csvs that already exist are kept (unless --overwrite)')
    args = parser.parse_args()
    run = pipeline.start(args, __file__)
    cfg = run.section('regression')
    feat_list = [f for f in cfg['features'] if f not in cfg.get('drop_features', [])]
    target = cfg.get('target', 'delta_pco2')
    splits = [('trainval', cfg['trainval_years'])] + ([('test', cfg['test_years'])] if cfg.get('test_years') else [])

    if args.from_splits:
        # ===== 1-2. Existing year splits and application set
        split_files = {name: run.input(f'P2_{name}') for name, _ in splits}
        split_dfs = {name: pd.read_parquet(f) for name, f in split_files.items()}
        app_parquet = run.input('P2_application') if cfg.get('application') else None
        p2_attrs = split_dfs['trainval'].attrs
        pcm_params = p2_attrs.get('pcm_params', run.pcm)
        prob_cols = prob_columns(split_dfs['trainval'])
        sources = os.path.basename(split_files['trainval'])
        check_columns(split_dfs['trainval'], feat_list + [target], sources)
    else:
        train_files = [run.input('P2', ds) for ds in cfg['train']]
        app_file = run.input('P2', cfg['application']) if cfg.get('application') else None

        # ===== 1. Training data: combine the train datasets, split by year
        parts = [pd.read_parquet(f) for f in train_files]
        p2_attrs = parts[0].attrs
        pcm_params = p2_attrs.get('pcm_params', run.pcm)
        combined = pd.concat(parts, ignore_index=True)
        combined['year'] = combined['datetime'].dt.year
        prob_cols = prob_columns(combined)
        sources = ', '.join(os.path.basename(f) for f in train_files)
        check_columns(combined, feat_list + [target], sources)
        combined = reorder(combined, ['platform', 'platform_id', 'sample_id', 'datetime', 'year', 'latitude', 'longitude',
                                      'pco2_ocean', 'pco2_atmos', 'delta_pco2'] + prob_cols + class_cols)
        split_dfs = {name: combined[combined['year'].isin(years)].reset_index(drop=True) for name, years in splits}

    # Sea ice clusters from the trainval samples (decides the reg_trainval {ice_tag})
    trainval, _ = rename_classes(split_dfs['trainval'])
    print(f'{sources}: {len(trainval)} trainval samples, {len(prob_cols)} clusters ({pcm_params})')
    print('Features:', feat_list, '| target:', target)
    sea_ice_clusters = find_sea_ice_clusters(trainval, len(prob_cols), cfg.get('ice_threshold', 0.1),
                                             cfg.get('min_ice_profiles', 10))
    ice_tag = 'seaiceclusters' + ''.join(str(k) for k in sea_ice_clusters)

    def csv_output(key, **fmt):
        """Output path; with --from-splits, None (kept) if it already exists, unless --overwrite."""
        if args.from_splits and not run.overwrite and not run.out_dir and os.path.exists(run.path(key, **fmt)):
            print(f'Keeping the existing [files] {key}: {run.path(key, **fmt)}')
            return None
        return run.output(key, **fmt)

    # Check every output before writing any
    if not args.from_splits:
        split_files = {name: run.output(f'P2_{name}') for name, _ in splits}
        app_parquet = run.output('P2_application') if app_file else None
    reg_out = csv_output('reg_trainval', ice_tag=ice_tag)
    test_out = csv_output('reg_test', ice_tag=ice_tag) if 'test' in split_files else None
    app_out = csv_output('reg_application', ice_tag=ice_tag) if app_parquet else None
    info_out = csv_output('reg_info')

    datetag = datetime.now().strftime('%Y%m%d')
    if not args.from_splits:
        common = {'pcm_params': pcm_params, 'pcm_finder': p2_attrs.get('pcm_finder', ''),
                  'class_numbering': p2_attrs.get('class_numbering', ''), **run.attrs(), 'date': datetag}
        for name, years in splits:
            tag = f'{min(years)}-{max(years)}' if len(years) > 1 else str(years[0])
            save_parquet(split_dfs[name], split_files[name], {
                **common,
                'title': f'P2 {name} dataset {tag}: {" and ".join(cfg["train"])} P1 samples with GMM class probabilities; '
                         'samples without a class dropped.',
                'p2_inputs': ', '.join(os.path.basename(f) for f in train_files),
                'gapfill': '; '.join(p.attrs.get('gapfill', '') for p in parts),
                'counts_all_years': {p['platform'].iloc[0] if len(p) else ds: p.attrs.get('counts', {})
                                     for ds, p in zip(cfg['train'], parts)}})
            by_class = split_dfs[name].groupby(['class_assign', 'platform']).size().unstack(fill_value=0)
            print(f'\n{name} {tag}: {len(split_dfs[name])} samples -> {split_files[name]}\n{by_class.to_string()}')

        # ===== 2. Application set, without the float training samples
        if app_file:
            app = pd.read_parquet(app_file)
            check_columns(app, feat_list, os.path.basename(app_file))
            app_attrs, n_app, n_excluded = dict(app.attrs), len(app), 0
            if cfg.get('exclude_training', True):
                train = set() # own profids of float training samples (trainval and test)
                for d in split_dfs.values():
                    if 'profid' in d.columns:
                        train |= set(d.loc[d['platform'].isin(float_platforms), 'profid'].dropna()) - {''}
                drop = app['profid'].isin(train)
                n_excluded = int(drop.sum())
                print(f'\nExcluding {n_excluded} application profiles that are float training samples')
                app = app[~drop].copy()
            app['year'] = app['datetime'].dt.year
            app = reorder(app, ['profid', 'platform', 'platform_id', 'sample_id', 'datetime', 'year', 'latitude',
                                'longitude'] + prob_columns(app) + class_cols)
            first, last = app['year'].min(), app['year'].max()
            yrs = f'{first}-{last}' if last > first else str(first)
            save_parquet(app, app_parquet, {
                **common,
                'title': f'P2 {cfg["application"]} application dataset {yrs}: P1 profiles with GMM class probabilities; '
                         'profiles without a class dropped' + ('; training profiles excluded.' if n_excluded else '.'),
                'p2_input': os.path.basename(app_file),
                'gapfill': app_attrs.get('gapfill', ''),
                'excluded_training_from': ', '.join(os.path.basename(f) for f in split_files.values()),
                'counts': {**app_attrs.get('counts', {}), 'excluded_training': n_excluded}})
            print(f'{len(app)} of {n_app} application profiles -> {app_parquet}')

    # ===== 3-5. Regression csvs, from the saved parquet files
    reg_file = reg_out or run.path('reg_trainval', ice_tag=ice_tag)
    trainval, prob_cols = regression_table(pd.read_parquet(split_files['trainval']), cfg, feat_list, target, 'trainval')
    if reg_out:
        trainval.to_csv(reg_out)
        print(f'\nSaved {len(trainval)} samples to {reg_out}')
    info = {'pcm_params': pcm_params, 'sea_ice_clusters': sea_ice_clusters, 'features': feat_list,
            'drop_features': cfg.get('drop_features', []), 'target': target,
            'trainval_file': os.path.basename(reg_file), 'n_trainval': len(trainval), **run.attrs(), 'date': datetag}
    if 'test' in split_files:
        test, test_probs = regression_table(pd.read_parquet(split_files['test']), cfg, feat_list, target, 'test')
        assert test_probs == prob_cols, 'test and trainval have different clusters'
        if test_out:
            test.to_csv(test_out)
            print(f'\nSaved {len(test)} test samples to {test_out} (run_model_application.py tests the final models on them)')
        info.update({'test_file': os.path.basename(test_out or run.path('reg_test', ice_tag=ice_tag)), 'n_test': len(test)})
    if app_parquet and app_out:
        app = pd.read_parquet(app_parquet)
        app, app_probs = rename_classes(app)
        assert app_probs == prob_cols, 'application and trainval have different clusters'
        app['platform_indexer'] = 'wmoid'
        n = len(app)
        cols = ['profid'] + with_columns([c for c in keep_cols if c in app.columns],
                                         [c for c in feat_list + [target] if c in app.columns]) + prob_cols
        app = app.dropna(subset=feat_list)[cols].set_index('profid')
        app.to_csv(app_out)
        print(f'Saved {len(app)} application profiles ({n - len(app)} dropped for missing features) to {app_out}')
        info.update({'application_file': os.path.basename(app_out), 'n_application': len(app)})
    if info_out:
        with open(info_out, 'w') as f:
            json.dump(info, f, indent=2)
        print(f'Sea ice clusters {sea_ice_clusters} and features saved to {info_out} (read by model-optimization --config)')
    run.finish(sea_ice_clusters=sea_ice_clusters)


if __name__ == '__main__':
    main()
