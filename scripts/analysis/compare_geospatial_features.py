"""Check whether a single-cluster model with geospatial/time predictors generalizes worse than
the 6-cluster model without them.

Hypothesis: pre-clustering lets the model leave out latitude, longitude and time; a single
model that uses them may score well in-sample but overfit, and do worse on unseen data.

Models (random forests, the same fixed hyperparameters for all; no tuning):
    clustered6       one model per GMM cluster on its assigned rows, [regression] features
                     (sea_ice only in the reg_info sea ice clusters), weighted by cluster{k}_prob
    single           one model, the same features (sea_ice everywhere)
    single_geo       + latitude, sin/cos(longitude), sin/cos(day of year)
    single_geo_year  + decimal year (a trend; trees cannot extrapolate it past 2023)
    clustered6_geo   clustered6 with the single_geo features, to separate the effect of the
                     clustering from that of the coordinates
Each model also gets the pipeline's linear calibration (decile means of out-of-fold predictions,
per assigned cluster for clustered6), fitted only on its own training rows.

Evaluations:
    1. platform CV:  GroupKFold by platform_id (as the nested CV, so in-sample validation)
    2. sector CV:    leave one 45 degree longitude sector out (regions with no training data)
       kmeans CV:    the pipeline's k-means folds (crossvalidation.subset_folds type='kmeans', on
                     scaled latitude and sin/cos longitude), made once on all of trainval so every
                     model holds out the same regions; errors also binned by each held-out sample's
                     distance to the nearest training sample of its fold
       forward CV:   for each of --forward-years, train on the earlier years and score that year,
                     split into platforms (floats, cruises) seen or not seen in training
    3. test:         final models trained on all trainval, scored on [files] model_test (else reg_test) or --test
                     (by platform/year, and 2024 bgcArgo floats seen vs not seen in training)
    4. application:  no truth for coreArgo, so the difference of each single model from
                     clustered6, binned by distance to the nearest training sample
Writes the tables and the test/application predictions to --out-dir.

Usage:
    python compare_geospatial_features.py --config ../../configs/southernOcean_CRUSOE.toml \\
        --test .../combined_test_bgc_socat_glodap_pc8_gmm6_dbar500.csv --out-dir .../geospatial_check
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.neighbors import BallTree
from summarize_cv_errors import error_metrics

from argoclare import crossvalidation, pipeline

GEO = ['latitude', 'lon_sin', 'lon_cos', 'doy_sin', 'doy_cos']
YEAR = ['decimal_year']
EARTH_RADIUS_KM = 6371.0
DIST_BINS = [0, 50, 200, 500, 1000, np.inf]
DIST_LABELS = ['<50', '50-200', '200-500', '500-1000', '>1000']


def add_encodings(df):
    """Cyclic longitude and day of year, and decimal year, from longitude and datetime."""
    t = pd.to_datetime(df['datetime'], format='mixed')
    days_in_year = np.where(t.dt.is_leap_year, 366, 365)
    year_frac = (t.dt.dayofyear - 1 + t.dt.hour / 24) / days_in_year
    lon = np.deg2rad(df['longitude'])
    return df.assign(
        lon_sin=np.sin(lon),
        lon_cos=np.cos(lon),
        doy_sin=np.sin(2 * np.pi * year_frac),
        doy_cos=np.cos(2 * np.pi * year_frac),
        decimal_year=t.dt.year + year_frac,
    )


class Model:
    """A clustered (one forest per cluster, probability-weighted) or single forest."""

    def __init__(self, features, n_clusters, sea_ice_clusters, rf_params):
        self.n_clusters = n_clusters  # 1 for a single model
        self.features = {
            k: [f for f in features if f != 'sea_ice' or n_clusters == 1 or k in sea_ice_clusters]
            for k in range(1, n_clusters + 1)
        }
        self.rf_params = rf_params

    def fit(self, df, target):
        self.models = {}
        for k, feats in self.features.items():
            rows = df if self.n_clusters == 1 else df[df['cluster'] == k]
            self.models[k] = RandomForestRegressor(**self.rf_params).fit(rows[feats], rows[target])
        return self

    def predict_parts(self, df):
        return pd.DataFrame({k: m.predict(df[self.features[k]]) for k, m in self.models.items()}, index=df.index)

    def combine(self, parts, df):
        if self.n_clusters == 1:
            return parts[1]
        return sum(df[f'cluster{k}_prob'] * parts[k] for k in parts.columns)

    def predict(self, df):
        return self.combine(self.predict_parts(df), df)

    def cal_groups(self, df):
        return df['cluster'] if self.n_clusters > 1 else pd.Series(1, index=df.index)


def calibration_coeffs(pred, truth, groups):
    """Slope and intercept per group from decile means of pred vs truth, as
    regression.get_clustered_calibration_coeffs."""
    coeffs = {}
    for k, idx in groups.groupby(groups).groups.items():
        p, y = pred.loc[idx], truth.loc[idx]
        decile = pd.qcut(p, 10, labels=False, duplicates='drop')
        fit = stats.linregress(p.groupby(decile).mean(), y.groupby(decile).mean())
        coeffs[k] = (fit.slope, fit.intercept)
    return coeffs


def fit_calibrated(spec, train, target, n_inner, seed):
    """Final model on all of train, with calibration from n_inner platform-grouped out-of-fold predictions."""
    oof = pd.Series(np.nan, index=train.index)
    inner = GroupKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    for tr, va in inner.split(train, groups=train['platform_id']):
        model = Model(**spec).fit(train.iloc[tr], target)
        oof.iloc[va] = model.predict(train.iloc[va]).values
    model = Model(**spec).fit(train, target)
    model.cal = calibration_coeffs(oof, train[target], model.cal_groups(train))
    return model


def predict_calibrated(model, df):
    """Raw and calibrated predictions (cluster k coefficients applied to cluster k's prediction)."""
    parts = model.predict_parts(df)
    calparts = pd.DataFrame({k: parts[k] * model.cal[k][0] + model.cal[k][1] for k in parts.columns})
    return model.combine(parts, df), model.combine(calparts, df)


def cross_validate(spec, data, target, splits, n_inner, seed):
    """Out-of-fold raw and calibrated predictions over the given (train, test) index splits."""
    raw, cal = pd.Series(np.nan, index=data.index), pd.Series(np.nan, index=data.index)
    for tr, te in splits:
        model = fit_calibrated(spec, data.iloc[tr], target, n_inner, seed)
        raw.iloc[te], cal.iloc[te] = (s.values for s in predict_calibrated(model, data.iloc[te]))
    return raw, cal


def metric_rows(preds, truth, subsets):
    """One row per model and subset, with raw and calibrated metrics."""
    rows = []
    for name, (raw, cal) in preds.items():
        for subset, mask in subsets.items():
            m_raw = error_metrics(raw[mask], truth[mask])
            m_cal = error_metrics(cal[mask], truth[mask])
            rows.append(
                {'model': name, 'subset': subset} | m_raw | {f'{k}_cal': v for k, v in m_cal.items() if k != 'n'}
            )
    return pd.DataFrame(rows)


def kmeans_splits(trainval, n_folds):
    """(train, test) positions of the pipeline's k-means spatial folds."""
    train_inds, val_inds = crossvalidation.subset_folds(
        trainval[['latitude', 'longitude']].copy(), type='kmeans', nfolds=n_folds
    )
    return [(trainval.index.get_indexer(train_inds[f]), trainval.index.get_indexer(val_inds[f])) for f in train_inds]


def nearest_km(train, df):
    """Great-circle distance (km) from each row of df to the nearest training sample."""
    tree = BallTree(np.deg2rad(train[['latitude', 'longitude']].values), metric='haversine')
    dist, _ = tree.query(np.deg2rad(df[['latitude', 'longitude']].values), k=1)
    return pd.Series(dist[:, 0] * EARTH_RADIUS_KM, index=df.index)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--config', required=True, help='Run config with [files] reg_trainval, reg_info, reg_application'
    )
    parser.add_argument('--test', default=None, help='Test csv (default: [files] model_test, else reg_test)')
    parser.add_argument('--out-dir', required=True, help='Folder for the output tables and predictions')
    parser.add_argument('--n-folds', type=int, default=5, help='Platform CV folds (default 5)')
    parser.add_argument('--sector-deg', type=float, default=45, help='Longitude sector width for sector CV')
    parser.add_argument('--n-inner', type=int, default=3, help='Folds for the calibration out-of-fold predictions')
    parser.add_argument('--n-estimators', type=int, default=200)
    parser.add_argument('--min-samples-leaf', type=int, default=7)
    parser.add_argument('--max-features', type=float, default=0.5, help='Fraction of features per split')
    parser.add_argument('--kmeans-folds', type=int, default=10, help='Regions in the k-means CV (default 10)')
    parser.add_argument(
        '--forward-years',
        type=int,
        nargs='+',
        default=[2020, 2021, 2022, 2023],
        help='Years scored in the forward CV, each with a model trained on the earlier years',
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--skip-cv', action='store_true', help='Only the test and application evaluations')
    args = parser.parse_args()

    run = pipeline.load(args.config)
    with open(run.path('reg_info')) as f:
        info = json.load(f)
    base, target, sea_ice_clusters = info['features'], info.get('target', 'delta_pco2'), info['sea_ice_clusters']
    test_file = args.test or run.model_test_path()[0]

    trainval = add_encodings(pd.read_csv(run.path('reg_trainval'), index_col=0))
    test = add_encodings(pd.read_csv(test_file, index_col=0))
    application = add_encodings(pd.read_csv(run.path('reg_application'), index_col=0, low_memory=False))
    application = application.dropna(subset=[f for f in base if f != 'sea_ice'] + GEO + YEAR)
    n_clusters = len([c for c in trainval.columns if c.startswith('cluster') and c.endswith('_prob')])
    print(f'trainval {len(trainval)}, test {len(test)}, application {len(application)}; {n_clusters} clusters')

    rf_params = {
        'n_estimators': args.n_estimators,
        'min_samples_leaf': args.min_samples_leaf,
        'max_features': args.max_features,
        'n_jobs': -1,
        'random_state': args.seed,
    }
    specs = {
        f'clustered{n_clusters}': dict(features=base, n_clusters=n_clusters),
        'single': dict(features=base, n_clusters=1),
        'single_geo': dict(features=base + GEO, n_clusters=1),
        'single_geo_year': dict(features=base + GEO + YEAR, n_clusters=1),
        f'clustered{n_clusters}_geo': dict(features=base + GEO, n_clusters=n_clusters),
    }
    for spec in specs.values():
        spec |= dict(sea_ice_clusters=sea_ice_clusters, rf_params=rf_params)
    for name, spec in specs.items():
        print(f'  {name}: {spec["features"]}')
    os.makedirs(args.out_dir, exist_ok=True)
    tables = {}

    # 1-2. Cross-validation on trainval
    if not args.skip_cv:
        sector = np.floor((trainval['longitude'] + 180) / args.sector_deg).astype(int)
        cv_splits = {
            'platform CV': list(
                GroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed).split(
                    trainval, groups=trainval['platform_id']
                )
            ),
            'sector CV': list(LeaveOneGroupOut().split(trainval, groups=sector)),
            'kmeans CV': kmeans_splits(trainval, args.kmeans_folds),
            'forward CV': [
                (np.flatnonzero(trainval['year'] < yr), np.flatnonzero(trainval['year'] == yr))
                for yr in args.forward_years
            ],
        }
        # Forward CV: is each scored row's platform in the earlier years it was trained on?
        pid = trainval['platform_id'].astype(str)
        seen = pd.Series(False, index=trainval.index)
        for yr in args.forward_years:
            rows = trainval['year'] == yr
            seen[rows] = pid[rows].isin(set(pid[trainval['year'] < yr]))
        cv_rows = []
        for cv_name, splits in cv_splits.items():
            preds = {}
            for name, spec in specs.items():
                t0 = time.time()
                preds[name] = cross_validate(spec, trainval, target, splits, args.n_inner, args.seed)
                print(f'{cv_name} {name}: {time.time() - t0:.0f} s')
            subsets = {'all': trainval.index == trainval.index}
            if cv_name == 'sector CV':
                labels = sector.map(lambda s: f'{-180 + s * args.sector_deg:g}E')
                subsets |= {f'sector {lab}': (labels == lab).values for lab in sorted(labels.unique(), key=float_key)}
            if cv_name == 'kmeans CV':
                held_out_km = pd.Series(np.nan, index=trainval.index)
                for tr, te in splits:
                    held_out_km.iloc[te] = nearest_km(trainval.iloc[tr], trainval.iloc[te]).values
                bins = pd.cut(held_out_km, DIST_BINS, labels=DIST_LABELS)
                subsets |= {f'{b} km from training': (bins == b).values for b in DIST_LABELS}
            if cv_name == 'forward CV':  # rows outside --forward-years have no prediction and are skipped
                scored = trainval['year'].isin(args.forward_years)
                subsets = {
                    f'{min(args.forward_years)}-{max(args.forward_years)}': scored.values,
                    'seen platform': (scored & seen).values,
                    'new platform': (scored & ~seen).values,
                }
                for yr in args.forward_years:
                    for label, flag in [('seen', seen), ('new', ~seen)]:
                        subsets[f'{yr} {label} platform'] = ((trainval['year'] == yr) & flag).values
            cv_rows.append(metric_rows(preds, trainval[target], subsets).assign(evaluation=cv_name))
        tables['cv'] = pd.concat(cv_rows, ignore_index=True)

    # 3-4. Final models: test set and application
    seen_floats = set(trainval['platform_id'].astype(str))
    test_pid = test['platform_id'].astype(str)
    is_2024_bgc = (test['platform'] == 'bgcArgo') & (test['year'] >= 2024)
    subsets = {'all': (test.index == test.index)}
    for plat, years in test.groupby('platform')['year']:
        span = f'{years.min()}' + (f'-{years.max()}' if years.max() != years.min() else '')
        subsets[f'{plat} {span}'] = (test['platform'] == plat).values
    subsets['bgcArgo 2024, float in trainval'] = (is_2024_bgc & test_pid.isin(seen_floats)).values
    subsets['bgcArgo 2024, new float'] = (is_2024_bgc & ~test_pid.isin(seen_floats)).values

    test_preds, app_preds = {}, {}
    for name, spec in specs.items():
        t0 = time.time()
        model = fit_calibrated(spec, trainval, target, args.n_folds, args.seed)
        test_preds[name] = predict_calibrated(model, test)
        app_preds[name] = predict_calibrated(model, application)
        print(f'final {name}: {time.time() - t0:.0f} s')
    tables['test'] = metric_rows(test_preds, test[target], subsets).assign(evaluation='test')

    reference = f'clustered{n_clusters}'
    dist = nearest_km(trainval, application)
    bins = pd.cut(dist, DIST_BINS, labels=DIST_LABELS)
    app_rows = []
    for name, (raw, cal) in app_preds.items():
        for label, idx in [('all', application.index)] + [
            (f'{b} km', bins.index[bins == b]) for b in bins.cat.categories
        ]:
            diff = raw.loc[idx] - app_preds[reference][0].loc[idx]
            app_rows.append(
                {
                    'model': name,
                    'nearest_training_sample': label,
                    'n': len(idx),
                    'mean_pred': raw.loc[idx].mean(),
                    'std_pred': raw.loc[idx].std(),
                    f'rms_diff_vs_{reference}': np.sqrt(np.mean(diff**2)),
                    f'mean_diff_vs_{reference}': diff.mean(),
                }
            )
    tables['application'] = pd.DataFrame(app_rows)

    # Print and save
    cols = ['model', 'subset', 'n', 'rmse', 'mae', 'bias', 'r2', 'rmse_cal', 'bias_cal', 'r2_cal']
    pd.set_option('display.width', 200)
    if 'cv' in tables:
        for cv_name, table in tables['cv'].groupby('evaluation', sort=False):
            print(f'\n{cv_name} (out-of-fold, prediction - truth):')
            print(table[cols].round(3).to_string(index=False))
    print('\nTest set (final models trained on all trainval):')
    print(tables['test'][cols].round(3).to_string(index=False))
    print(f'\ncoreArgo application (no truth): predictions vs {reference}, by distance to nearest training sample:')
    print(tables['application'].round(3).to_string(index=False))

    for key, table in tables.items():
        table.to_csv(os.path.join(args.out_dir, f'geospatial_check_{key}Metrics.csv'), index=False)
    pd.DataFrame(
        {f'{n}_{t}': p for n, (raw, cal) in test_preds.items() for t, p in [('pred', raw), ('calpred', cal)]}
    ).join(test[['platform', 'platform_id', 'year', 'latitude', 'longitude', target]]).to_csv(
        os.path.join(args.out_dir, 'geospatial_check_testPredictions.csv')
    )
    pd.DataFrame({f'{n}_pred': raw for n, (raw, _) in app_preds.items()}).assign(
        nearest_training_km=dist, latitude=application['latitude'], longitude=application['longitude']
    ).to_parquet(os.path.join(args.out_dir, 'geospatial_check_applicationPredictions.parquet'))
    with open(os.path.join(args.out_dir, 'geospatial_check_settings.json'), 'w') as f:
        json.dump(
            vars(args) | {'features': {n: s['features'] for n, s in specs.items()}, 'rf_params': rf_params},
            f,
            indent=2,
        )
    print(f'\nSaved tables and predictions to {args.out_dir}')


def float_key(label):
    return float(label.rstrip('E'))


if __name__ == '__main__':
    main()
