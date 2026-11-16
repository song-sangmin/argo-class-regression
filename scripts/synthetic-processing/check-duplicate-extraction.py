"""Remove duplicate synthetic samples from the regression inputs.

The synthetic floats are matched to the nearest model particle at their real deployment
(PKG-synthetic-extract/build_realworldINDEX.py), and nothing stops two floats from taking the same
particle. Floats deployed together then follow the same particle, so they give identical samples
(same position, time, features and target) under different WMOs. This script, run after
regression/regression-input-prep.py, removes them from [files] reg_trainval and reg_application:
    1. trainval: samples at the same position and time as an earlier one (synBgcArgo kept over
       synSOCAT, then the first in file order)
    2. application: synCoreArgo profiles at the same position and time as an earlier one
    3. application: synCoreArgo profiles at the same position and time as a trainval sample
       (synBgcArgo or synSOCAT; copies of the data seen in cross validation)
Position is matched to 1e-6 degrees and time to the second. The duplicates are checked to have the
same features and target (within TOLERANCE, or TS_TOLERANCE for CT and SA) before they are dropped.
Without --overwrite the counts are printed and nothing is written; with it, both csvs are rewritten and
[files] reg_info gets the new n_trainval and n_application, and the counts removed are added to
duplicates_removed (so a rerun on cleaned files keeps the earlier counts).

Usage:
    python check-duplicate-extraction.py --config ../../configs/synthetic_CRUSOE.toml
    python check-duplicate-extraction.py --config ../../configs/synthetic_CRUSOE.toml --overwrite
"""

import json

import numpy as np
import pandas as pd

from argoclare import pipeline

PLATFORM_ORDER = {'synBgcArgo': 0, 'synSOCAT': 1}  # which copy of a trainval duplicate is kept
# largest difference allowed between duplicates: a synBgcArgo and a synSOCAT sample of the same model
# point can differ by ~1e-5 in CT/SA (surface values computed separately)
TOLERANCE = 1e-3
# a synSOCAT sample and a synCoreArgo profile at the same model point differ by up to ~0.02 in CT/SA
# (the float value is from the regridded profile)
TS_TOLERANCE = 0.05


def sample_keys(df):
    """Position (1e-6 deg) and time (s) of each row, as one string."""
    lat = df['latitude'].round(6).map('{:.6f}'.format)
    lon = df['longitude'].round(6).map('{:.6f}'.format)
    time = pd.to_datetime(df['datetime'], format='mixed').dt.strftime('%Y%m%d%H%M%S')
    return lat + '_' + lon + '_' + time


def check_identical(df, keys, cols, label):
    """Largest spread of cols within the rows that share a key; stops if any differ by more than the tolerance."""
    dup = keys.duplicated(keep=False)
    if not dup.any():
        return
    spread = df.loc[dup, cols].groupby(keys[dup]).agg(lambda s: s.max() - s.min()).max()
    tol = pd.Series([TS_TOLERANCE if c in ('CT', 'SA') else TOLERANCE for c in cols], index=cols)
    print(f'  {label}: largest spread within duplicates {spread.round(4).to_dict()}')
    if (spread > tol).any():
        raise SystemExit(f'{label}: duplicates differ in {list(spread[spread > tol].index)}; not removing them')


def add_counts(old, new):
    """Sum two (nested) count dicts."""
    out = dict(old)
    for k, v in new.items():
        out[k] = add_counts(out.get(k, {}), v) if isinstance(v, dict) else out.get(k, 0) + v
    return out


def main():
    args = pipeline.parser(__doc__).parse_args()
    run = pipeline.start(args, __file__)
    info_path = run.input('reg_info')
    with open(info_path) as f:
        info = json.load(f)
    ice_tag = 'seaiceclusters' + ''.join(str(k) for k in info['sea_ice_clusters'])
    trainval_path = run.input('reg_trainval', ice_tag=ice_tag)
    app_path = run.input('reg_application', ice_tag=ice_tag)
    cols = info['features'] + [info['target']]

    trainval = pd.read_csv(trainval_path, index_col=0)
    app = pd.read_csv(app_path, index_col=0)
    print(f'{len(trainval)} trainval samples: {trainval_path}\n{len(app)} application profiles: {app_path}')

    # 1. trainval: keep synBgcArgo over synSOCAT, then file order
    tv_keys = sample_keys(trainval)
    check_identical(trainval, tv_keys, cols, 'trainval')
    order = trainval['platform'].map(PLATFORM_ORDER).fillna(len(PLATFORM_ORDER))
    ranked = tv_keys.to_frame('key').assign(order=order.values, pos=np.arange(len(trainval)))
    drop_tv = ranked.sort_values(['order', 'pos']).duplicated('key')
    drop_tv = drop_tv.reindex(trainval.index)
    counts = {'trainval_duplicates': trainval.loc[drop_tv, 'platform'].value_counts().to_dict()}
    trainval_out = trainval[~drop_tv]

    # 2. application: repeats within synCoreArgo
    app_keys = sample_keys(app)
    check_identical(app, app_keys, cols, 'application')
    drop_app = app_keys.duplicated()
    counts['application_duplicates'] = int(drop_app.sum())

    # 3. application: copies of training samples (synBgcArgo and synSOCAT)
    tv_out_keys = tv_keys[~drop_tv]
    in_tv = app_keys.isin(set(tv_out_keys)) & ~drop_app
    if in_tv.any():
        both = pd.concat([trainval_out[cols].set_index(tv_out_keys), app.loc[in_tv, cols].set_index(app_keys[in_tv])])
        check_identical(both.reset_index(drop=True), pd.Series(both.index), cols, 'application vs trainval')
    platform_of = trainval_out['platform'].set_axis(tv_out_keys)
    counts['application_in_trainval'] = platform_of.loc[app_keys[in_tv]].value_counts().to_dict()
    app_out = app[~(drop_app | in_tv)]

    print(f'\ntrainval: {int(drop_tv.sum())} duplicates {counts["trainval_duplicates"]} -> {len(trainval_out)} samples')
    print(
        f'application: {counts["application_duplicates"]} duplicates, {int(in_tv.sum())} copies of trainval '
        f'samples {counts["application_in_trainval"]} -> {len(app_out)} profiles'
    )
    if 'cluster' in trainval_out:
        print('\ntrainval samples by cluster and platform:')
        print(pd.crosstab(trainval_out['cluster'], trainval_out['platform'], margins=True).to_string())

    if not args.overwrite and not args.out_dir:
        print('\nNothing written; pass --overwrite to rewrite the csvs and reg_info in place.')
        return
    trainval_save = run.output('reg_trainval', ice_tag=ice_tag)
    app_save = run.output('reg_application', ice_tag=ice_tag)
    info_save = run.output('reg_info')
    trainval_out.to_csv(trainval_save)
    app_out.to_csv(app_save)
    info.update(
        {
            'n_trainval': len(trainval_out),
            'n_application': len(app_out),
            'duplicates_removed': add_counts(info.get('duplicates_removed', {}), counts),
            'duplicates_script': run.script,
        }
    )
    with open(info_save, 'w') as f:
        json.dump(info, f, indent=2)
    print(f'\nSaved {trainval_save}\nSaved {app_save}\nSaved {info_save}')
    run.finish(duplicates_removed=counts)


if __name__ == '__main__':
    main()
