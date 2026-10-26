"""Create the synthetic P2 training/validation dataset: GMM classes for synthetic bgc floats and SOCAT.

Synthetic counterpart of clustering/apply_P2_classes.py for soccom + socat (the synthetic core application
dataset is made by apply_P2_synthetic_synCore_clustering.py).
Uses a PCM_finder fitted on the synthetic L3 profiles with clustering/fit_argo_clustering.py
(same parameters as the real data, e.g. pc8_gmm6_dbar500), and the shared steps in
clustering/p2_classes.py:
    synthetic bgc floats (platform 'synBgcArgo'):
        each sample is a profile and takes the class of its own profid; profiles that were not
        fit are filled from the same float's classified profiles just before and just after,
        within [P2] float_gapfill_days (12) days (both must agree; nearest in time)
    synthetic SOCAT ('synSOCAT'): the class of its nearest_profid from P1 (the nearest
        clusterable synthetic profile within 7 days); samples whose nearest profile has no
        class are filled from the same cruise's classified samples within [P2] ship_gapfill_days (1.5)
Samples still without a class are dropped (counts printed and stored in df.attrs).
Subsampling by year (so the synthetic training data is sampled like the real data): for each
year, keep a random ([synthetic] subsample_seed) subset of the classified synthetic bgc profiles with as many samples
as real SOCCOM samples in the real P2 trainval file ([files] real_P2_trainval); years with fewer
synthetic profiles keep them all. Counts by year are printed and stored in df.attrs.
Synthetic SOCAT is already sampled at the real SOCAT samples (build_socatINDEX.py), so it is
not subsampled. --no-subsample keeps every classified synthetic bgc profile.
Adds platform_id (wmoid or cruiseid), sample_id (prof_datetag or sample_datetag) and the class
columns from p2_classes.py; all P1 columns are kept (empty where a column does not apply).
Reads [files] P2_finder, P1_bgc and P1_socat, and writes [files] P2_syn_trainval (parquet; run
parameters, inputs and counts in df.attrs), the training data for regression-input-prep.py.

Usage:
    python apply_P2_synthetic_clustering.py --config ../../configs/synthetic_CRUSOE.toml
"""
import os
import sys
from datetime import datetime

import pandas as pd
import xarray as xr

from argoclare import pipeline

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'clustering'))
import p2_classes # noqa: E402


def read_p1(path):
    df = xr.open_dataset(path).to_dataframe().reset_index()
    df = df.drop(columns=[c for c in ['sample'] if c in df.columns])
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def classify_floats(path, finder, prob_cols, args, label):
    df = read_p1(path)
    df['platform_id'], df['sample_id'] = df['wmoid'].astype(int).astype(str), df['prof_datetag']
    df, (n_matched, n_filled) = p2_classes.classify_float_samples(df, 'profid', finder, prob_cols, args.float_days)
    n = len(df)
    df = p2_classes.drop_unclassified(df, label, n_matched, n_filled,
                                      f'from the same float within {args.float_days:g} days')
    return df, {'samples': n, 'matched': n_matched, 'gapfilled': n_filled, 'dropped': n - len(df)}


def classify_socat(path, finder, prob_cols, args):
    ship = read_p1(path)
    ship['platform_id'], ship['sample_id'] = ship['cruiseid'], ship['sample_datetag']
    ship = p2_classes.assign_from_profid(ship, 'nearest_profid', finder, prob_cols)
    n_matched = int(ship['class_assign'].notna().sum())
    cands = ship[ship['class_assign'].notna()].set_index('sample_id', drop=False)
    ship, n_filled = p2_classes.fill_from_neighbours(ship, 'platform_id', cands, 'platform_id', args.ship_days, prob_cols)
    n = len(ship)
    ship = p2_classes.drop_unclassified(ship, 'synthetic SOCAT', n_matched, n_filled,
                                        f'from the same cruise within {args.ship_days:g} days')
    return ship, {'samples': n, 'matched': n_matched, 'gapfilled': n_filled, 'dropped': n - len(ship)}


def subsample_by_year(syn, real, seed=0):
    """Random subset of synthetic samples with the real number of samples in each year
    (all synthetic samples kept in years where there are fewer than real). Returns the subset
    and a DataFrame of counts by year."""
    real_n = real['datetime'].dt.year.value_counts()
    years = syn['datetime'].dt.year
    parts = []
    for yr, g in syn.groupby(years):
        n = int(real_n.get(yr, 0))
        parts.append(g if len(g) <= n else g.sample(n=n, random_state=seed))
    kept = pd.concat(parts).sort_index()
    counts = pd.DataFrame({'real_SOCCOM': real_n, 'synBgc_classified': years.value_counts(),
                           'synBgc_kept': kept['datetime'].dt.year.value_counts()}).fillna(0).astype(int).sort_index()
    return kept, counts


def finish(df, prob_cols, extra_lead=()):
    df['year'] = df['datetime'].dt.year
    lead = list(extra_lead) + ['platform', 'platform_id', 'sample_id', 'datetime', 'year', 'latitude', 'longitude',
                               'pco2_ocean', 'pco2_atmos', 'delta_pco2'] + prob_cols + \
        ['class_assign', 'class_profid', 'class_source', 'class_dt_days']
    return df[lead + [c for c in df.columns if c not in lead]]


def years_tag(df):
    first, last = df['year'].min(), df['year'].max()
    return f'{first}-{last}' if last > first else str(first)


def main():
    parser = pipeline.add_pcm_args(pipeline.parser(__doc__))
    parser.add_argument('--no-subsample', action='store_true', help='Keep every classified synthetic bgc profile')
    args = parser.parse_args()
    run = pipeline.start(args, __file__)
    args.float_days = run.p2.get('float_gapfill_days', 12)
    args.ship_days = run.p2.get('ship_gapfill_days', 1.5)
    args.seed = run.section('synthetic').get('subsample_seed', 0)
    args.pcm_finder, args.bgc_p1, args.socat_p1 = run.input('P2_finder'), run.input('P1', 'bgc'), run.input('P1', 'socat')
    args.real_trainval = None if args.no_subsample else run.input('real_P2_trainval')
    savefile = run.output('P2', 'syn_trainval')

    finder, prob_cols, meta = p2_classes.read_finder(args.pcm_finder)
    datetag = datetime.now().strftime('%Y%m%d')
    common = {'pcm_params': meta.get('pcm_params', ''), 'pcm_finder': os.path.basename(args.pcm_finder),
              'class_numbering': meta.get('class_numbering', ''), **run.attrs(),
              'date': datetag}

    # ===== Training/validation: synthetic bgc floats + synthetic SOCAT
    bgc, bgc_counts = classify_floats(args.bgc_p1, finder, prob_cols, args, 'synthetic bgc')
    sampling = 'none (all classified synthetic bgc profiles kept)'
    if not args.no_subsample:
        real = pd.read_parquet(args.real_trainval, columns=['platform', 'datetime'])
        real = real[real['platform'] == 'bgcArgo'].copy()
        real['datetime'] = pd.to_datetime(real['datetime'])
        n_classified = len(bgc)
        bgc, by_year = subsample_by_year(bgc, real, args.seed)
        print(f'\nSubsampling by year to the real SOCCOM counts ({os.path.basename(args.real_trainval)}): '
              f'kept {len(bgc)} of {n_classified} synthetic bgc profiles ({len(real)} real samples)')
        print(by_year.to_string())
        bgc_counts.update({'subsampled': len(bgc), 'real_SOCCOM_samples': len(real),
                           'by_year_real_classified_kept': {str(y): [int(v) for v in r] for y, r in by_year.iterrows()}})
        sampling = (f'random subset (seed {args.seed}) of synthetic bgc per year, sized to the real SOCCOM '
                    f'samples per year in {os.path.basename(args.real_trainval)}; all kept where fewer')
    ship, ship_counts = classify_socat(args.socat_p1, finder, prob_cols, args)
    trainval = finish(pd.concat([bgc, ship], ignore_index=True), prob_cols)
    yrs = years_tag(trainval)
    p2_classes.save_parquet(trainval, savefile, {
        **common,
        'title': f'P2 synthetic trainval dataset {yrs}: synthetic bgc floats (synBgcArgo) and synthetic SOCAT '
                 '(synSOCAT) with GMM class probabilities (GMM fit on synthetic profiles); unclassified dropped.',
        'gapfill': f'nearest classified profile/sample of the same float (<{args.float_days:g} d) or '
                   f'cruise (<{args.ship_days:g} d), before/after must agree',
        'p1_inputs': f'{os.path.basename(args.bgc_p1)}, {os.path.basename(args.socat_p1)}',
        'bgc_sampling': sampling,
        'counts': {'synBgcArgo': bgc_counts, 'synSOCAT': ship_counts}})
    by_class = trainval.groupby(['class_assign', 'platform']).size().unstack(fill_value=0)
    print(f'\ntrainval {yrs}: {len(trainval)} samples -> {savefile}\n{by_class.to_string()}')
    run.finish(counts={'synBgcArgo': bgc_counts, 'synSOCAT': ship_counts})


if __name__ == '__main__':
    main()
