"""Create a P2 dataset: the GMM classes of each sample in a P1 dataset (core, bgc, soccom, socat, ...).

Replaces the class assignment and gap filling of 2.2_PCM_classify_P1processed_bgcArgo_socat.ipynb,
using the [files] P2_finder from fit_argo_clustering.py and the shared steps in p2_classes.py.
How a sample gets its class depends on the P1 file:
    floats (P1 has profid: core, bgc, soccom): the class probabilities of the sample's own
        profid. Samples without a classified profile (e.g. missing CT or SA at a level above
        the fit's dbar_limit, or SOCCOM samples with no profile within 1 h) are filled from
        the same float's classified profiles in PCM_finder just before and just after the
        sample, within [P2] float_gapfill_days (12): if both exist they must have the same
        class; the probabilities are copied from the one nearest in time.
    ships (P1 has nearest_profid: socat): the class probabilities of the sample's nearest
        clusterable profile (create_P1_socat_features.py). Samples whose nearest profile has no
        class are filled the same way from the same cruise's classified samples within
        [P2] ship_gapfill_days (1.5).
Samples still without a class are dropped (counts printed and saved in the attributes).
Nothing else is dropped: the training/test split, the exclusion of training profiles from the
application set, and missing targets or features are left to regression/regression-input-prep.py.
Adds platform ([P2.platform] <dataset>, else the P1 platform column), platform_id (wmoid or
cruiseid), sample_id (prof_datetag or sample_datetag), and the class columns from p2_classes.py,
after all P1 columns (in the P1 order). Writes [files] P2_<dataset>, one parquet file with one
row per sample (run parameters, inputs and counts in df.attrs after pd.read_parquet).

Usage:
    python apply_P2_classes.py --config ../../configs/southernOcean_CRUSOE.toml --dataset soccom
    python apply_P2_classes.py --config ../../configs/southernOcean_CRUSOE.toml --dataset socat
    python apply_P2_classes.py --config ../../configs/southernOcean_CRUSOE.toml --dataset core
    python apply_P2_classes.py --config ../../configs/southernOcean_CRUSOE.toml --dataset core --n-gmm 7   # another fit
"""
import os
from datetime import datetime

import pandas as pd
import xarray as xr

from argoclare import pipeline
import p2_classes


def read_p1(path):
    df = xr.open_dataset(path).to_dataframe()
    # one row per profid (floats) or per sample; keep profid as a column, drop a plain sample counter
    df = df.reset_index() if 'profid' in df.index.names else df.reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def main():
    parser = pipeline.add_pcm_args(pipeline.parser(__doc__))
    parser.add_argument('--dataset', required=True, help='P1 dataset to classify (core, bgc, soccom, socat, ...)')
    parser.add_argument('--float-days', type=float, default=None, help='Float gap-fill window (default: [P2] float_gapfill_days)')
    parser.add_argument('--ship-days', type=float, default=None, help='Ship gap-fill window (default: [P2] ship_gapfill_days)')
    args = parser.parse_args()
    run = pipeline.start(args, __file__)
    p2 = run.p2
    finder_file, p1_file = run.input('P2_finder'), run.input('P1', args.dataset)
    savefile = run.output('P2', args.dataset)

    finder, prob_cols, meta = p2_classes.read_finder(finder_file)
    df = read_p1(p1_file)
    ship = 'nearest_profid' in df.columns
    if args.dataset in run.section('P2.platform'):
        df['platform'] = run.section('P2.platform')[args.dataset]
    label = f'{args.dataset} ({df["platform"].iloc[0] if len(df) else ""})'

    if ship:
        days = args.ship_days or p2.get('ship_gapfill_days', 1.5)
        df['platform_id'], df['sample_id'] = df['cruiseid'], df['sample_datetag']
        df = p2_classes.assign_from_profid(df, 'nearest_profid', finder, prob_cols)
        n_matched = int(df['class_assign'].notna().sum())
        cands = df[df['class_assign'].notna()].set_index('sample_id', drop=False)
        df, n_filled = p2_classes.fill_from_neighbours(df, 'platform_id', cands, 'platform_id', days, prob_cols)
        how = f'from the same cruise within {days:g} days'
    else:
        days = args.float_days or p2.get('float_gapfill_days', 12)
        df['platform_id'], df['sample_id'] = df['wmoid'].astype(int).astype(str), df['prof_datetag']
        df, (n_matched, n_filled) = p2_classes.classify_float_samples(df, 'profid', finder, prob_cols, days)
        how = f'from the same float within {days:g} days'
    n = len(df)
    df = p2_classes.drop_unclassified(df, label, n_matched, n_filled, how)

    datetag = datetime.now().strftime('%Y%m%d')
    years = df['datetime'].dt.year
    yrs = f'{years.min()}-{years.max()}' if years.max() > years.min() else str(years.min())
    p2_classes.save_parquet(df, savefile, {
        'title': f'P2 {args.dataset} dataset {yrs}: P1 samples with GMM class probabilities; '
                 'samples without a class dropped.',
        'pcm_params': meta.get('pcm_params', run.pcm),
        'pcm_finder': os.path.basename(finder_file),
        'class_numbering': meta.get('class_numbering', ''),
        'gapfill': f'nearest classified {"sample of the same cruise" if ship else "profile of the same float"} '
                   f'(<{days:g} d), before/after must agree',
        'p1_input': os.path.basename(p1_file),
        'counts': {'samples': n, 'matched': n_matched, 'gapfilled': n_filled, 'dropped': n - len(df)},
        **run.attrs(), 'date': datetag})
    print(f'\n{len(df)} samples -> {savefile}\n{df["class_assign"].value_counts().sort_index().to_string()}')
    run.finish(counts={'samples': n, 'matched': n_matched, 'gapfilled': n_filled, 'dropped': n - len(df)})


if __name__ == '__main__':
    main()
