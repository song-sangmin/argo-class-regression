"""Create the P2 core Argo application dataset: classes for the P1 core profiles.

Uses the PCM_finder from fit_argo_clustering.py and the shared steps in p2_classes.py:
    Each P1 core profile (platform 'coreArgo') takes the class probabilities of its own profid.
    Most are in PCM_finder, since the GMM is fit on these profiles; profiles that were not fit
    (missing CT or SA at a level above the fit's dbar_limit) are filled from the same float's
    classified profiles just before and just after, within --float-days (12) days: if both
    exist they must have the same class; the probabilities are copied from the one nearest in
    time. Profiles still without a class are dropped (counts printed and saved in the attributes).
    Profiles that are SOCCOM training/validation or test samples (the matched profid of a bgcArgo
    sample in the --exclude-p2 files) are removed, so the application set does not overlap the
    training data. Other profiles of those floats, profiles colocated to SOCAT (nearest_profid)
    and gap-fill class sources are kept.
Adds platform, platform_id (wmoid), sample_id (prof_datetag), and the class columns from
p2_classes.py; all P1 columns are kept. Writes one parquet file (one row per profid; run
parameters, inputs and counts in df.attrs after pd.read_parquet) to the PCM_finder's folder:
    P2_coreArgo_application_<years>_acc<datetag>.parquet

Usage:
    python apply_P2_core_clustering.py --pcm-finder <P2-clustered/pc8_gmm6_dbar500/PCM_finder_..._acc<date>.parquet> \
        --exclude-p2 <P2_combined_trainval_...parquet> <P2_combined_test_...parquet>
"""
import argparse
import os
import sys
from datetime import datetime

import pandas as pd
import xarray as xr

import p2_classes

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argoclare.importer as importer # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pcm-finder', required=True, help='PCM_finder (parquet) from fit_argo_clustering.py')
    parser.add_argument('--core-p1', default=importer.path_to('P1', 'core'), help='P1 core application netcdf')
    parser.add_argument('--exclude-p2', nargs='+', required=True,
                        help='P2 combined trainval/test parquet files; their SOCCOM profiles are excluded')
    parser.add_argument('--float-days', type=float, default=12, help='Gap-fill window (days)')
    parser.add_argument('--outpath', default=None, help='Output folder (default: the PCM_finder folder)')
    args = parser.parse_args()

    finder, prob_cols, meta = p2_classes.read_finder(args.pcm_finder)

    core = xr.open_dataset(args.core_p1).to_dataframe().reset_index()
    core['datetime'] = pd.to_datetime(core['datetime'])
    core['platform'], core['platform_id'], core['sample_id'] = ('coreArgo', core['wmoid'].astype(int).astype(str),
                                                                core['prof_datetag'])
    core, (n_matched, n_filled) = p2_classes.classify_float_samples(core, 'profid', finder, prob_cols, args.float_days)
    n_core = len(core)
    core = p2_classes.drop_unclassified(core, 'core Argo', n_matched, n_filled,
                                        f'from the same float within {args.float_days:g} days')
    # Profiles that are SOCCOM training/validation (or test) samples are not part of the application set
    core, n_excluded = p2_classes.exclude_training_profiles(core, args.exclude_p2)

    lead = ['profid', 'platform', 'platform_id', 'sample_id', 'datetime', 'year', 'latitude', 'longitude'] \
        + prob_cols + ['class_assign', 'class_profid', 'class_source', 'class_dt_days']
    core['year'] = core['datetime'].dt.year
    core = core[lead + [c for c in core.columns if c not in lead]]

    first, last = core['year'].min(), core['year'].max()
    yrs = f'{first}-{last}' if last > first else str(first)
    outpath = args.outpath or os.path.dirname(os.path.abspath(args.pcm_finder))
    os.makedirs(outpath, exist_ok=True)
    datetag = datetime.now().strftime('%Y%m%d')
    savefile = os.path.join(outpath, f'P2_coreArgo_application_{yrs}_acc{datetag}.parquet')
    p2_classes.save_parquet(core, savefile, {
        'title': f'P2 core Argo application dataset {yrs}: P1 core profiles with GMM class probabilities; '
                 'profiles without a class dropped.',
        'pcm_params': meta.get('pcm_params', ''),
        'pcm_finder': os.path.basename(args.pcm_finder),
        'class_numbering': meta.get('class_numbering', ''),
        'gapfill': f'nearest classified profile of the same float (<{args.float_days:g} d), before/after must agree',
        'p1_input': os.path.basename(args.core_p1),
        'excluded_training_from': ', '.join(os.path.basename(f) for f in args.exclude_p2),
        'counts': {'profiles': n_core, 'matched': n_matched, 'gapfilled': n_filled,
                   'dropped_unclassified': n_core - len(core) - n_excluded, 'excluded_training': n_excluded},
        'script': os.path.basename(__file__), 'date': datetag})
    by_class = core['class_assign'].value_counts().sort_index()
    print(f'\n{len(core)} profiles -> {savefile}\n{by_class.to_string()}')


if __name__ == '__main__':
    main()
