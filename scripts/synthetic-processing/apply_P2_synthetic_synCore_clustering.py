"""Create the synthetic core Argo application dataset: GMM classes for the synthetic core P1 profiles.

Synthetic counterpart of clustering/apply_P2_core_clustering.py: the file the trained models are
applied to in the synthetic experiment. Uses a PCM_finder fitted on the synthetic L3 profiles with
clustering/fit_argo_clustering.py and the shared steps in clustering/p2_classes.py:
    Each synthetic core profile (platform 'synCoreArgo') takes the class probabilities of its own
    profid. Profiles that were not fit (missing CT or SA at a level above the fit's dbar_limit)
    are filled from the same float's classified profiles just before and just after, within
    --float-days (12) days (both must agree; nearest in time). Profiles still without a class
    are dropped (counts printed and stored in df.attrs).
Adds platform_id (wmoid), sample_id (prof_datetag) and the class columns from p2_classes.py;
all P1 columns (model features, pco2_ocean as the synthetic truth) are kept. Writes one parquet
file (run parameters, inputs and counts in df.attrs) to the PCM_finder's folder:
    P2_synArgo_core_application_<years>_acc<datetag>.parquet

Usage:
    python apply_P2_synthetic_synCore_clustering.py --pcm-finder <synthetic PCM_finder parquet> \\
        --core-p1 <P1_synArgo_core_application...nc>
"""
import argparse
import os
import sys
from datetime import datetime

import pandas as pd
import xarray as xr

scripts_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(os.path.join(scripts_folder, 'clustering'))
import p2_classes # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pcm-finder', required=True, help='Synthetic PCM_finder parquet')
    parser.add_argument('--core-p1', required=True, help='Synthetic P1 core application netcdf')
    parser.add_argument('--float-days', type=float, default=12, help='Gap-fill window (days)')
    parser.add_argument('--outpath', default=None, help='Output folder (default: the PCM_finder folder)')
    args = parser.parse_args()

    finder, prob_cols, meta = p2_classes.read_finder(args.pcm_finder)

    core = xr.open_dataset(args.core_p1).to_dataframe().reset_index()
    core['datetime'] = pd.to_datetime(core['datetime'])
    core['platform_id'], core['sample_id'] = core['wmoid'].astype(int).astype(str), core['prof_datetag']
    core, (n_matched, n_filled) = p2_classes.classify_float_samples(core, 'profid', finder, prob_cols, args.float_days)
    n_core = len(core)
    core = p2_classes.drop_unclassified(core, 'synthetic core', n_matched, n_filled,
                                        f'from the same float within {args.float_days:g} days')

    core['year'] = core['datetime'].dt.year
    lead = ['profid', 'platform', 'platform_id', 'sample_id', 'datetime', 'year', 'latitude', 'longitude',
            'pco2_ocean', 'pco2_atmos', 'delta_pco2'] + prob_cols + ['class_assign', 'class_profid',
            'class_source', 'class_dt_days']
    core = core[lead + [c for c in core.columns if c not in lead]]

    first, last = core['year'].min(), core['year'].max()
    yrs = f'{first}-{last}' if last > first else str(first)
    outpath = args.outpath or os.path.dirname(os.path.abspath(args.pcm_finder))
    os.makedirs(outpath, exist_ok=True)
    datetag = datetime.now().strftime('%Y%m%d')
    savefile = os.path.join(outpath, f'P2_synArgo_core_application_{yrs}_acc{datetag}.parquet')
    p2_classes.save_parquet(core, savefile, {
        'title': f'P2 synthetic core Argo application dataset {yrs}: synthetic core profiles with model '
                 'features and GMM class probabilities (GMM fit on synthetic profiles); unclassified dropped.',
        'pcm_params': meta.get('pcm_params', ''),
        'pcm_finder': os.path.basename(args.pcm_finder),
        'class_numbering': meta.get('class_numbering', ''),
        'gapfill': f'nearest classified profile of the same float (<{args.float_days:g} d), before/after must agree',
        'p1_input': os.path.basename(args.core_p1),
        'counts': {'profiles': n_core, 'matched': n_matched, 'gapfilled': n_filled, 'dropped': n_core - len(core)},
        'script': os.path.basename(__file__), 'date': datetag})
    print(f'\n{len(core)} profiles -> {savefile}\n{core["class_assign"].value_counts().sort_index().to_string()}')


if __name__ == '__main__':
    main()
