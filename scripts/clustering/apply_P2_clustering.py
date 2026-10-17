"""Create the P2 training/validation and test datasets: classes for SOCCOM and SOCAT P1 samples.

Replaces the class assignment and gap filling of 2.2_PCM_classify_P1processed_bgcArgo_socat.ipynb,
using the PCM_finder from fit_argo_clustering.py and the shared steps in p2_classes.py:
    SOCCOM 5 m averages (platform 'bgcArgo'): class probabilities of the matched profid from
        P1. Samples without a matched or classified profile are filled from the same float's
        classified profiles (any in PCM_finder, not only SOCCOM samples) just before and just
        after the sample, within --float-days (12) days: if both exist they must have the same
        class; the probabilities are copied from the one nearest in time.
    SOCAT (platform 'socatv2025'): class probabilities of the sample's nearest_profid from P1
        (the nearest clusterable profile within 7 days). Samples whose nearest profile has no
        class are filled the same way from the same cruise's classified samples within
        --ship-days (1.5) days.
    Samples still without a class are dropped (counts printed and saved in the attributes).
    Nothing else is dropped (missing targets or features are left for regression input prep).
Adds platform, platform_id (wmoid or cruiseid), sample_id (prof_datetag or sample_datetag),
and the class columns from p2_classes.py. All P1 columns of both platforms are kept (empty
where a column does not apply). Writes parquet files (one row per sample; run parameters,
inputs and counts in df.attrs after pd.read_parquet) to the PCM_finder's folder:
    P2_combined_trainval_socat_soccom_2014-2023_acc<datetag>.parquet
    P2_combined_test_socat_soccom_2024_acc<datetag>.parquet

Usage:
    python apply_P2_clustering.py --pcm-finder <P2-clustered/pc8_gmm6_dbar500/PCM_finder_..._acc<date>.parquet>
"""
import argparse
import os
import sys
from datetime import datetime

import pandas as pd
import xarray as xr

import p2_classes

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import importer # noqa: E402

splits = [('trainval', list(range(2014, 2024)), '2014-2023'), ('test', [2024], '2024')]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pcm-finder', required=True, help='PCM_finder (parquet) from fit_argo_clustering.py')
    parser.add_argument('--bgc-p1', default=importer.path_to('P1', 'bgc'), help='P1 SOCCOM netcdf')
    parser.add_argument('--socat-p1', default=importer.path_to('P1', 'socat'), help='P1 SOCAT netcdf')
    parser.add_argument('--float-days', type=float, default=12, help='Gap-fill window for SOCCOM samples (days)')
    parser.add_argument('--ship-days', type=float, default=1.5, help='Gap-fill window for SOCAT samples (days)')
    parser.add_argument('--outpath', default=None, help='Output folder (default: the PCM_finder folder)')
    args = parser.parse_args()

    finder, prob_cols, meta = p2_classes.read_finder(args.pcm_finder)
    counts = {}

    # ===== SOCCOM 5 m averages
    bgc = xr.open_dataset(args.bgc_p1).to_dataframe().reset_index(drop=True)
    bgc['datetime'] = pd.to_datetime(bgc['datetime'])
    bgc['platform'], bgc['platform_id'], bgc['sample_id'] = 'bgcArgo', bgc['wmoid'].astype(str), bgc['prof_datetag']
    bgc, (n_matched, n_filled) = p2_classes.classify_float_samples(bgc, 'profid', finder, prob_cols, args.float_days)
    n_bgc = len(bgc)
    bgc = p2_classes.drop_unclassified(bgc, 'SOCCOM', n_matched, n_filled,
                                       f'from the same float within {args.float_days:g} days')
    counts['bgcArgo'] = {'samples': n_bgc, 'matched': n_matched, 'gapfilled': n_filled, 'dropped': n_bgc - len(bgc)}

    # ===== SOCAT
    ship = xr.open_dataset(args.socat_p1).to_dataframe().reset_index(drop=True)
    ship['datetime'] = pd.to_datetime(ship['datetime'])
    ship['platform'], ship['platform_id'], ship['sample_id'] = 'socatv2025', ship['cruiseid'], ship['sample_datetag']
    ship = p2_classes.assign_from_profid(ship, 'nearest_profid', finder, prob_cols)
    n_matched = int(ship['class_assign'].notna().sum())
    ship_cands = ship[ship['class_assign'].notna()].set_index('sample_id', drop=False)
    ship, n_filled = p2_classes.fill_from_neighbours(ship, 'platform_id', ship_cands, 'platform_id',
                                                     args.ship_days, prob_cols)
    n_ship = len(ship)
    ship = p2_classes.drop_unclassified(ship, 'SOCAT', n_matched, n_filled,
                                        f'from the same cruise within {args.ship_days:g} days')
    counts['socatv2025'] = {'samples': n_ship, 'matched': n_matched, 'gapfilled': n_filled, 'dropped': n_ship - len(ship)}

    # ===== Combine, split, save
    lead = ['platform', 'platform_id', 'sample_id', 'datetime', 'year', 'latitude', 'longitude',
            'pco2_ocean', 'pco2_atmos', 'delta_pco2'] + prob_cols + ['class_assign', 'class_profid',
            'class_source', 'class_dt_days']
    combined = pd.concat([bgc, ship], ignore_index=True)
    combined['year'] = combined['datetime'].dt.year
    combined = combined[lead + [c for c in combined.columns if c not in lead]]

    outpath = args.outpath or os.path.dirname(os.path.abspath(args.pcm_finder))
    os.makedirs(outpath, exist_ok=True)
    datetag = datetime.now().strftime('%Y%m%d')
    for split, years, tag in splits:
        part = combined[combined['year'].isin(years)].reset_index(drop=True)
        savefile = os.path.join(outpath, f'P2_combined_{split}_socat_soccom_{tag}_acc{datetag}.parquet')
        p2_classes.save_parquet(part, savefile, {
            'title': f'P2 {split} dataset {tag}: SOCCOM 5 m averages (bgcArgo) and SOCAT (socatv2025) '
                     'P1 samples with GMM class probabilities; samples without a class dropped.',
            'pcm_params': meta.get('pcm_params', ''),
            'pcm_finder': os.path.basename(args.pcm_finder),
            'class_numbering': meta.get('class_numbering', ''),
            'gapfill': f'nearest classified profile/sample of the same float (<{args.float_days:g} d) or '
                       f'cruise (<{args.ship_days:g} d), before/after must agree',
            'p1_inputs': f'{os.path.basename(args.bgc_p1)}, {os.path.basename(args.socat_p1)}',
            'counts_all_years': counts,
            'script': os.path.basename(__file__), 'date': datetag})
        by_class = part.groupby(['class_assign', 'platform']).size().unstack(fill_value=0)
        print(f'\n{split} {tag}: {len(part)} samples -> {savefile}\n{by_class.to_string()}')


if __name__ == '__main__':
    main()
