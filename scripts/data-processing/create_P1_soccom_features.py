"""Create the P1 SOCCOM dataset: regression features for each SOCCOM/GO-BGC surface average sample.

Replaces the feature steps of 1.2_P1_bgcArgo_feature_prep.ipynb, one row per SOCCOM sample:
    1. Read the SOCCOM/GO-BGC surface averages in [files] soccom_dir (var_pK1_<depth>_<wmo>.mat,
       [P1.soccom] depth = 5 m) as in data_loader.import_soccom_20m_averages(), and keep samples
       in the [region] box during the run years ([P1.soccom] years, else [run] years). Samples
       without a time or position are dropped (printed).
    2. Rename with argoclare's process_soccom_20m_averages() and set
       pco2_ocean = [P1.soccom] pco2_var (CRUSOE: pCO2_pHbias5_pK01)
    3. Add the same features as the core P1 dataset with p1_features.add_all_features(), from
       the sample's own time, position, CT, SA and mld; then delta_pco2 = pco2_ocean - pco2_atmos
       (so ERA5 sea level pressure and MBL CO2 are always needed)
    4. Match each sample to a float profid: the nearest profile of the same WMO within
       [P1.soccom] match_hours (1 h) in the [files] L3_index core + bgc (core first, as in
       clustering). Unmatched samples are kept with profid '' and match_dt_minutes NaN;
       samples are dropped in P2, when every training sample must have a class.
prof_datetag (wmoid_YYYYMMDDHH) is kept from the SOCCOM sample time.
Writes [files] P1_soccom, one netcdf along 'sample' (longitude -180 to 180), with platform
'bgcArgo'. This is the CRUSOE training data; create_P1_float_features.py --dataset bgc makes
the general P1 for BGC float profiles.

Usage:
    python create_P1_soccom_features.py --config ../../configs/southernOcean_CRUSOE.toml
    python create_P1_soccom_features.py --config ../../configs/southernOcean_CRUSOE.toml --years 2020 --max-samples 300 --out-dir /tmp/test/
"""
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import scipy.io as sio
import xarray as xr
from joblib import Parallel, delayed

import argoclare.oceantools as mod_ocean
import argoclare.preprocessing as mod_prep
from argoclare import pipeline
import p1_features


def datenum_to_datetime(matlab_datenum):
    days = matlab_datenum % 1
    return datetime.fromordinal(int(matlab_datenum)) + timedelta(days=days) - timedelta(days=366)


def read_soccom_averages(path, depth, start, end, in_box):
    """SOCCOM surface averages for every float, as in data_loader.import_soccom_20m_averages()."""
    parts = []
    for f in sorted(os.listdir(path)):
        if f.startswith(f'var_pK1_{depth}_') and f.endswith('.mat'):
            mat = sio.loadmat(os.path.join(path, f))
            wmo_df = pd.DataFrame(columns=[x[0][0] for x in mat['var_names']], data=mat['var'])
            wmo_df['wmoid'] = f[-11:-4]
            wmo_df['id'] = wmo_df.index.values # row within the float's file
            parts.append(wmo_df)
    soccom_df = pd.concat(parts, ignore_index=True)

    missing = soccom_df[['datenum', 'Lat', 'Lon']].isna().any(axis=1)
    if missing.any():
        print(f'Dropping {int(missing.sum())} samples without datenum, Lat or Lon')
    soccom_df = soccom_df[~missing]
    soccom_df['datetime'] = soccom_df['datenum'].apply(datenum_to_datetime)
    soccom_df = soccom_df[in_box(soccom_df.Lon, soccom_df.Lat)]
    soccom_df = soccom_df[(soccom_df.datetime >= pd.Timestamp(start)) & (soccom_df.datetime < pd.Timestamp(end))]

    soccom_df['prof_datetag'] = soccom_df['wmoid'] + '_' + soccom_df['datetime'].dt.strftime('%Y%m%d%H')
    return soccom_df.reset_index(drop=True)


def match_profids(df, index_files, tolerance):
    """Nearest float profile of the same WMO within `tolerance`; adds profid, match_dt_minutes."""
    floats = []
    for f in index_files:
        idx = xr.open_dataset(f)
        floats.append(pd.DataFrame({'profid': idx.profid.values.astype(str),
                                    'wmoid': idx.wmoid.values.astype(int),
                                    'profid_datetime': pd.to_datetime(idx.datetime.values).astype('datetime64[ns]')}))
    floats = pd.concat(floats).drop_duplicates('profid', keep='first').sort_values('profid_datetime')

    samples = df[['wmoid', 'datetime']].assign(wmoid=df['wmoid'].astype(int), row=np.arange(len(df)),
                                               datetime=df['datetime'].astype('datetime64[ns]'))
    matched = pd.merge_asof(samples.sort_values('datetime'), floats,
                            left_on='datetime', right_on='profid_datetime', by='wmoid',
                            direction='nearest', tolerance=tolerance).set_index('row').sort_index()
    df['profid'] = matched['profid'].values
    df['match_dt_minutes'] = ((matched['datetime'] - matched['profid_datetime']).abs()
                              / pd.Timedelta(minutes=1)).values
    return df


def process_year(yearDF, yr, files, ref_time, seaice_days_before=None):
    t0 = datetime.now()
    yearDF = p1_features.add_all_features(yearDF, yr, files, ref_time, seaice_days_before=seaice_days_before)
    print(f'  {yr}: {len(yearDF)} samples in {(datetime.now() - t0).total_seconds() / 60:.1f} min')
    return yearDF


def main():
    parser = pipeline.parser(__doc__)
    parser.add_argument('--years', type=int, nargs='+', default=None, help='Default: [P1.soccom] or [run] years')
    parser.add_argument('--n-jobs', type=int, default=None, help='Years to process in parallel (default: [P1] n_jobs)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Only use N random samples from each year (for testing)')
    args = parser.parse_args()
    run = pipeline.start(args, __file__)
    cfg, bgc_cfg = run.section('P1'), run.section('P1.soccom')
    depth, pco2_var, match_hours = bgc_cfg['depth'], bgc_cfg['pco2_var'], bgc_cfg.get('match_hours', 1)
    years = run.years('P1.soccom')
    soccom_dir = run.input('soccom_dir')
    indexes = [run.input('L3_index', 'core'), run.input('L3_index', 'bgc')]
    files = p1_features.run_feature_files(run, years, always={'slp', 'mbl'})
    savefile = run.output('P1', 'soccom')

    soccom_df = read_soccom_averages(soccom_dir, depth, f'{min(years)}-01-01', f'{max(years) + 1}-01-01', run.in_box)
    soccom_df = soccom_df[soccom_df.datetime.dt.year.isin(years)]
    soccom_df = mod_prep.process_soccom_20m_averages(soccom_df, use_var=pco2_var)
    soccom_df = mod_ocean.set_longitude_range(soccom_df, end_type='180')
    print(f'{len(soccom_df)} SOCCOM {depth} m samples from {soccom_df.wmoid.nunique()} floats, '
          f'{soccom_df.datetime.min()} to {soccom_df.datetime.max()}')

    yearly = [(yr, df) for yr, df in soccom_df.groupby(soccom_df.datetime.dt.year)]
    if args.max_samples:
        yearly = [(yr, df.sample(min(len(df), args.max_samples), random_state=0).sort_index())
                  for yr, df in yearly]
    print(f'Adding features to {sum(len(df) for _, df in yearly)} samples: ',
          datetime.now().strftime('%Y%m%d %H:%M:%S'))
    parts = Parallel(n_jobs=args.n_jobs or cfg.get('n_jobs', 1))(
        delayed(process_year)(df, yr, files, run.ref_time, cfg.get('seaice_days_before'))
        for yr, df in yearly)
    bgcP1 = pd.concat(parts).sort_values(['wmoid', 'datetime']).reset_index(drop=True)
    bgcP1['delta_pco2'] = bgcP1['pco2_ocean'] - bgcP1['pco2_atmos']
    bgcP1['platform'] = 'bgcArgo'

    tolerance = pd.Timedelta(hours=match_hours)
    bgcP1 = match_profids(bgcP1, indexes, tolerance)
    n_matched = int(bgcP1.profid.notna().sum())
    n_shared = int(bgcP1.profid.dropna().duplicated(keep=False).sum())
    print(f'Matched {n_matched} of {len(bgcP1)} samples to a profid within {match_hours:g} h '
          f'({n_shared} samples share a profid with another sample)')

    datetag = datetime.now().strftime('%Y%m%d')
    first, last = bgcP1.datetime.dt.year.min(), bgcP1.datetime.dt.year.max()
    yrs = f'{first}-{last}' if last > first else str(first)
    p1_features.save_netcdf(bgcP1, 'sample', savefile, {
        'title': f'P1 BGC Argo dataset {yrs}: SOCCOM/GO-BGC {depth} m surface averages with '
                 f'regression features; pco2_ocean = {pco2_var}. One row per sample, including '
                 'samples without a matching float profile (profid = "").',
        'pco2_ocean_variable': pco2_var,
        'profid_match': f'nearest profile of the same WMO within {match_hours:g} h in: '
                        + ', '.join(os.path.basename(f) for f in indexes),
        'samples_matched': n_matched,
        'source': soccom_dir,
        **run.attrs(),
        'date': datetag})
    print(f'Saved {len(bgcP1)} samples to {savefile}: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))
    run.finish(samples_matched=n_matched)


if __name__ == '__main__':
    main()
