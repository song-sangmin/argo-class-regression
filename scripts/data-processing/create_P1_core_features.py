"""Create the P1 core Argo application dataset: regression features for each L3 profile.

Same features as 1.1_P1_coreArgo_feature_prep.ipynb (section I), computed from the L3 index
(one row per profile: surfacemost CT, SA, and mld/mlp) instead of every pressure level:
    time:    year, month, day, linear_time (days since [run] ref_time), ydcos, ydsin, decimalyr
    atmos:   ERA5 sea level pressure (atmos_pres_Pa/_atm), MBL xCO2 (atmos_co2_ppm),
             vapor pressure from surface T/S, and pco2_atmos (uatm)
    ocean:   CMEMS ADT (adt), ERA5 10 m wind speed (wind_speed), NSIDC sea ice (sea_ice;
             optionally sea_ice_avg_<N>d with [P1] seaice_days_before = N), clear-sky
             max_solar_rad, log_mld
Features are looked up with the vectorized functions in p1_features.py (see there for the
differences from argoclare.preprocessing, including the sea level pressure longitude fix).
Years ([P1.core] years, else [run] years) are processed in parallel. Reads [files] L3_index
for core and writes [files] P1_core, one netcdf along profid (longitude -180 to 180).

Usage:
    python create_P1_core_features.py --config ../../configs/example.toml
    python create_P1_core_features.py --config ../../configs/example.toml --max-profiles 500 --out-dir /tmp/test/
"""
import os
from datetime import datetime

import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

import argoclare.oceantools as mod_ocean

import p1_features
import pipeline


def process_year(yearDF, yr, files, ref_time, seaice_days_before=None):
    t0 = datetime.now()
    yearDF = p1_features.add_all_features(yearDF, yr, files, ref_time, seaice_days_before=seaice_days_before)
    print(f'  {yr}: {len(yearDF)} profiles in {(datetime.now() - t0).total_seconds() / 60:.1f} min')
    return yearDF


def main():
    parser = pipeline.parser(__doc__)
    parser.add_argument('--years', type=int, nargs='+', default=None, help='Default: [P1.core] or [run] years')
    parser.add_argument('--n-jobs', type=int, default=None, help='Years to process in parallel (default: [P1] n_jobs)')
    parser.add_argument('--max-profiles', type=int, default=None,
                        help='Only use N random profiles from each year (for testing)')
    args = parser.parse_args()
    run = pipeline.start(args, __file__)
    cfg = run.section('P1')
    years = run.years('P1.core')
    core_index = run.input('L3_index', 'core')
    files = run.feature_files(years)
    savefile = run.output('P1_core')

    print('Reading ' + core_index)
    coreDF = xr.open_dataset(core_index).to_dataframe().reset_index()
    coreDF['datetime'] = pd.to_datetime(coreDF['datetime'])
    coreDF = mod_ocean.set_longitude_range(coreDF, end_type='180')
    coreDF = coreDF[coreDF.datetime.dt.year.isin(years)]

    yearly = [(yr, df) for yr, df in coreDF.groupby(coreDF.datetime.dt.year)]
    if args.max_profiles:
        yearly = [(yr, df.sample(min(len(df), args.max_profiles), random_state=0).sort_index())
                  for yr, df in yearly]
    print(f'Adding features to {sum(len(df) for _, df in yearly)} profiles: ',
          datetime.now().strftime('%Y%m%d %H:%M:%S'))

    parts = Parallel(n_jobs=args.n_jobs or cfg.get('n_jobs', 1))(
        delayed(process_year)(df, yr, files, run.ref_time, cfg.get('seaice_days_before'))
        for yr, df in yearly)
    coreP1 = pd.concat(parts).sort_values('datetime').reset_index(drop=True)
    coreP1['platform'] = 'coreArgo'

    datetag = datetime.now().strftime('%Y%m%d')
    yrs = f'{min(years)}-{max(years)}' if len(years) > 1 else str(years[0])
    p1_features.save_netcdf(coreP1, 'profid', savefile, {
        'title': f'P1 core Argo application dataset {yrs}: regression features at the surface '
                 'location and time of each L3 profile (one row per profid).',
        'source': os.path.basename(core_index),
        **run.attrs(),
        'date': datetag})
    print(f'Saved {len(coreP1)} profiles to {savefile}: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))
    run.finish()


if __name__ == '__main__':
    main()
