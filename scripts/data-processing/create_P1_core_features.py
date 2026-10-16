"""Create the P1 core Argo application dataset: regression features for each L3 profile.

Same features as 1.1_P1_coreArgo_feature_prep.ipynb (section I), computed from the L3 index
(one row per profile: surfacemost CT, SA, and mld/mlp) instead of every pressure level:
    time:    year, month, day, linear_time (days since 2014-01-01), ydcos, ydsin, decimalyr
    atmos:   ERA5 sea level pressure (atmos_pres_Pa/_atm), MBL xCO2 (atmos_co2_ppm),
             vapor pressure from surface T/S, and pco2_atmos (uatm)
    ocean:   CMEMS ADT (adt), ERA5 10 m wind speed (wind_speed), NSIDC sea ice (sea_ice;
             optionally sea_ice_avg_<N>d with --seaice-days-before N), clear-sky max_solar_rad,
             log_mld
Features are looked up with the vectorized functions in p1_features.py (see there for the
differences from argoclare.preprocessing, including the sea level pressure longitude fix).
Years are processed in parallel. Writes one netcdf along profid (longitude -180 to 180) to
working-vars/P1-processed/P1_coreArgo_application_features_yr<years>_acc<datetag>.nc.

Usage:
    python create_P1_core_features.py                                      # 2014-2023
    python create_P1_core_features.py --years 2018 2023 --max-profiles 500 --outpath /tmp/test/
"""
import argparse
import os
from datetime import datetime

import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

import argoclare.oceantools as mod_ocean
import p1_features

local_data_folder = p1_features.local_data_folder
default_index = local_data_folder + 'core/L3-interp/coreINDEX_byProfid_valid_interp_2014-2024_acc20261015.nc'
default_outpath = local_data_folder + 'working-vars/P1-processed/'


def process_year(yearDF, yr, seaice_days_before=None):
    t0 = datetime.now()
    yearDF = p1_features.add_all_features(yearDF, yr, seaice_days_before=seaice_days_before)
    print(f'  {yr}: {len(yearDF)} profiles in {(datetime.now() - t0).total_seconds() / 60:.1f} min')
    return yearDF


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--core-index', default=default_index, help='L3 coreINDEX netcdf')
    parser.add_argument('--years', type=int, nargs='+', default=list(range(2014, 2024)))
    parser.add_argument('--outpath', default=default_outpath, help='Folder to save the P1 netcdf in')
    parser.add_argument('--n-jobs', type=int, default=4, help='Years to process in parallel')
    parser.add_argument('--max-profiles', type=int, default=None,
                        help='Only use N random profiles from each year (for testing)')
    parser.add_argument('--seaice-days-before', type=int, default=None,
                        help='Also add sea_ice_avg_<N>d, the mean sea ice 1 to N-1 days before')
    args = parser.parse_args()

    print('Reading ' + args.core_index)
    coreDF = xr.open_dataset(args.core_index).to_dataframe().reset_index()
    coreDF['datetime'] = pd.to_datetime(coreDF['datetime'])
    coreDF = mod_ocean.set_longitude_range(coreDF, end_type='180')
    coreDF = coreDF[coreDF.datetime.dt.year.isin(args.years)]

    yearly = [(yr, df) for yr, df in coreDF.groupby(coreDF.datetime.dt.year)]
    if args.max_profiles:
        yearly = [(yr, df.sample(min(len(df), args.max_profiles), random_state=0).sort_index())
                  for yr, df in yearly]
    print(f'Adding features to {sum(len(df) for _, df in yearly)} profiles: ',
          datetime.now().strftime('%Y%m%d %H:%M:%S'))

    parts = Parallel(n_jobs=args.n_jobs)(delayed(process_year)(df, yr, args.seaice_days_before)
                                         for yr, df in yearly)
    coreP1 = pd.concat(parts).sort_values('datetime').reset_index(drop=True)
    coreP1['platform'] = 'coreArgo'

    datetag = datetime.now().strftime('%Y%m%d')
    yrs = f'{min(args.years)}-{max(args.years)}' if len(args.years) > 1 else str(args.years[0])
    os.makedirs(args.outpath, exist_ok=True)
    savefile = os.path.join(args.outpath, f'P1_coreArgo_application_features_yr{yrs}_acc{datetag}.nc')
    p1_features.save_netcdf(coreP1, 'profid', savefile, {
        'title': f'P1 core Argo application dataset {yrs}: regression features at the surface '
                 'location and time of each L3 profile (one row per profid).',
        'source': os.path.basename(args.core_index),
        'script': os.path.basename(__file__),
        'date': datetag})
    print(f'Saved {len(coreP1)} profiles to {savefile}: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))


if __name__ == '__main__':
    main()
