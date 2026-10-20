"""Create synthetic Argo P1 datasets: regression features for each synthetic L3 profile.

The synthetic features come from the E3SM model itself (model-consistent), as in 4.2: sea_ice,
mld, wind_speed, atmos_pres_Pa/_atm, pco2_ocean, pco2_atmos, delta_pco2 and CO2_gas_flux are
already in the synthetic L3 INDEX (create_L3_synthetic_argo.py), with CT and SA from the
shallowest model level. This adds, with the same functions as the real P1 data (p1_features):
    time:  year, month, day, linear_time (days since 2014-01-01), ydcos, ydsin, decimalyr
    ocean: max_solar_rad (clear-sky, from time and latitude), log_mld
There is no ADT (the model output has no sea surface height). Years run in parallel.
Writes netcdfs along profid (longitude -180 to 180) to working-vars/synthetic/P1-processed/:
    P1_synArgo_bgc_features_yr<years>_acc<datetag>.nc               (platform 'synBgcArgo')
    P1_synArgo_core_application_features_yr<years>_acc<datetag>.nc  (platform 'synCoreArgo')

Usage:
    python create_P1_synthetic_features.py --bgc-index <synthetic_bgcINDEX...nc> --core-index <synthetic_coreINDEX...nc>
"""
import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

import argoclare.oceantools as mod_ocean

scripts_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(scripts_folder)
sys.path.append(os.path.join(scripts_folder, 'data-processing'))
import argoclare.importer as importer # noqa: E402
import p1_features # noqa: E402

default_outpath = importer.local_working_folder + 'synthetic/P1-processed/'


def process_year(yearDF, yr):
    df = p1_features.add_time_vars(yearDF)
    df = p1_features.add_solar_radiation(df)
    df['log_mld'] = np.log(df['mld'])
    print(f'  {yr}: {len(df)} profiles')
    return df


def make_p1(index_file, platform, n_jobs):
    df = xr.open_dataset(index_file).to_dataframe().reset_index()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = mod_ocean.set_longitude_range(df, end_type='180')
    parts = Parallel(n_jobs=n_jobs)(delayed(process_year)(g, yr) for yr, g in df.groupby(df.datetime.dt.year))
    df = pd.concat(parts).sort_values(['wmoid', 'datetime']).reset_index(drop=True)
    df['platform'] = platform
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bgc-index', required=True, help='Synthetic bgc L3 INDEX netcdf')
    parser.add_argument('--core-index', required=True, help='Synthetic core L3 INDEX netcdf')
    parser.add_argument('--outpath', default=default_outpath)
    parser.add_argument('--n-jobs', type=int, default=4, help='Years to process in parallel')
    args = parser.parse_args()

    os.makedirs(args.outpath, exist_ok=True)
    datetag = datetime.now().strftime('%Y%m%d')
    for name, index_file, platform, tag in [('bgc', args.bgc_index, 'synBgcArgo', 'bgc'),
                                            ('core', args.core_index, 'synCoreArgo', 'core_application')]:
        print(f'===> {name}: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))
        df = make_p1(index_file, platform, args.n_jobs)
        first, last = df.datetime.dt.year.min(), df.datetime.dt.year.max()
        yrs = f'{first}-{last}' if last > first else str(first)
        savefile = os.path.join(args.outpath, f'P1_synArgo_{tag}_features_yr{yrs}_acc{datetag}.nc')
        p1_features.save_netcdf(df, 'profid', savefile, {
            'title': f'P1 synthetic {name} Argo dataset {yrs}: model (E3SM) surface variables and '
                     'regression features at each synthetic profile (one row per profid).',
            'source': os.path.basename(index_file),
            'features': 'sea_ice, mld, wind_speed, atmos_pres, pco2 from the model; time variables, '
                        'max_solar_rad and log_mld added; no ADT',
            'script': os.path.basename(__file__), 'date': datetag})
        print(f'  saved {len(df)} profiles to {savefile}')


if __name__ == '__main__':
    main()
