"""Create synthetic Argo P1 datasets: regression features for each synthetic L3 profile.

The synthetic features come from the E3SM model itself (model-consistent), as in 4.2: sea_ice,
mld, wind_speed, atmos_pres_Pa/_atm, pco2_ocean, pco2_atmos, delta_pco2 and CO2_gas_flux are
already in the synthetic L3 INDEX (create_L3_synthetic_argo.py), with CT and SA from the
shallowest model level. This adds, with the same functions as the real P1 data (p1_features):
    time:  year, month, day, linear_time (days since 2014-01-01), ydcos, ydsin, decimalyr
    ocean: max_solar_rad (clear-sky, from time and latitude), log_mld
There is no ADT (the model output has no sea surface height). Years run in parallel.
Reads [files] L3_index for bgc and core, and writes netcdfs along profid (longitude -180 to 180):
    [files] P1_bgc   (platform 'synBgcArgo')
    [files] P1_core  (platform 'synCoreArgo')

Usage:
    python create_P1_synthetic_features.py --config ../../configs/synthetic_CRUSOE.toml
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

import argoclare.oceantools as mod_ocean
from argoclare import pipeline

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data-processing'))
import p1_features # noqa: E402


def process_year(yearDF, yr, ref_time):
    df = p1_features.add_time_vars(yearDF, ref_time)
    df = p1_features.add_solar_radiation(df)
    df['log_mld'] = np.log(df['mld'])
    print(f'  {yr}: {len(df)} profiles')
    return df


def make_p1(index_file, platform, n_jobs, ref_time):
    df = xr.open_dataset(index_file).to_dataframe().reset_index()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = mod_ocean.set_longitude_range(df, end_type='180')
    parts = Parallel(n_jobs=n_jobs)(delayed(process_year)(g, yr, ref_time) for yr, g in df.groupby(df.datetime.dt.year))
    df = pd.concat(parts).sort_values(['wmoid', 'datetime']).reset_index(drop=True)
    df['platform'] = platform
    return df


def main():
    parser = pipeline.parser(__doc__)
    parser.add_argument('--n-jobs', type=int, default=None, help='Years to process in parallel (default: [P1] n_jobs)')
    args = parser.parse_args()
    run = pipeline.start(args, __file__)
    n_jobs = args.n_jobs or run.section('P1').get('n_jobs', 4)
    jobs = [(name, run.input('L3_index', name), platform, run.output('P1', name))
            for name, platform in [('bgc', 'synBgcArgo'), ('core', 'synCoreArgo')]]

    datetag = datetime.now().strftime('%Y%m%d')
    for name, index_file, platform, savefile in jobs:
        print(f'===> {name}: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))
        df = make_p1(index_file, platform, n_jobs, run.ref_time)
        first, last = df.datetime.dt.year.min(), df.datetime.dt.year.max()
        yrs = f'{first}-{last}' if last > first else str(first)
        p1_features.save_netcdf(df, 'profid', savefile, {
            'title': f'P1 synthetic {name} Argo dataset {yrs}: model (E3SM) surface variables and '
                     'regression features at each synthetic profile (one row per profid).',
            'source': os.path.basename(index_file),
            'features': 'sea_ice, mld, wind_speed, atmos_pres, pco2 from the model; time variables, '
                        'max_solar_rad and log_mld added; no ADT',
            **run.attrs(), 'date': datetag})
        print(f'  saved {len(df)} profiles to {savefile}')
    run.finish()


if __name__ == '__main__':
    main()
