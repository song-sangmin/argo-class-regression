"""Create the P1 float datasets (core and/or bgc Argo): regression features for each L3 profile.

Same features as 1.1_P1_coreArgo_feature_prep.ipynb (section I), computed from the L3 index
(one row per profile: surfacemost CT, SA (and pH etc. for bgc), and mld/mlp) instead of every
pressure level:
    time:    year, month, day, linear_time (days since [run] ref_time), ydcos, ydsin, decimalyr
    atmos:   ERA5 sea level pressure (atmos_pres_Pa/_atm), MBL xCO2 (atmos_co2_ppm),
             vapor pressure from surface T/S, and pco2_atmos (uatm)
    ocean:   CMEMS ADT (adt), ERA5 10 m wind speed (wind_speed), NSIDC sea ice (sea_ice;
             optionally sea_ice_avg_<N>d with [P1] seaice_days_before = N), clear-sky
             max_solar_rad, log_mld
Only the features the run uses are added: [P1] features, plus the [regression] features and
target (the time variables, max_solar_rad and log_mld always). The gridded files the others
need (p1_features.FILE_COLUMNS) must exist for every year; the rest are not read. Without
features in the config, all of them are added.
Features are looked up with the vectorized functions in p1_features.py (see there for the
differences from argoclare.preprocessing, including the sea level pressure longitude fix).
For each of --datasets (in the order given), years ([P1.<dataset>] years, else [run] years) are
processed in parallel. Reads [files] L3_index for the dataset and writes [files] P1_<dataset>,
one netcdf along profid (longitude -180 to 180), with platform 'coreArgo' or 'bgcArgo'. Every
input and output is checked before any processing.
The CRUSOE run uses the SOCCOM surface averages for bgc instead (create_P1_soccom_features.py).

Usage:
    python create_P1_float_features.py --config ../../configs/example.toml --datasets core bgc
    python create_P1_float_features.py --config ../../configs/example.toml --datasets bgc --max-profiles 500 --out-dir /tmp/test/
"""

import os
from datetime import datetime

import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

import argoclare.oceantools as mod_ocean
from argoclare import pipeline
import p1_features

platforms = {'core': 'coreArgo', 'bgc': 'bgcArgo'}


def process_year(yearDF, yr, files, ref_time, seaice_days_before=None):
    t0 = datetime.now()
    yearDF = p1_features.add_all_features(yearDF, yr, files, ref_time, seaice_days_before=seaice_days_before)
    print(f'  {yr}: {len(yearDF)} profiles in {(datetime.now() - t0).total_seconds() / 60:.1f} min')
    return yearDF


def make_dataset(run, cfg, dataset, years, index_file, files, savefile, args):
    """Add the features to one dataset's L3 profiles and save [files] P1_<dataset>."""
    print(f'===> {dataset}: reading ' + index_file)
    floatDF = xr.open_dataset(index_file).to_dataframe().reset_index()
    floatDF['datetime'] = pd.to_datetime(floatDF['datetime'])
    floatDF = mod_ocean.set_longitude_range(floatDF, end_type='180')
    floatDF = floatDF[floatDF.datetime.dt.year.isin(years)]

    yearly = [(yr, df) for yr, df in floatDF.groupby(floatDF.datetime.dt.year)]
    if args.max_profiles:
        yearly = [(yr, df.sample(min(len(df), args.max_profiles), random_state=0).sort_index()) for yr, df in yearly]
    print(
        f'Adding features to {sum(len(df) for _, df in yearly)} profiles: ', datetime.now().strftime('%Y%m%d %H:%M:%S')
    )

    parts = Parallel(n_jobs=args.n_jobs or cfg.get('n_jobs', 1))(
        delayed(process_year)(df, yr, files, run.ref_time, cfg.get('seaice_days_before')) for yr, df in yearly
    )
    floatP1 = pd.concat(parts).sort_values('datetime').reset_index(drop=True)
    floatP1['platform'] = platforms[dataset]

    datetag = datetime.now().strftime('%Y%m%d')
    yrs = f'{min(years)}-{max(years)}' if len(years) > 1 else str(years[0])
    label = 'core Argo application' if dataset == 'core' else 'BGC Argo float'
    p1_features.save_netcdf(
        floatP1,
        'profid',
        savefile,
        {
            'title': f'P1 {label} dataset {yrs}: regression features at the surface '
            'location and time of each L3 profile (one row per profid).',
            'source': os.path.basename(index_file),
            **run.attrs(),
            'date': datetag,
        },
    )
    print(f'Saved {len(floatP1)} profiles to {savefile}: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))


def main():
    parser = pipeline.add_skip_arg(pipeline.add_datasets_arg(pipeline.parser(__doc__)))
    parser.add_argument('--years', type=int, nargs='+', default=None, help='Default: [P1.<dataset>] or [run] years')
    parser.add_argument('--n-jobs', type=int, default=None, help='Years to process in parallel (default: [P1] n_jobs)')
    parser.add_argument(
        '--max-profiles', type=int, default=None, help='Only use N random profiles from each year (for testing)'
    )
    args = parser.parse_args()
    run = pipeline.start(args, __file__)
    cfg = run.section('P1')

    # Every input and output checked before any processing
    jobs = []
    for dataset in [ds for ds in pipeline.datasets(args) if not run.skip(['P1'], ds)]:
        years = run.years(f'P1.{dataset}')
        jobs.append(
            (
                dataset,
                years,
                run.input('L3_index', dataset),
                p1_features.run_feature_files(run, years),
                run.output('P1', dataset),
            )
        )
    if not jobs:
        return run.nothing_to_do()
    for job in jobs:
        make_dataset(run, cfg, *job, args)
    run.finish()


if __name__ == '__main__':
    main()
