"""Create the synthetic SOCAT P1 dataset: model surface features and colocation to synthetic floats.

Same steps as 4.4_synthetic_SOCAT.ipynb, following the real SOCAT workflow
(data-processing/create_P1_socat_features.py), one row per synthetic SOCAT sample:
    1. Read the synthetic profiles matched to each real SOCAT sample (synSOCAT-daily_matches.nc,
       from PKG-synthetic-extract/extract_SOCAT.py; sample_datetag x 3 model levels)
    2. Model surface variables as for the synthetic floats (create_L3_synthetic_argo.py
       process_surface(): wind speed from stress, pCO2surface == 0 as missing, real-data names,
       QC south of 35S and atmospheric pressure > 0.88 atm). The model MLD at the sample is
       kept as mld_model only.
    3. CT and SA from the 3 model levels, regridded to 0, 5, 10 dbar (pchip, as 4.4); the
       surface values are taken at 5 dbar, as for the synthetic floats
    4. Colocation as for the real SOCAT data: the nearest synthetic profile within +-7 days
       ([P1.socat] window_days) by the weighted distance sqrt(dlon^2/Lx^2 + dlat^2), among synthetic
       core + bgc L3 profiles with CT and SA at every level < [P1.socat] dbar_limit (the profiles the
       synthetic fit clusters). mld and mlp come from that profile, like the real SOCAT data.
       Samples with no profile in the window are kept with nearest_profid '' (dropped in P2).
    5. Time variables, max_solar_rad and log_mld (p1_features)
Reads [files] syn_socat and the synthetic L3 core + bgc files, and writes [files] P1_socat, one
netcdf along 'sample' (longitude -180 to 180; platform 'synSOCAT').

Usage:
    python create_P1_synthetic_socat.py --config ../../configs/synthetic_CRUSOE.toml
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

import argoclare.oceantools as mod_ocean
from argoclare import pipeline
import create_L3_synthetic_argo as l3syn

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data-processing'))
import p1_features # noqa: E402
import create_P1_socat_features as socat_p1 # noqa: E402

socat_levels = [0, 5, 10] # as 4.4
surface_pressure = l3syn.surface_pressure


def read_synthetic_socat(path, ref_time=p1_features.ref_time):
    """Model surface variables (after QC) with CT/SA at 5 dbar, one row per sample."""
    syn = xr.open_dataset(path)
    surf = syn[l3syn.surface_vars].drop_dims('nVertLevels', errors='ignore').to_dataframe().reset_index()
    n_all = len(surf)
    surf = l3syn.process_surface(surf).rename(columns={'mld': 'mld_model'}).set_index('sample_datetag')
    print(f'{n_all} synthetic SOCAT samples; {len(surf)} after QC (south of 35S, atmos pressure > 0.88 atm)')

    prof = syn[['Temperature', 'Salinity', 'Depth', 'lat', 'lon']].to_dataframe().reset_index()
    prof = prof.dropna(subset=['Temperature', 'Salinity', 'Depth', 'lat'])
    prof = prof[prof['sample_datetag'].isin(surf.index)].rename(columns={'lat': 'latitude', 'lon': 'longitude'})
    prof = l3syn.add_ct_sa(prof).assign(wmoid='socat').set_index('sample_datetag')
    prof.index.name = 'profid' # regrid_pressure_levels groups by 'profid'
    regular = l3syn.regrid_ct_sa(prof, socat_levels).reset_index()
    at5 = regular[regular['pressure'] == surface_pressure].set_index('profid')[['CT', 'SA']]

    df = surf.join(at5, how='inner').rename_axis('sample_datetag').reset_index()
    print(f'  {len(df)} samples with CT/SA at {surface_pressure} dbar')
    df = mod_ocean.set_longitude_range(df, end_type='180')
    df['cruiseid'] = df['sample_datetag'].str.split('_').str[0]
    df['linear_time'] = (df['datetime'] - pd.Timestamp(ref_time)) / pd.Timedelta(days=1)
    return df.drop(columns=[c for c in ['record_index', 'nParticles_index'] if c in df.columns])


def main():
    parser = pipeline.parser(__doc__)
    args = parser.parse_args()
    run = pipeline.start(args, __file__)
    cfg = run.section('P1.socat')
    args.dbar_limit, args.window_days, args.Lx = cfg['dbar_limit'], cfg['window_days'], cfg['Lx']
    args.socat = run.input('syn_socat')
    args.indexes = [run.input('L3_index', 'core'), run.input('L3_index', 'bgc')]
    args.datas = [run.input('L3_data', 'core'), run.input('L3_data', 'bgc')]
    savefile = run.output('P1', 'socat')

    df = read_synthetic_socat(args.socat, run.ref_time)

    print(f'Finding clusterable synthetic profiles (CT and SA above {args.dbar_limit:g} dbar): ',
          datetime.now().strftime('%Y%m%d %H:%M:%S'))
    floats = socat_p1.clusterable_profiles(args.indexes, args.datas, args.dbar_limit, run.ref_time)
    df = socat_p1.find_nearest_float(df, floats, window_days=args.window_days, Lx=args.Lx, ref_time=run.ref_time)
    n_found = int((df.nearest_profid != '').sum())
    print(f'Colocated {n_found} of {len(df)} samples within {args.window_days:g} days; '
          f'median km_sep {df.km_sep.median():.0f} km')

    df = p1_features.add_time_vars(df, run.ref_time)
    df = p1_features.add_solar_radiation(df)
    df['log_mld'] = np.log(df['mld'])
    df['platform'] = 'synSOCAT'
    df = df.sort_values(['cruiseid', 'datetime']).reset_index(drop=True)

    datetag = datetime.now().strftime('%Y%m%d')
    first, last = df.datetime.dt.year.min(), df.datetime.dt.year.max()
    yrs = f'{first}-{last}' if last > first else str(first)
    p1_features.save_netcdf(df, 'sample', savefile, {
        'title': f'P1 synthetic SOCAT dataset {yrs}: model (E3SM) surface variables at each SOCAT sample, '
                 'CT/SA at 5 dbar, and the nearest clusterable synthetic profile (mld from that profile).',
        'colocation': f'weighted distance sqrt(dlon^2/{args.Lx:g}^2 + dlat^2) within +-{args.window_days:g} '
                      f'days, among synthetic profiles with CT and SA at all levels < {args.dbar_limit:g} dbar in: '
                      + ', '.join(os.path.basename(f) for f in args.indexes),
        'dbar_limit': args.dbar_limit, 'samples_colocated': n_found,
        'source': os.path.basename(args.socat), **run.attrs(), 'date': datetag})
    print(f'Saved {len(df)} samples to {savefile}')
    run.finish(samples_colocated=n_found)


if __name__ == '__main__':
    main()
