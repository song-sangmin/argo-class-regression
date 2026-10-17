"""Create the synthetic SOCAT P1 dataset: model surface features and colocation to synthetic floats.

Same steps as 4.4_synthetic_SOCAT.ipynb, following the real SOCAT workflow
(data-processing/create_P1_socat_features.py), one row per synthetic SOCAT sample:
    1. Read the synthetic profiles matched to each real SOCAT sample (synSOCAT-daily_matches.nc,
       from perlmutter-synthetic/extract_SOCAT.py; sample_datetag x 3 model levels)
    2. Model surface variables as for the synthetic floats (create_L3_synthetic_argo.py
       process_surface(): wind speed from stress, pCO2surface == 0 as missing, real-data names,
       QC south of 35S and atmospheric pressure > 0.88 atm). The model MLD at the sample is
       kept as mld_model only.
    3. CT and SA from the 3 model levels, regridded to 0, 5, 10 dbar (pchip, as 4.4); the
       surface values are taken at 5 dbar, as for the synthetic floats
    4. Colocation as for the real SOCAT data: the nearest synthetic profile within +-7 days
       (--window-days) by the weighted distance sqrt(dlon^2/Lx^2 + dlat^2), among synthetic
       core + bgc L3 profiles with CT and SA at every level < --dbar-limit (the profiles the
       synthetic fit clusters). mld and mlp come from that profile, like the real SOCAT data.
       Samples with no profile in the window are kept with nearest_profid '' (dropped in P2).
    5. Time variables, max_solar_rad and log_mld (p1_features)
Writes one netcdf along 'sample' (longitude -180 to 180) to working-vars/synthetic/P1-processed/:
    P1_synSOCAT_features_yr<years>_acc<datetag>.nc  (platform 'synSOCAT')

Usage:
    python create_P1_synthetic_socat.py --indexes <syn core INDEX> <syn bgc INDEX> --datas <syn core DATA> <syn bgc DATA>
"""
import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

import argoclare.oceantools as mod_ocean
import create_L3_synthetic_argo as l3syn

scripts_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(scripts_folder)
sys.path.append(os.path.join(scripts_folder, 'data-processing'))
import importer # noqa: E402
import p1_features # noqa: E402
import create_P1_socat_features as socat_p1 # noqa: E402

default_socat = importer.local_working_folder + 'synthetic/synSOCAT-daily_matches.nc'
default_outpath = importer.local_working_folder + 'synthetic/P1-processed/'
socat_levels = [0, 5, 10] # as 4.4
surface_pressure = l3syn.surface_pressure


def read_synthetic_socat(path):
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
    df['linear_time'] = (df['datetime'] - pd.Timestamp(p1_features.ref_time)) / pd.Timedelta(days=1)
    return df.drop(columns=[c for c in ['record_index', 'nParticles_index'] if c in df.columns])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--socat', default=default_socat, help='Synthetic SOCAT matches netcdf')
    parser.add_argument('--indexes', nargs=2, required=True, help='Synthetic core and bgc L3 INDEX (core first)')
    parser.add_argument('--datas', nargs=2, required=True, help='Matching synthetic core and bgc L3 DATA')
    parser.add_argument('--dbar-limit', type=float, default=500, help='Clusterable: CT and SA at all levels < this')
    parser.add_argument('--window-days', type=float, default=7, help='Colocation time window (+- days)')
    parser.add_argument('--Lx', type=float, default=2.5, help='Longitude weight in the colocation distance')
    parser.add_argument('--outpath', default=default_outpath)
    args = parser.parse_args()

    df = read_synthetic_socat(args.socat)

    print(f'Finding clusterable synthetic profiles (CT and SA above {args.dbar_limit:g} dbar): ',
          datetime.now().strftime('%Y%m%d %H:%M:%S'))
    floats = socat_p1.clusterable_profiles(args.indexes, args.datas, args.dbar_limit)
    df = socat_p1.find_nearest_float(df, floats, window_days=args.window_days, Lx=args.Lx)
    n_found = int((df.nearest_profid != '').sum())
    print(f'Colocated {n_found} of {len(df)} samples within {args.window_days:g} days; '
          f'median km_sep {df.km_sep.median():.0f} km')

    df = p1_features.add_time_vars(df)
    df = p1_features.add_solar_radiation(df)
    df['log_mld'] = np.log(df['mld'])
    df['platform'] = 'synSOCAT'
    df = df.sort_values(['cruiseid', 'datetime']).reset_index(drop=True)

    datetag = datetime.now().strftime('%Y%m%d')
    first, last = df.datetime.dt.year.min(), df.datetime.dt.year.max()
    yrs = f'{first}-{last}' if last > first else str(first)
    os.makedirs(args.outpath, exist_ok=True)
    savefile = os.path.join(args.outpath, f'P1_synSOCAT_features_yr{yrs}_acc{datetag}.nc')
    p1_features.save_netcdf(df, 'sample', savefile, {
        'title': f'P1 synthetic SOCAT dataset {yrs}: model (E3SM) surface variables at each SOCAT sample, '
                 'CT/SA at 5 dbar, and the nearest clusterable synthetic profile (mld from that profile).',
        'colocation': f'weighted distance sqrt(dlon^2/{args.Lx:g}^2 + dlat^2) within +-{args.window_days:g} '
                      f'days, among synthetic profiles with CT and SA at all levels < {args.dbar_limit:g} dbar in: '
                      + ', '.join(os.path.basename(f) for f in args.indexes),
        'dbar_limit': args.dbar_limit, 'samples_colocated': n_found,
        'source': os.path.basename(args.socat), 'script': os.path.basename(__file__), 'date': datetag})
    print(f'Saved {len(df)} samples to {savefile}')


if __name__ == '__main__':
    main()
