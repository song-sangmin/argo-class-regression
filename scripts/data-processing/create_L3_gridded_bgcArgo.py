"""Create L3 gridded BGC Argo datasets from the L1 QC'ed profiles.

Same steps as 0.4_bgcArgo_processing.ipynb (sections 3-5), one year at a time:
    1. L2 mask: keep profiles in bgc_open_ocean_INDEX_<yr>.nc (bathymetry < -1000 m)
    2. Read those profiles from bgc_QCflags1258_wO2_yr<yr>_<tag>.csv (L1), keep only the latest cast of each
       profid (see below), and keep profiles whose shallowest QC'ed observation (incl. pH)
       is within 25 dbar of the surface
    3. Regrid T/S and pH to 0-1000 dbar every 5 dbar and 1050-1950 dbar every 50 dbar with argoclare's regrid_pressure_levels()
       (with the interpolate_z_profile() fix that keeps levels falling exactly
       on the first/last observation, e.g. a first observation at 5.0 dbar)
    4. Add mixed layer pressure 'mlp' (0.03 kg/m3 sigma0 threshold from 10 dbar) and
       mixed layer depth 'mld' (positive, m) with argoclare's add_mixedlayer_pressure()
profid is wmoid + cycle number, so a cycle's descending (*D.nc) and ascending casts share a
profid: they were merged in L1, or split across files at a year boundary. The ascending cast
is the last of the cycle, so only the latest cast is kept; the number of descending casts
dropped is printed and saved in the 'descending_casts_dropped' attribute.
Each profile also gets a 'prof_datetag' (wmoid_YYYYMMDDHH) coordinate along profid.
Writes bgcDATA_* (profid, pressure) and bgcINDEX_* (profid) to bgc/L3-interp/.

Usage:
    python create_L3_gridded_bgcArgo.py                                    # 2014-2024
    python create_L3_gridded_bgcArgo.py --years 2014 --max-profiles 200 --outpath /tmp/test/
"""
import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

import argoclare.argotools as mod_argo

local_data_folder = '/Users/sangminsong/Documents/PROJECTS-PYTHON/CRUSOE-SOCarbon/x_localdata/'
default_l1_path = local_data_folder + 'bgc/L1-qc/'
default_l2_path = local_data_folder + 'bgc/L2-mask/'
default_outpath = local_data_folder + 'bgc/L3-interp/'
default_l1_tag = 'acc20260731'

ref_time = '2014-01-01'
pres_regular = np.arange(0, 1001, 5).tolist() + np.arange(1050, 1951, 50).tolist()
bgc_list = ['pH']
read_cols = ['profid', 'pressure', 'wmoid', 'cycle_number', 'latitude', 'longitude', 'datetime',
             'CT', 'SA', 'sigma0', 'spice', 'temperature', 'salinity'] + bgc_list


def open_ocean_index(yr, l2_path):
    """L2 index for one year: open ocean profiles (bathymetry < -1000 m)."""
    ds = xr.open_dataset(os.path.join(l2_path, f'bgc_open_ocean_INDEX_{yr}.nc')).load()
    ds['datetime'] = ('profid', pd.to_datetime(ds.datetime.values.astype(str), format='mixed'))
    return ds


def read_l1_year(path, profids, chunksize=5_000_000):
    """Read the selected profiles from one L1 csv; returns the data and # of descending casts dropped."""
    profids = set(profids)
    df = pd.concat([chunk[chunk.profid.isin(profids)]
                    for chunk in pd.read_csv(path, usecols=read_cols, chunksize=chunksize)])
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')

    # Descending and ascending casts of a cycle share a profid; keep the later (ascending) one
    n_desc = int((df.groupby('profid')['datetime'].nunique() > 1).sum())
    df = df[df['datetime'] == df.groupby('profid')['datetime'].transform('max')]

    df['linear_time'] = (df['datetime'] - pd.Timestamp(ref_time)) / pd.Timedelta(days=1)
    # Pchip needs at least 2 observations
    df = df[df.groupby('profid').pressure.transform('size') >= 2]
    return df.set_index('profid'), n_desc


def grid_chunk(chunkDF, chunkINDEX, bgc_list):
    """Regrid a group of profiles, then add mixed layer pressure/depth to the data and index."""
    chunk_regular = mod_argo.regrid_pressure_levels(chunkDF, pres_regular, bgc_list=bgc_list,
                                                    z_gap=1000, surface_fill=25, ref_time=ref_time)
    chunk_regular['datetime'] = pd.Timestamp(ref_time) + pd.to_timedelta(chunk_regular['linear_time'], unit='D')
    chunkDS = mod_argo.to_xr_dataset(chunk_regular, {})
    chunkINDEX = chunkINDEX.sel(profid=chunkDS.profid.values)
    return mod_argo.add_mixedlayer_pressure(chunkINDEX, chunkDS)


def grid_parallel(df, yearINDEX, bgc_list, n_jobs):
    """Run grid_chunk() on groups of profiles in parallel; returns lists of index and data chunks."""
    profids = df.index.unique()
    n_chunks = min(len(profids), n_jobs * 4)
    chunk_of = pd.Series(np.arange(len(profids)) % n_chunks, index=profids)
    parts = Parallel(n_jobs=n_jobs)(
        delayed(grid_chunk)(chunk, yearINDEX.sel(profid=chunk.index.unique().values), bgc_list)
        for _, chunk in df.groupby(df.index.map(chunk_of)))
    return [p[0] for p in parts], [p[1] for p in parts]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--years', type=int, nargs='+', default=list(range(2014, 2025)))
    parser.add_argument('--l1-path', default=default_l1_path, help='Folder with L1 QC csv files')
    parser.add_argument('--l1-tag', default=default_l1_tag, help='Date tag in the L1 file names')
    parser.add_argument('--l2-path', default=default_l2_path, help='Folder with L2 open ocean index files')
    parser.add_argument('--outpath', default=default_outpath, help='Folder to save L3 files in')
    parser.add_argument('--surface-max', type=float, default=25,
                        help='Keep profiles with a QC\'ed observation shallower than this (dbar)')
    parser.add_argument('--n-jobs', type=int, default=8, help='Parallel workers for regridding')
    parser.add_argument('--max-profiles', type=int, default=None,
                        help='Only regrid the first N profiles of each year (for testing)')
    args = parser.parse_args()

    data_parts, index_parts = [], []
    n_desc_total = 0
    for yr in args.years:
        print(f'===> {yr}: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))
        yearINDEX = open_ocean_index(yr, args.l2_path)
        yearDF, n_desc = read_l1_year(os.path.join(args.l1_path, f'bgc_QCflags1258_wO2_yr{yr}_{args.l1_tag}.csv'),
                                      yearINDEX.profid.values)
        n_desc_total += n_desc

        # Surfacemost observation of the kept cast (L2 values may come from the descending cast)
        first = yearDF.reset_index().sort_values('pressure').groupby('profid').first()
        first = first[first.pressure <= args.surface_max]
        if args.max_profiles:
            first = first.iloc[:args.max_profiles]
        yearDF = yearDF[yearDF.index.isin(first.index)]
        yearINDEX = yearINDEX.sel(profid=first.index.values)
        for var in ['pressure', 'latitude', 'longitude', 'datetime', 'CT', 'SA', 'sigma0', 'spice',
                    'temperature', 'salinity']:
            yearINDEX[var] = ('profid', first[var].values)

        print(f'  {yearINDEX.sizes["profid"]} open ocean profiles with surface data '
              f'({n_desc} descending casts dropped); regridding and adding MLD...')

        year_index_parts, year_data_parts = grid_parallel(yearDF, yearINDEX, bgc_list=bgc_list, n_jobs=args.n_jobs)
        index_parts += year_index_parts
        data_parts += year_data_parts
        print(f'  regridded {sum(p.sizes["profid"] for p in year_data_parts)} profiles: ',
              datetime.now().strftime('%Y%m%d %H:%M:%S'))

    # Chunks only hold the pressure levels their profiles reach, so line them up on the full grid
    bgcDS = xr.concat(data_parts, dim='profid').reindex(pressure=pres_regular)
    bgcINDEX = xr.concat(index_parts, dim='profid') # surfacemost observed values, plus mlp/mld
    assert (bgcINDEX.profid.values == bgcDS.profid.values).all()

    # A descending cast at the end of one year and its ascending cast in the next share a profid;
    # keep the latest one
    casts = pd.DataFrame({'profid': bgcINDEX.profid.values, 't': bgcINDEX.datetime.values})
    keep = ~casts.sort_values('t').duplicated('profid', keep='last').sort_index().values
    n_desc_total += int((~keep).sum())
    bgcDS, bgcINDEX = bgcDS.isel(profid=keep), bgcINDEX.isel(profid=keep)
    print(f'Dropped {n_desc_total} descending casts in total ({int((~keep).sum())} across year boundaries)')

    datetag = datetime.now().strftime('%Y%m%d')
    yrs = f'{min(args.years)}-{max(args.years)}' if len(args.years) > 1 else str(args.years[0])
    bgcDS = bgcDS.assign_attrs({
        'title': f'BGC float profiles {yrs} with valid data (QC flags 1,2,5,8 incl. pH), masked to open '
                 'ocean bathymetry (elevation < -1000 m), with valid pH observations above 25 dbar. '
                 f'Interpolated (pchip) to n={len(pres_regular)} pressure levels: 0-1000 dbar every 5 dbar, '
                 '1050-1950 dbar every 50 dbar. '
                 'Includes mixed layer pressure (mlp, dbar) and depth (mld, m) from a 0.03 kg/m3 '
                 'sigma0 threshold relative to 10 dbar.',
        'pressure_levels': 'Pressure levels in dbar: ' + str(pres_regular),
        'source': 'Argopy, expert mode; ERDDAP snapshot Feb 2025',
        'script': os.path.basename(__file__),
        'descending_casts_dropped': n_desc_total,
        'date': datetag})

    # One prof_datetag (wmoid_YYYYMMDDHH) per profile, same format as index_by_datetag()
    prof_datetag = np.char.add(bgcINDEX.wmoid.values.astype(int).astype(str) + '_',
                               pd.to_datetime(bgcINDEX.datetime.values).strftime('%Y%m%d%H').to_numpy(dtype=str))
    bgcINDEX = bgcINDEX.assign_coords(prof_datetag=('profid', prof_datetag))
    bgcDS = bgcDS.assign_coords(prof_datetag=('profid', prof_datetag)) # same profid order as bgcINDEX
    n_dup = pd.Index(prof_datetag).duplicated().sum()
    if n_dup > 0:
        print(f'Note: {n_dup} prof_datetags are shared by more than one profid')
    bgcINDEX = bgcINDEX.assign_attrs({
        'title': 'Surface locations of BGC Argo profiles in bgcDATA, masked to open ocean bathymetry '
                 'where elevation < -1000m. Variables of pressure, CT, SA... represent surfacemost '
                 'observation within the profile.',
        'source': 'Argopy, expert mode; ERDDAP snapshot Feb 2025',
        'script': os.path.basename(__file__),
        'descending_casts_dropped': n_desc_total,
        'date': datetag})

    os.makedirs(args.outpath, exist_ok=True)
    data_file = os.path.join(args.outpath, f'bgcDATA_byProfid_valid_interp_{yrs}_acc{datetag}.nc')
    index_file = os.path.join(args.outpath, f'bgcINDEX_byProfid_valid_interp_{yrs}_acc{datetag}.nc')
    # Drop encodings inherited from the L2 files (e.g. profid stored as <U13, which would
    # truncate 4-digit cycle numbers like 7900947cyc1029)
    bgcDS.drop_encoding().to_netcdf(data_file)
    bgcINDEX.drop_encoding().to_netcdf(index_file)
    print(f'Saved {bgcDS.sizes["profid"]} profiles to {data_file}')
    print(f'Saved index to {index_file}: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))


if __name__ == '__main__':
    main()
