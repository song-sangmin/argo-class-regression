"""Create L3 gridded BGC Argo datasets from the L1 QC'ed profiles.

Same steps as 0.4_bgcArgo_processing.ipynb (sections 3-5), one year at a time:
    1. L2 mask: keep profiles in the [files] L2 open ocean index (bathymetry < -1000 m)
    2. Read those profiles from the [files] L1 csv, keep only the latest cast of each
       profid (see below), and keep profiles whose shallowest QC'ed observation (incl. pH)
       is within [L3] surface_max (25) dbar of the surface
    3. Regrid T/S and [L3] bgc_vars (pH) to 0-1000 dbar every 5 dbar and 1050-1950 dbar every 50 dbar with argoclare's regrid_pressure_levels()
       (with the interpolate_z_profile() fix that keeps levels falling exactly
       on the first/last observation, e.g. a first observation at 5.0 dbar)
    4. Add mixed layer pressure 'mlp' (0.03 kg/m3 sigma0 threshold from 10 dbar) and
       mixed layer depth 'mld' (positive, m) with argoclare's add_mixedlayer_pressure()
profid is wmoid + cycle number, so a cycle's descending (*D.nc) and ascending casts share a
profid: they were merged in L1, or split across files at a year boundary. The ascending cast
is the last of the cycle, so only the latest cast is kept; the number of descending casts
dropped is printed and saved in the 'descending_casts_dropped' attribute.
Each profile also gets a 'prof_datetag' (wmoid_YYYYMMDDHH) coordinate along profid.
Writes [files] L3_data (profid, pressure) and L3_index (profid) for bgc.

Usage:
    python create_L3_gridded_bgcArgo.py --config ../../configs/example.toml
    python create_L3_gridded_bgcArgo.py --config ../../configs/example.toml --years 2020 --max-profiles 200 --out-dir /tmp/test/
"""
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

import argoclare.argotools as mod_argo
from argoclare import pipeline


pres_regular = np.arange(0, 1001, 5).tolist() + np.arange(1050, 1951, 50).tolist()
read_cols = ['profid', 'pressure', 'wmoid', 'cycle_number', 'latitude', 'longitude', 'datetime',
             'CT', 'SA', 'sigma0', 'spice', 'temperature', 'salinity']


def open_ocean_index(path):
    """L2 index for one year: open ocean profiles (bathymetry < -1000 m)."""
    ds = xr.open_dataset(path).load()
    ds['datetime'] = ('profid', pd.to_datetime(ds.datetime.values.astype(str), format='mixed'))
    return ds


def read_l1_year(path, profids, ref_time, bgc_list, chunksize=5_000_000):
    """Read the selected profiles from one L1 csv; returns the data and # of descending casts dropped."""
    profids = set(profids)
    df = pd.concat([chunk[chunk.profid.isin(profids)]
                    for chunk in pd.read_csv(path, usecols=read_cols + bgc_list, chunksize=chunksize)])
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')

    # Descending and ascending casts of a cycle share a profid; keep the later (ascending) one
    n_desc = int((df.groupby('profid')['datetime'].nunique() > 1).sum())
    df = df[df['datetime'] == df.groupby('profid')['datetime'].transform('max')]

    df['linear_time'] = (df['datetime'] - pd.Timestamp(ref_time)) / pd.Timedelta(days=1)
    # Pchip needs at least 2 observations
    df = df[df.groupby('profid').pressure.transform('size') >= 2]
    return df.set_index('profid'), n_desc


def grid_chunk(chunkDF, chunkINDEX, bgc_list, ref_time):
    """Regrid a group of profiles, then add mixed layer pressure/depth to the data and index."""
    chunk_regular = mod_argo.regrid_pressure_levels(chunkDF, pres_regular, bgc_list=bgc_list,
                                                    z_gap=1000, surface_fill=25, ref_time=ref_time)
    chunk_regular['datetime'] = pd.Timestamp(ref_time) + pd.to_timedelta(chunk_regular['linear_time'], unit='D')
    chunkDS = mod_argo.to_xr_dataset(chunk_regular, {})
    chunkINDEX = chunkINDEX.sel(profid=chunkDS.profid.values)
    return mod_argo.add_mixedlayer_pressure(chunkINDEX, chunkDS)


def grid_parallel(df, yearINDEX, bgc_list, n_jobs, ref_time):
    """Run grid_chunk() on groups of profiles in parallel; returns lists of index and data chunks."""
    profids = df.index.unique()
    n_chunks = min(len(profids), n_jobs * 4)
    chunk_of = pd.Series(np.arange(len(profids)) % n_chunks, index=profids)
    parts = Parallel(n_jobs=n_jobs)(
        delayed(grid_chunk)(chunk, yearINDEX.sel(profid=chunk.index.unique().values), bgc_list, ref_time)
        for _, chunk in df.groupby(df.index.map(chunk_of)))
    return [p[0] for p in parts], [p[1] for p in parts]


def main():
    parser = pipeline.parser(__doc__)
    parser.add_argument('--years', type=int, nargs='+', default=None, help='Default: [run] years')
    parser.add_argument('--n-jobs', type=int, default=None, help='Parallel workers for regridding (default: [L3] n_jobs)')
    parser.add_argument('--max-profiles', type=int, default=None,
                        help='Only regrid the first N profiles of each year (for testing)')
    args = parser.parse_args()
    run = pipeline.start(args, __file__)
    cfg = run.section('L3')
    years, ref_time = run.years(), run.ref_time
    bgc_list = cfg.get('bgc_vars', ['pH'])
    n_jobs = args.n_jobs or cfg.get('n_jobs', 1)
    data_file, index_file = run.output('L3_data', 'bgc'), run.output('L3_index', 'bgc')
    l0_source = run.section('L0.bgc').get('description', '')

    data_parts, index_parts = [], []
    n_desc_total = 0
    for yr in years:
        print(f'===> {yr}: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))
        yearINDEX = open_ocean_index(run.input('L2', 'bgc', yr=yr))
        yearDF, n_desc = read_l1_year(run.input('L1', 'bgc', yr=yr), yearINDEX.profid.values,
                                      ref_time, bgc_list)
        n_desc_total += n_desc

        # Surfacemost observation of the kept cast (L2 values may come from the descending cast)
        first = yearDF.reset_index().sort_values('pressure').groupby('profid').first()
        first = first[first.pressure <= cfg.get('surface_max', 25)]
        if args.max_profiles:
            first = first.iloc[:args.max_profiles]
        yearDF = yearDF[yearDF.index.isin(first.index)]
        yearINDEX = yearINDEX.sel(profid=first.index.values)
        for var in ['pressure', 'latitude', 'longitude', 'datetime', 'CT', 'SA', 'sigma0', 'spice',
                    'temperature', 'salinity']:
            yearINDEX[var] = ('profid', first[var].values)

        print(f'  {yearINDEX.sizes["profid"]} open ocean profiles with surface data '
              f'({n_desc} descending casts dropped); regridding and adding MLD...')

        year_index_parts, year_data_parts = grid_parallel(yearDF, yearINDEX, bgc_list, n_jobs, ref_time)
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
    flags = ','.join(run.section('L1').get('use_flags', []))
    bgc_tag = ', '.join(bgc_list)
    bathy_lim, surface_max = run.section('L2').get('bathy_lim', -1000), cfg.get('surface_max', 25)
    yrs = f'{min(years)}-{max(years)}' if len(years) > 1 else str(years[0])
    source = f"Argopy, {run.section('L0.fetcher').get('mode', 'expert')} mode; {l0_source}"
    bgcDS = bgcDS.assign_attrs({
        'title': f'BGC float profiles {yrs} with valid data (QC flags {flags} incl. {bgc_tag}), masked to open '
                 f'ocean bathymetry (elevation < {bathy_lim:g} m), with valid observations above {surface_max:g} dbar. '
                 f'Interpolated (pchip) to n={len(pres_regular)} pressure levels: 0-1000 dbar every 5 dbar, '
                 '1050-1950 dbar every 50 dbar. '
                 'Includes mixed layer pressure (mlp, dbar) and depth (mld, m) from a 0.03 kg/m3 '
                 'sigma0 threshold relative to 10 dbar.',
        'pressure_levels': 'Pressure levels in dbar: ' + str(pres_regular),
        'source': source,
        **run.attrs(),
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
                 f'where elevation < {bathy_lim:g}m. Variables of pressure, CT, SA... represent surfacemost '
                 'observation within the profile.',
        'source': source,
        **run.attrs(),
        'descending_casts_dropped': n_desc_total,
        'date': datetag})

    # Drop encodings inherited from the L2 files (e.g. profid stored as <U13, which would
    # truncate 4-digit cycle numbers like 7900947cyc1029)
    bgcDS.drop_encoding().to_netcdf(data_file)
    bgcINDEX.drop_encoding().to_netcdf(index_file)
    print(f'Saved {bgcDS.sizes["profid"]} profiles to {data_file}')
    print(f'Saved index to {index_file}: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))
    run.finish(descending_casts_dropped=n_desc_total)


if __name__ == '__main__':
    main()
