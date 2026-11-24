"""Look up ERA5 from Earthmover at each P1 sample, for [P1.era5] source = "public" or "earthmover".

Instead of the yearly ERA5 files ([files] era5_slp, era5_wind), P1 can take its ERA5 features
from Earthmover's Icechunk copy of ERA5, read from either
    "public"       its public S3 bucket, anonymously (no account; the icechunk package)
    "earthmover"   an Arraylake repo (Earthmover login and the arraylake package)
The two hold the same repository, so a snapshot ID gives the same values from either. This script reads the samples P1 will use and writes
their ERA5 values to [files] era5_points (one parquet per dataset, in the run's P1/ folder), which
create_P1_*_features.py then read (p1_features.add_era5_points()):
    msl (Pa)         -> atmos_pres_Pa, atmos_pres_atm
    u10, v10 (m/s)   -> wind_speed
    sst (K)          -> SST_era5 (deg C)
Only the variables the run's P1 features need are looked up (p1_features.earthmover_names()).
Each sample takes the nearest 0.25 deg grid point and the nearest hour, as with the local files
(sample longitudes are converted to 0-360 for the lookup, wrapping at 360; the file keeps -180 to
180). A sample whose nearest cell has no SST (land) takes the nearest cell with one at the same
hour within [P1.era5] sst_ocean_search cells (sst_km: distance to it; NaN if none).

Samples, matched to P1 rows on datetime, latitude and longitude:
    core, bgc   [files] L3_index profiles in [P1.<dataset>] (or [run]) years
    socat       [files] socat_L2 samples in the [region] box and [P1.socat] (or [run]) years
Chunks are read whole (the temporal group's chunks are 8736 h x 12 x 12 cells), several at a time.

Rerunning reuses the file: only samples or variables not in it yet are looked up, if its group
and snapshot match the config (otherwise pass --overwrite to look everything up again).
source = "earthmover" needs the pixi environment `arraylake` and an Earthmover login
(`pixi run -e arraylake arraylake auth login`).

Config:
    [P1.era5]
    source = "public"                             # "file" (default): the local yearly files
    bucket = "earthmover-icechunk-era5"           # source = "public": S3 bucket, prefix and region
    prefix = "icechunkV2"
    region = "us-east-1"
    # repo = "<org>/<repo>"                       # source = "earthmover": Arraylake repo
    group = "single/temporal"
    snapshot = "VKGPX45BPEQJMVZZXZEG"             # pinned snapshot ID of the repo
    sst_ocean_search = 3                          # cells

Usage:
    pixi run python create_P1_era5_points.py --config ../../configs/example_drake.toml --datasets core bgc
    pixi run -e arraylake python create_P1_era5_points.py ...    # source = "earthmover"
"""

import functools
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import p1_features
import pandas as pd
import xarray as xr

import argoclare.oceantools as mod_ocean
from argoclare import pipeline

DIMS = ['valid_time', 'latitude', 'longitude']
CELL_COLS = ['era5_time', 'era5_lat', 'era5_lon']
# Config keys each source needs; group and snapshot identify the values (SAME_DATA)
SOURCE_ATTRS = {
    'public': ['bucket', 'prefix', 'region', 'group', 'snapshot'],
    'earthmover': ['repo', 'group', 'snapshot'],
}
SAME_DATA = ['group', 'snapshot']
ERA5_ALL = [v for vs in p1_features.ERA5_VARS.values() for v in vs]


def float_samples(run, dataset):
    """L3 profiles in the dataset's P1 years, as create_P1_float_features.py reads them."""
    df = xr.open_dataset(run.input('L3_index', dataset)).to_dataframe().reset_index()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = mod_ocean.set_longitude_range(df, end_type='180')
    return df[df.datetime.dt.year.isin(run.years(f'P1.{dataset}'))]


def socat_samples(run):
    """SOCAT L2 samples in the box and P1 years, as create_P1_socat_features.py reads them."""
    df = xr.open_dataset(run.input('socat_L2')).to_dataframe().reset_index()
    df['datetime'] = pd.to_datetime(df['datetime']).astype('datetime64[ns]')
    df = df[run.in_box(df.longitude, df.latitude) & df.datetime.dt.year.isin(run.years('P1.socat'))]
    return mod_ocean.set_longitude_range(df, end_type='180')


def open_era5(cfg):
    if cfg['source'] == 'public':
        import icechunk

        storage = icechunk.s3_storage(bucket=cfg['bucket'], prefix=cfg['prefix'], region=cfg['region'], anonymous=True)
        session = icechunk.Repository.open(storage).readonly_session(snapshot_id=cfg['snapshot'])
    else:
        try:
            from arraylake import Client  # only in the arraylake environment
        except ModuleNotFoundError:
            raise SystemExit('source = "earthmover" needs the arraylake package: run with `pixi run -e arraylake`')
        session = Client().get_repo(cfg['repo']).readonly_session(snapshot_id=cfg['snapshot'])
    return xr.open_zarr(session.store, group=cfg['group'], chunks=None, consolidated=False)


def grid_index(ds, df):
    """Nearest hour and 0.25 deg grid point of each sample: {dim: integer index}."""
    lon = ds.longitude.values
    wrapped = np.append(lon, lon[0] + 360)  # past the last longitude, the nearest may be the first
    return {
        'valid_time': p1_features.nearest_index(ds.valid_time, df['datetime']),
        'latitude': p1_features.nearest_index(ds.latitude, df['latitude']),
        'longitude': p1_features.nearest_index(wrapped, df['longitude'].values % 360) % len(lon),
    }


def read_points(da, idx, n_threads):
    """Values of `da` at integer indices {dim: array}, one storage chunk at a time, n_threads at once."""
    sizes = dict(zip(da.dims, da.encoding['chunks']))
    keys = pd.DataFrame({d: idx[d] // sizes[d] for d in da.dims})
    out = np.full(len(keys), np.nan, dtype=np.float32)

    def read(item):
        key, pos = item
        starts = {d: k * sizes[d] for d, k in zip(da.dims, key)}
        block = da.isel({d: slice(starts[d], starts[d] + sizes[d]) for d in da.dims}).values
        out[pos] = block[tuple(idx[d][pos] - starts[d] for d in da.dims)]

    with ThreadPoolExecutor(n_threads) as pool:
        list(pool.map(read, keys.groupby(list(da.dims)).indices.items()))
    return out


def km_between(lat1, lon1, lat2, lon2):
    """Distance (km), equirectangular (fine within a few cells)."""
    dlon = (lon2 - lon1 + 180) % 360 - 180
    x = np.deg2rad(dlon) * np.cos(np.deg2rad((lat1 + lat2) / 2))
    return 6371 * np.hypot(x, np.deg2rad(lat2 - lat1))


def sst_ocean_fallback(ds, df, idx, sst, search, n_threads):
    """For samples whose nearest cell has no SST, the nearest cell with one at the same hour
    within `search` cells. Returns sst and sst_km (0 for the own cell, NaN if none found)."""
    km = np.where(np.isnan(sst), np.nan, 0.0)
    gap = np.flatnonzero(np.isnan(sst))
    if not len(gap) or not search:
        return sst, km
    di, dj = (a.ravel() for a in np.meshgrid(np.arange(-search, search + 1), np.arange(-search, search + 1)))
    n_lat, n_lon = ds.sizes['latitude'], ds.sizes['longitude']
    near = {  # (gap sample, offset) pairs, flattened
        'valid_time': np.repeat(idx['valid_time'][gap], len(di)),
        'latitude': np.clip(idx['latitude'][gap][:, None] + di, 0, n_lat - 1).ravel(),
        'longitude': ((idx['longitude'][gap][:, None] + dj) % n_lon).ravel(),
    }
    values = read_points(ds['sst'], near, n_threads).reshape(len(gap), -1)
    dist = km_between(
        df['latitude'].values[gap][:, None],
        df['longitude'].values[gap][:, None],
        ds.latitude.values[near['latitude']].reshape(len(gap), -1),
        ds.longitude.values[near['longitude']].reshape(len(gap), -1),
    )
    dist = np.where(np.isnan(values), np.inf, dist)
    best = dist.argmin(axis=1)
    ok = np.isfinite(dist[np.arange(len(gap)), best])
    sst, km = sst.copy(), km.copy()
    sst[gap[ok]] = values[np.flatnonzero(ok), best[ok]]
    km[gap[ok]] = dist[np.flatnonzero(ok), best[ok]]
    print(f'  sst: {len(gap)} samples on a cell without SST; {int(ok.sum())} filled from within {search} cells')
    return sst, km


def lookup(ds, df, variables, cfg, n_threads):
    """The ERA5 cell and the values of `variables` for each sample of df."""
    idx = grid_index(ds, df)
    out = df[p1_features.ERA5_KEY].reset_index(drop=True).copy()
    out['era5_time'] = ds.valid_time.values[idx['valid_time']]
    out['era5_lat'] = ds.latitude.values[idx['latitude']]
    out['era5_lon'] = ds.longitude.values[idx['longitude']]
    for v in variables:
        t0 = datetime.now()
        out[v] = read_points(ds[v], idx, n_threads)
        if v == 'sst':
            out['sst'], out['sst_km'] = sst_ocean_fallback(
                ds, df, idx, out['sst'].values, cfg.get('sst_ocean_search', 3), n_threads
            )
        print(f'  {v}: {len(out)} samples in {(datetime.now() - t0).total_seconds():.0f} s')
    return out


def update_points(run, era5, cfg, dataset, samples, variables, path, n_threads):
    """Look up what [files] era5_points for the dataset is missing, and save it. era5() opens the
    ERA5 source, only once something needs looking up."""
    source = {'source': cfg['source'], **{k: cfg[k] for k in SOURCE_ATTRS[cfg['source']]}}
    samples = samples[p1_features.ERA5_KEY].drop_duplicates().reset_index(drop=True)
    old = None
    if os.path.exists(path) and not run.overwrite:
        old = pd.read_parquet(path)
        differs = {k: (old.attrs.get(k), source[k]) for k in SAME_DATA if old.attrs.get(k) != source[k]}
        if differs:
            raise SystemExit(f'{path} was made from another ERA5 source {differs}; pass --overwrite to redo it')
    if old is None:
        new_rows, new_vars = samples, []
    else:
        known = samples.merge(old[p1_features.ERA5_KEY], on=p1_features.ERA5_KEY, how='left', indicator=True)
        new_rows = samples[(known['_merge'] == 'left_only').values]
        new_vars = [v for v in variables if v not in old.columns]
    if not len(new_rows) and not new_vars:
        print(f'{dataset}: all {len(samples)} samples already in {path}')
        return False
    print(f'{dataset}: looking up {len(new_rows)} new samples' + (f', and {new_vars} for the rest' if new_vars else ''))
    keep_vars = list(dict.fromkeys(variables + [v for v in ERA5_ALL if old is not None and v in old.columns]))
    parts = []
    if old is not None:
        if new_vars:
            extra = lookup(era5(), old, new_vars, cfg, n_threads).drop(columns=p1_features.ERA5_KEY + CELL_COLS)
            old = pd.concat([old.reset_index(drop=True), extra], axis=1)
        parts.append(old)
    if len(new_rows):
        parts.append(lookup(era5(), new_rows, keep_vars, cfg, n_threads))
    points = pd.concat(parts, ignore_index=True).sort_values(p1_features.ERA5_KEY).reset_index(drop=True)
    points.attrs = {
        **source,
        'title': 'ERA5 at the nearest 0.25 deg grid point and hour of each P1 sample (create_P1_era5_points.py)',
        'variables': ', '.join(v for v in ERA5_ALL if v in points.columns),
        'sst_ocean_search': cfg.get('sst_ocean_search', 3),
        'date': datetime.now().strftime('%Y%m%d'),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    points.to_parquet(path, index=False)
    run.outputs.append(path)
    print(f'Saved {len(points)} samples to {path}')
    return True


def main():
    parser = pipeline.add_skip_arg(
        pipeline.add_datasets_arg(pipeline.parser(__doc__), choices=('core', 'bgc', 'socat'))
    )
    parser.add_argument('--n-threads', type=int, default=16, help='Chunks read at once (default 16)')
    args = parser.parse_args()
    run = pipeline.start(args, __file__)
    cfg = run.section('P1.era5')
    datasets = pipeline.datasets(args)
    if cfg.get('source', 'file') not in p1_features.ERA5_SOURCES:
        print('[P1.era5] source is not "public" or "earthmover": P1 uses the local ERA5 files')
        return run.nothing_to_do()
    missing = [k for k in SOURCE_ATTRS[cfg['source']] if k not in cfg]
    if missing:
        raise SystemExit(f'[P1.era5] source = "{cfg["source"]}" needs {", ".join(missing)}')

    # Every input checked before anything is read from Earthmover
    jobs = []
    for dataset in datasets:
        names = p1_features.earthmover_names(run, always={'slp'} if dataset == 'socat' else ())
        if not names:
            print(f'{dataset}: no ERA5 features in this run')
            continue
        variables = [v for n in sorted(names) for v in p1_features.ERA5_VARS[n]]
        samples = socat_samples(run) if dataset == 'socat' else float_samples(run, dataset)
        jobs.append((dataset, samples, variables, run.output_path('era5_points', dataset)))
    if not jobs:
        return run.nothing_to_do()

    # Opened only if a sample or variable is missing, so a complete era5_points needs no
    # Earthmover access (e.g. source = "earthmover" outside the arraylake environment)
    @functools.cache
    def era5():
        where = cfg['repo'] if cfg['source'] == 'earthmover' else f's3://{cfg["bucket"]}/{cfg["prefix"]}'
        print(f'Opening {where} {cfg["group"]} at snapshot {cfg["snapshot"]}: ', datetime.now().strftime('%H:%M:%S'))
        return open_era5(cfg)

    for dataset, samples, variables, path in jobs:
        update_points(run, era5, cfg, dataset, samples, variables, path, args.n_threads)
    run.finish()


if __name__ == '__main__':
    main()
