"""Vectorized P1 regression features for one row per sample (profile, 5m average, or ship obs).

Same features as argoclare.preprocessing (notebooks 1.1-1.3), but looked up for all samples at
once instead of row-by-row:
    - nearest-neighbour grid indices are found with pandas get_indexer(method='nearest'),
      the same lookup xarray's .sel(method='nearest') uses
    - each gridded file is opened once per year, and read one chunk-aligned block at a time,
      so every chunk is decompressed at most once (ERA5 files are chunked 974 hours deep,
      so row-wise reads decompress a ~14 MB chunk per sample)
Input DataFrames need 'latitude', 'longitude' (-180 to 180), 'datetime', and 'SA'/'CT' for
the vapor pressure. Functions that read gridded files take samples from a single year.

Differences from argoclare.preprocessing:
    - add_atmospheric_co2() prints how many samples fall after the end of the MBL record,
      which get its last value. The MBL file now runs through decimalyr 2025.0
      (create_mbl_co2_dataset.py); the old 2014-2023 file ended at 2024.0.
    - add_sea_ice() only adds the instantaneous value unless days_before is given.
add_sea_level_pressure() uses the nearest ERA5 grid point, matching the Oct 2026 fix to
argoclare preprocessing.add_sea_level_pressure().
Gridded files are passed as paths with {yr} (from the run config, see pipeline.Run.feature_files()),
since joblib workers re-import this module. Only the files that the run's features need are
used (needed_files(), from [P1] features and the [regression] features and target).
With [P1.era5] source = "public" or "earthmover" (ERA5 on Earthmover, see ERA5_SOURCES), the ERA5
features (sea level pressure, wind speed, and SST_era5, which only comes from Earthmover) are taken
from [files] era5_points instead: the ERA5 values at each sample, looked up by
create_P1_era5_points.py (add_era5_points()).
"""

import os

import gsw
import numpy as np
import pandas as pd
import pyproj
import xarray as xr

from argoclare import oceantools as mod_ocean
from argoclare import preprocessing as mod_prep

ref_time = '2014-01-01'  # default origin of linear_time; runs use [run] ref_time

# Gridded inputs (pipeline.Run.feature_files() names) and the P1 columns that need them.
# Time variables, max_solar_rad and log_mld need no external file and are always added.
FILE_COLUMNS = {
    'slp': ['atmos_pres_Pa', 'atmos_pres_atm'],
    'mbl': ['atmos_co2_ppm', 'sst', 'sss', 'vapor_pres_atm', 'pco2_atmos', 'delta_pco2'],
    'adt': ['adt'],
    'wind': ['wind_speed'],
    'seaice': ['sea_ice'],
}
# [P1.era5] sources that read ERA5 on Earthmover: its public bucket (no account) or an Arraylake
# repo (Earthmover login). Both are looked up into [files] era5_points by create_P1_era5_points.py.
ERA5_SOURCES = ('public', 'earthmover')
# ERA5 inputs that these sources take from [files] era5_points, the ERA5 variables each needs
# there, and the P1 columns that need sst (only from Earthmover)
ERA5_VARS = {'slp': ['msl'], 'wind': ['u10', 'v10'], 'sst': ['sst']}
EARTHMOVER_COLUMNS = {'sst': ['SST_era5']}
ERA5_KEY = ['datetime', 'latitude', 'longitude']  # era5_points rows match P1 samples on these


def needed_files(features):
    """Gridded inputs needed for a list of P1 columns (features and target); all of them if
    features is None. The MBL pCO2 also needs the sea level pressure."""
    if features is None:
        return set(FILE_COLUMNS)
    names = {
        name
        for name, cols in FILE_COLUMNS.items()
        if any(f in cols or (name == 'seaice' and f.startswith('sea_ice_avg_')) for f in features)
    }
    return names | {'slp'} if 'mbl' in names else names


def earthmover_names(run, always=()):
    """ERA5 inputs (ERA5_VARS names) a run takes from Earthmover: none unless [P1.era5] source is
    one of ERA5_SOURCES; then those its P1 features need (plus `always`), sst included."""
    features = run.p1_features()
    source = run.section('P1.era5').get('source', 'file')
    if source not in ('file', *ERA5_SOURCES):
        raise SystemExit(f'[P1.era5] source = "{source}": use "file", "public" or "earthmover"')
    if source == 'file':
        if features and any(f in cols for cols in EARTHMOVER_COLUMNS.values() for f in features):
            raise SystemExit('SST_era5 only comes from Earthmover: set [P1.era5] source = "public" or "earthmover"')
        return set()
    names = needed_files(features) | set(always)
    names |= {n for n, cols in EARTHMOVER_COLUMNS.items() if features is None or any(f in cols for f in features)}
    return names & set(ERA5_VARS)


def run_feature_files(run, years, always=(), dataset=None):
    """The gridded files a run's P1 step needs (plus `always`), checked for every year. With
    [P1.era5] source = "public" or "earthmover", the ERA5 ones are replaced by [files] era5_points for `dataset`."""
    features = run.p1_features()
    names = needed_files(features) | set(always)
    era5 = earthmover_names(run, always)
    print(
        f'P1 features: {"all" if features is None else ", ".join(features)}; gridded inputs: '
        + (', '.join(n for n in FILE_COLUMNS if n in names - era5) or 'none')
        + (f'; from Earthmover: {", ".join(v for n in sorted(era5) for v in ERA5_VARS[n])}' if era5 else '')
    )
    files = run.feature_files(years, names - era5)
    if era5:
        path = run.path('era5_points', dataset)
        if not os.path.exists(path):
            raise SystemExit(
                f'Missing the ERA5 values for {dataset} ([files] era5_points): {path}\nLook them up first with '
                f'`pixi run era5-points {run.config_path} {dataset}` (`pixi run -e arraylake ...` for source = '
                f'"earthmover"), or use local files with [P1.era5] source = "file"'
            )
        files['era5_points'], files['era5_names'] = run.input('era5_points', dataset), sorted(era5)
    return files


# %% Grid sampling


def nearest_index(coord, values):
    """Index of the nearest coordinate value, as in .sel(method='nearest')."""
    return pd.Index(np.asarray(coord)).get_indexer(np.asarray(values), method='nearest')


def sample_points(da, indexers, blocks=None):
    """Values of `da` at integer indices {dim: array}, read one block of the grid at a time.

    blocks: {dim: block length}; dims left out are read whole. Defaults to the file chunks.
    """
    if blocks is None:
        chunks = da.encoding.get('chunksizes')
        blocks = dict(zip(da.dims, chunks)) if chunks else {}
    n = len(next(iter(indexers.values())))
    out = np.full(n, np.nan)
    sizes = [blocks.get(d) or da.sizes[d] for d in da.dims]
    keys = pd.DataFrame({d: indexers[d] // b for d, b in zip(da.dims, sizes)})
    for key, pos in keys.groupby(list(da.dims)).indices.items():
        starts = [k * b for k, b in zip(np.atleast_1d(key), sizes)]
        block = da.isel({d: slice(s, s + b) for d, s, b in zip(da.dims, starts, sizes)}).values
        out[pos] = block[tuple(indexers[d][pos] - s for d, s in zip(da.dims, starts))]
    return out


# %% Time


def add_time_vars(df, ref_time=ref_time):
    """year, month, day, linear_time, ydcos, ydsin, decimalyr (argoclare add_regression_time_vars)."""
    df = df.copy()
    t = pd.to_datetime(df['datetime'])
    df['datetime'] = t
    df['year'], df['month'], df['day'] = t.dt.year, t.dt.month, t.dt.day
    df['linear_time'] = (t - pd.Timestamp(ref_time)) / pd.Timedelta(days=1)
    df['ydcos'], df['ydsin'] = (np.asarray(x) for x in mod_ocean.get_ydsines(df['linear_time'].values))
    df['decimalyr'] = t.dt.year + (t.dt.dayofyear - 1) / np.where(t.dt.is_leap_year, 366, 365)
    return df


# %% Atmospheric pressure and CO2


def add_sea_level_pressure(df, yr, slp_file):
    """ERA5 mean sea level pressure at the nearest 0.25 deg grid point and hour."""
    da = xr.open_dataset(slp_file.format(yr=yr))['msl']
    df['atmos_pres_Pa'] = sample_points(
        da,
        {
            'valid_time': nearest_index(da.valid_time, df['datetime']),
            'latitude': nearest_index(da.latitude, df['latitude']),
            'longitude': nearest_index(da.longitude, df['longitude']),
        },
    )
    df['atmos_pres_atm'] = df['atmos_pres_Pa'] / 1013.25 / 100
    return df


def add_era5_points(df, points_file, names):
    """ERA5 features from [files] era5_points (create_P1_era5_points.py: Earthmover ERA5 at the
    nearest 0.25 deg grid point and hour of each sample), for the ERA5_VARS names given:
    slp -> atmos_pres_Pa, atmos_pres_atm; wind -> wind_speed; sst -> SST_era5 (deg C)."""
    need = [v for n in names for v in ERA5_VARS[n]]
    points = pd.read_parquet(points_file)
    missing = [v for v in need if v not in points.columns]
    if missing:
        raise SystemExit(f'{points_file} has no {", ".join(missing)}: rerun create_P1_era5_points.py')
    points = points[ERA5_KEY + need].drop_duplicates(ERA5_KEY)
    found = df[ERA5_KEY].merge(points, on=ERA5_KEY, how='left', indicator=True)
    n_missing = int((found['_merge'] == 'left_only').sum())
    if n_missing:
        raise SystemExit(f'{n_missing} samples are not in {points_file}: rerun create_P1_era5_points.py')
    if 'slp' in names:
        df['atmos_pres_Pa'] = found['msl'].values.astype(float)
        df['atmos_pres_atm'] = df['atmos_pres_Pa'] / 1013.25 / 100
    if 'wind' in names:
        df['wind_speed'] = np.sqrt(found['u10'].values ** 2 + found['v10'].values ** 2)
    if 'sst' in names:
        df['SST_era5'] = found['sst'].values.astype(float) - 273.15
    return df


def add_atmospheric_co2(df, mbl_file):
    """MBL xCO2 (ppm), nearest in time and linear in sin(lat); converted to pCO2 (uatm) with
    the Zeebe and Wolf-Gladrow (2001) vapor pressure. Needs atmos_pres_atm."""
    mbl = xr.open_dataset(mbl_file)
    grid_sinlat = mbl.sin_lat.values
    df['sin_lat'] = np.sin(df['latitude'] * np.pi / 180)
    it = nearest_index(mbl.decimalyr, df['decimalyr'])
    n_after = int((df['decimalyr'] > mbl.decimalyr.values[-1]).sum())
    if n_after:
        print(f'  {n_after} samples after the MBL record ends ({mbl.decimalyr.values[-1]}); using its last value')
    ppm = np.full(len(df), np.nan)
    mean = mbl['mean'].values
    for i in np.unique(it):
        pos = np.flatnonzero(it == i)
        ppm[pos] = np.interp(df['sin_lat'].values[pos], grid_sinlat, mean[i], left=np.nan, right=np.nan)
    df['atmos_co2_ppm'] = ppm

    zeros = np.zeros(len(df))
    df['sst'] = gsw.t_from_CT(df['SA'], df['CT'], zeros)
    df['sss'] = gsw.SP_from_SA(df['SA'], zeros, df['longitude'], df['latitude'])
    sst_kelvin = df['sst'] + 273.15
    df['vapor_pres_atm'] = np.exp(
        24.4542 - 67.4509 * (100 / sst_kelvin) - 4.8489 * np.log(sst_kelvin / 100) - 0.000544 * df['sss']
    )
    df['pco2_atmos'] = df['atmos_co2_ppm'] * (df['atmos_pres_atm'] - df['vapor_pres_atm'])
    return df


# %% Ocean surface


def add_adt(df, yr, adt_file, time_block=16):
    """CMEMS DUACS L4 absolute dynamic topography (m), nearest 0.125 deg grid point and day."""
    da = xr.open_dataset(adt_file.format(yr=yr))['adt']
    df['adt'] = sample_points(
        da,
        {
            'time': nearest_index(da.time, df['datetime']),
            'latitude': nearest_index(da.latitude, df['latitude']),
            'longitude': nearest_index(da.longitude, df['longitude']),
        },
        blocks={'time': time_block},
    )  # file is not chunked; read 16 days at a time
    return df


def add_wind_speed(df, yr, wind_file):
    """ERA5 10 m wind speed sqrt(u10^2 + v10^2) at the nearest hour, after rounding to 0.25 deg."""
    ds = xr.open_dataset(wind_file.format(yr=yr))
    indexers = {
        'valid_time': nearest_index(ds.valid_time, df['datetime']),
        'latitude': nearest_index(ds.latitude, np.round(df['latitude'].values * 4) / 4),
        'longitude': nearest_index(ds.longitude, np.round(df['longitude'].values * 4) / 4),
    }
    u10 = sample_points(ds['u10'], indexers).astype(np.float32)
    v10 = sample_points(ds['v10'], indexers).astype(np.float32)
    df['wind_speed'] = np.sqrt(u10**2 + v10**2)
    return df


def add_sea_ice(df, yr, seaice_file, days_before=None):
    """NSIDC CDR v6 sea ice concentration (25 km polar stereographic), nearest cell and day.
    Optionally also 'sea_ice_avg_<days_before>d': the mean over 1 to days_before-1 days before
    (as in argoclare get_row_avg_seaice), with each day taken as nearest in this year's file."""
    da = xr.open_dataset(seaice_file.format(yr=yr))['cdr_seaice_conc'].load()  # ~300 MB
    transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:3412', always_xy=True)
    x_m, y_m = transformer.transform(df['longitude'].values, df['latitude'].values)
    space = {'x': nearest_index(da.x, x_m), 'y': nearest_index(da.y, y_m)}

    def at(times):
        return sample_points(da, {'time': nearest_index(da.time, times), **space}, blocks={})

    df['sea_ice'] = at(df['datetime'])
    if days_before:
        past = [at(df['datetime'] - pd.Timedelta(days=k)) for k in range(1, days_before)]
        df[f'sea_ice_avg_{days_before}d'] = np.mean(past, axis=0)
    return df


def add_solar_radiation(df):
    """Daily maximum clear-sky irradiance (W/m2) within +-12 h (solarpy, ~1 ms per sample)."""
    df['max_solar_rad'] = [
        mod_prep.get_max_solar_radiation({'datetime': t, 'latitude': lat})
        for t, lat in zip(df['datetime'], df['latitude'])
    ]
    return df


def save_netcdf(df, dim, path, attrs, coords=('datetime', 'latitude', 'longitude')):
    """Save one row per sample as a Dataset along `dim` (a column of df, or a new 0..N-1 index).
    Text columns are written as strings, with missing values as ''."""
    df = df.copy()
    for c in df.columns[df.dtypes == object]:  # noqa: E721
        df[c] = df[c].where(df[c].notna(), '').astype(str)
    df.columns = [c.replace(' ', '_') for c in df.columns]  # e.g. 'Potential Density_kg_m3'
    df = df.set_index(dim) if dim in df.columns else df.rename_axis(dim)
    ds = xr.Dataset.from_dataframe(df).set_coords([c for c in coords if c in df.columns])
    ds.assign_attrs(attrs).to_netcdf(path)
    return ds


def add_all_features(df, yr, files, ref_time=ref_time, seaice_days_before=None):
    """P1 features for samples from one year. files: paths for any of {slp, mbl, adt, wind, seaice},
    or era5_points with era5_names (run_feature_files()); features from a file that is not given are left out."""
    df = add_time_vars(df, ref_time)
    if 'era5_points' in files:  # [P1.era5] source = "public" or "earthmover"
        df = add_era5_points(df, files['era5_points'], files['era5_names'])
    if 'slp' in files:
        df = add_sea_level_pressure(df, yr, files['slp'])
    if 'mbl' in files:
        df = add_atmospheric_co2(df, files['mbl'])
    if 'adt' in files:
        df = add_adt(df, yr, files['adt'])
    if 'wind' in files:
        df = add_wind_speed(df, yr, files['wind'])
    if 'seaice' in files:
        df = add_sea_ice(df, yr, files['seaice'], days_before=seaice_days_before)
    df = add_solar_radiation(df)
    if 'mld' in df.columns:
        df['log_mld'] = np.log(df['mld'])
    return df
