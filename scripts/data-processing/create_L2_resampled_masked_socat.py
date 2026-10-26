"""Create the L2 SOCAT dataset: random 23.5 h resample of each cruise, masked to open ocean.

Same steps as 0.1_socat2024_processing.ipynb (sections II-III, "Random daily resample"):
    1. Read [files] socat_raw, the SOCAT Southern Oceans synthesis file (fco2rec WOCE flag 2
       only), keep the [socat] years (else [run] years) and the [region] box, and build datetime and yearday (days
       since [run] ref_time)
    2. Split each cruise (expocode) into sections (subID) wherever consecutive observations are
       more than [socat] gap_hours (72) apart; socatid = <expocode>_id<subID>
    3. Within each section, bin the observations into [socat] freq (23.5h) periods, as in pandas
       .resample() (bins start at midnight of the section's first day), and keep one random
       observation per non-empty bin. The pick uses [socat] seed, so a rerun reproduces the
       file (the notebook version was unseeded).
    4. Look up the nearest elevation in [files] bathy and keep samples with elevation <
       [L2] bathy_lim, with argoclare's get_bathymetry_DS()
Writes [files] socat_L2, one netcdf along 'sample' (datetime is a coordinate).

Usage:
    python create_L2_resampled_masked_socat.py --config ../../configs/example.toml
    python create_L2_resampled_masked_socat.py --config ../../configs/example.toml --max-cruises 20 --out-dir /tmp/test/
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

import argoclare.oceantools as mod_ocean
from argoclare import pipeline


def read_socat(path, years, in_box, ref_time, max_cruises=None):
    """SOCAT observations in the run years and region box, with datetime and yearday."""
    df = pd.read_csv(path, low_memory=False)
    df = df[df.yr.isin(years) & in_box(df.longitude, df.latitude)].copy()
    df['expocode'] = df['expocode'].astype(str)
    if max_cruises:
        df = df[df.expocode.isin(df.expocode.unique()[:max_cruises])]
    df['datetime'] = pd.to_datetime(df[['yr', 'mon', 'day', 'hh', 'mm', 'ss']].rename(columns={
        'yr': 'year', 'mon': 'month', 'hh': 'hour', 'mm': 'minute', 'ss': 'second'}))
    df['yearday'] = (df['datetime'] - pd.Timestamp(ref_time)) / pd.Timedelta(days=1)
    return df


def random_resample(df, freq='23.5h', gap_hours=72, seed=0):
    """One random observation per `freq` bin within each continuous section of each cruise."""
    df = df.sort_values(['expocode', 'datetime'], kind='stable').reset_index(drop=True)
    df['hr_diff'] = df.groupby('expocode')['datetime'].diff().dt.total_seconds() / 3600
    df['temp_mask'] = (df['hr_diff'] > gap_hours).astype(int)
    df['subID'] = df.groupby('expocode')['temp_mask'].cumsum()
    df['socatid'] = df['expocode'] + '_id' + df['subID'].astype(str)

    # pandas resample bins: origin at midnight of the section's first day, closed on the left
    origin = df.groupby('socatid')['datetime'].transform('min').dt.floor('D')
    df['resample_bin'] = ((df['datetime'] - origin) // pd.Timedelta(freq)).astype(int)
    resampled = df.groupby(['socatid', 'resample_bin']).sample(n=1, random_state=seed)
    return resampled.drop(columns='resample_bin').sort_values(['expocode', 'datetime']).reset_index(drop=True)


def main():
    parser = pipeline.parser(__doc__)
    parser.add_argument('--years', type=int, nargs='+', default=None, help='Default: [socat] or [run] years')
    parser.add_argument('--max-cruises', type=int, default=None, help='Only use the first N cruises (for testing)')
    args = parser.parse_args()
    run = pipeline.start(args, __file__)
    cfg = run.section('socat')
    freq, gap_hours, seed = cfg.get('freq', '23.5h'), cfg.get('gap_hours', 72), cfg.get('seed', 0)
    bathy_lim = run.section('L2').get('bathy_lim', -1000)
    years = run.years('socat')
    raw_file, bathy_file = run.input('socat_raw'), run.input('bathy')
    savefile = run.output('socat_L2')

    print('Reading ' + raw_file + ': ', datetime.now().strftime('%Y%m%d %H:%M:%S'))
    socat_df = read_socat(raw_file, years, run.in_box, run.ref_time, args.max_cruises)
    resampled = random_resample(socat_df, freq=freq, gap_hours=gap_hours, seed=seed)
    print(f'{len(socat_df)} observations from {socat_df.expocode.nunique()} cruises; '
          f'{len(resampled)} after resampling to one per {freq}')

    yrs = f'{min(years)}-{max(years)}' if len(years) > 1 else str(years[0])
    attrs = {'title': f'{cfg.get("version", "SOCAT")} data {yrs}, resampled to a random observation per '
                      f'{freq} period. Masked to open ocean (elevation < {bathy_lim:g} m) and to the region '
                      f'box {run.box} (lon_min, lon_max, lat_min, lat_max).',
             'source': f'DOI: {cfg.get("doi", "")}',
             'site': cfg.get('site', ''),
             'notes': f'Resample frequency {freq}, gap time {gap_hours:g}h, random seed {seed}',
             'raw_file': os.path.basename(raw_file),
             'bathy_file': os.path.basename(bathy_file),
             **run.attrs(),
             'date_processed': datetime.now().strftime('%Y-%m-%d')}
    socatINDEX = mod_ocean.get_bathymetry_DS(resampled.rename_axis('sample'),
                                             bathy_nc=xr.open_dataset(bathy_file).elevation,
                                             dropvars=[], bathy_lim=bathy_lim, nc_attrs=attrs)
    socatINDEX = socatINDEX.set_coords(['datetime', 'latitude', 'longitude'])
    socatINDEX['linear_time'] = socatINDEX['yearday'] # as in data_loader.import_socat_L2()
    socatINDEX = socatINDEX.assign_coords(sample=np.arange(socatINDEX.sizes['sample']))

    socatINDEX.to_netcdf(savefile)
    print(f'Kept {socatINDEX.sizes["sample"]} of {len(resampled)} samples in open ocean; saved to {savefile}: ',
          datetime.now().strftime('%Y%m%d %H:%M:%S'))
    run.finish()


if __name__ == '__main__':
    main()
