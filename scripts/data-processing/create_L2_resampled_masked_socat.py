"""Create the L2 SOCAT dataset: random 23.5 h resample of each cruise, masked to open ocean.

Same steps as 0.1_socat2024_processing.ipynb (sections II-III, "Random daily resample"):
    1. Read the SOCAT Southern Oceans synthesis file (fco2rec WOCE flag 2 only), keep
       yr >= 2014 and latitude < -35, and build datetime and yearday (days since 2014-01-01)
    2. Split each cruise (expocode) into sections (subID) wherever consecutive observations are
       more than --gap-hours (72) apart; socatid = <expocode>_id<subID>
    3. Within each section, bin the observations into --freq (23.5h) periods, as in pandas
       .resample() (bins start at midnight of the section's first day), and keep one random
       observation per non-empty bin. The pick uses --seed, so a rerun reproduces the file
       (the notebook version was unseeded).
    4. Look up the nearest GEBCO 2024 elevation and keep samples with elevation < -1000 m,
       with argoclare's get_bathymetry_DS()
Writes one netcdf along 'sample' (datetime is a coordinate) to x_localdata/socat/.

Usage:
    python create_L2_resampled_masked_socat.py
    python create_L2_resampled_masked_socat.py --max-cruises 20 --outpath /tmp/test/
"""
import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

import argoclare.oceantools as mod_ocean

default_raw = ('/Users/sangminsong/Library/CloudStorage/OneDrive-UW/Code/CRUSOE/MATLAB-SOCAT/'
               'socatv2025_SouthernOceans_DATA.csv')
default_bathy_path = '/Volumes/crusoe-repo/data/etopo/gebco_2024_sub_ice_topo/GEBCO_2024_sub_ice_topo.nc'
default_outpath = '/Users/sangminsong/Documents/PROJECTS-PYTHON/CRUSOE-SOCarbon/x_localdata/socat/'
ref_time = '2014-01-01'


def read_socat(path, max_cruises=None):
    """SOCAT observations from 2014 on, south of 35S, with datetime and yearday."""
    df = pd.read_csv(path, low_memory=False)
    df = df[(df.yr >= 2014) & (df.latitude < -35)].copy()
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
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raw', default=default_raw, help='SOCAT synthesis csv (exported with the SOCAT MATLAB code)')
    parser.add_argument('--freq', default='23.5h', help='Resampling period')
    parser.add_argument('--gap-hours', type=float, default=72, help='Gap that starts a new cruise section')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for the resample')
    parser.add_argument('--bathy-path', default=default_bathy_path, help='GEBCO elevation netcdf')
    parser.add_argument('--outpath', default=default_outpath, help='Folder to save the L2 netcdf in')
    parser.add_argument('--max-cruises', type=int, default=None, help='Only use the first N cruises (for testing)')
    args = parser.parse_args()

    print('Reading ' + args.raw + ': ', datetime.now().strftime('%Y%m%d %H:%M:%S'))
    socat_df = read_socat(args.raw, args.max_cruises)
    resampled = random_resample(socat_df, freq=args.freq, gap_hours=args.gap_hours, seed=args.seed)
    print(f'{len(socat_df)} observations from {socat_df.expocode.nunique()} cruises; '
          f'{len(resampled)} after resampling to one per {args.freq}')

    attrs = {'title': f'SOCATv2025 Southern Ocean data from 2014, resampled to a random observation per '
                      f'{args.freq} period. Masked to open ocean (GEBCO elevation < -1000 m) and south of 35S.',
             'source': 'DOI: 10.25921/648f-fv35',
             'site': 'https://socat.info/index.php/version-2025/',
             'notes': f'Resample frequency {args.freq}, gap time {args.gap_hours:g}h, random seed {args.seed}',
             'raw_file': os.path.basename(args.raw),
             'script': os.path.basename(__file__),
             'date_processed': datetime.now().strftime('%Y-%m-%d')}
    socatINDEX = mod_ocean.get_bathymetry_DS(resampled.rename_axis('sample'),
                                             bathy_nc=xr.open_dataset(args.bathy_path).elevation,
                                             dropvars=[], bathy_lim=-1000, nc_attrs=attrs)
    socatINDEX = socatINDEX.set_coords(['datetime', 'latitude', 'longitude'])
    socatINDEX['linear_time'] = socatINDEX['yearday'] # as in data_loader.import_socat_L2()
    socatINDEX = socatINDEX.assign_coords(sample=np.arange(socatINDEX.sizes['sample']))

    os.makedirs(args.outpath, exist_ok=True)
    tag = args.freq.replace('.', 'p')
    savefile = os.path.join(args.outpath, f'SOCATv2025_SO_randomresample{tag}_seed{args.seed}_open_ocean_INDEX_'
                                          f'acc{datetime.now().strftime("%Y%m%d")}.nc')
    socatINDEX.to_netcdf(savefile)
    print(f'Kept {socatINDEX.sizes["sample"]} of {len(resampled)} samples in open ocean; saved to {savefile}: ',
          datetime.now().strftime('%Y%m%d %H:%M:%S'))


if __name__ == '__main__':
    main()
