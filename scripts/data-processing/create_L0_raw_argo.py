"""Download raw (L0) core and BGC Argo profiles with Argopy, one file per year.

Archive of the import steps in 0.2_argopy_import.ipynb (sections 3-4):
    core: local copy of the GDAC (January 2025 snapshot), ds='phy'
          -> <outpath>/core/202501snap_yr<yr>_core_profiles_<datetag>.nc (+ _index_<datetag>.csv)
          files in use: datetag 20250207 (~1 hr per year)
    bgc:  Ifremer ERDDAP, ds='bgc'
          -> <outpath>/bgc/ERDDAP_yr<yr>_bgc_profiles_<datetag>.nc (+ _index_<datetag>.csv)
          files in use: datetag 20250219 (~40 min per year)
Both use expert mode (all data and QC flags; QC is applied in create_L1_QC_argo.py) over
BOX = [-180, 180, -90, -35, 0, 2000 dbar, <yr>-01-01, <yr>-12-31], and are converted from
points to profiles (N_PROF, N_LEVELS) before saving.

Note: ERDDAP is updated daily, so a BGC rerun will not exactly reproduce the 20250219 files.
To use the local GDAC, the index .txt files had to be copied out of the "dac" folder into
the snapshot root (202501-ArgoData/); otherwise the fetcher raises an error.

Cite: Argo (2025). Argo float data and metadata from Global Data Assembly Centre (Argo GDAC).
SEANOE. https://doi.org/10.17882/42182

Usage:
    python create_L0_raw_argo.py --dataset core --years 2024
    python create_L0_raw_argo.py --dataset bgc                                  # 2014-2024
"""
import argparse
import os
from datetime import datetime

import argopy
from argopy import DataFetcher

default_outpath = '/Volumes/crusoe-repo/data/'
default_gdac = '/Volumes/ocean-repo/202501-ArgoData' # January 2025 GDAC snapshot

fetcher_kwargs = dict(mode='expert', params='all', parallel=True, progress=True,
                      chunks_maxsize={'time': 30})


def fetch_year(yr, ds):
    """Fetch one year of profiles in the Southern Ocean box; returns (profiles Dataset, index DataFrame)."""
    # Format: [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max, datim_min, datim_max]
    BOX = [-180, 180, -90, -35, 0, 2000, f'{yr}-01-01', f'{yr}-12-31']
    fetcher = DataFetcher(ds=ds, **fetcher_kwargs).region(BOX).load()

    # Point observations -> profiles; simplify attributes so they can be written to netcdf
    profiles = fetcher.data.argo.point2profile()
    attrs = {'raw_attrs': '', 'Fetched_uri': ''}
    if ds == 'bgc':
        attrs['Fetched_constraints'] = str(profiles.Fetched_constraints).replace('/', '_')
    return profiles.assign_attrs(attrs), fetcher.index


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=['core', 'bgc'], required=True)
    parser.add_argument('--years', type=int, nargs='+', default=list(range(2014, 2025)))
    parser.add_argument('--outpath', default=default_outpath,
                        help='Root folder; files are saved in <outpath>/core/ or <outpath>/bgc/')
    parser.add_argument('--gdac', default=default_gdac, help='Local GDAC snapshot (core only)')
    parser.add_argument('--cachedir', default='cache_bgc', help='Argopy cache folder')
    args = parser.parse_args()

    # Fresh start
    argopy.reset_options()
    argopy.clear_cache()
    argopy.set_options(cachedir=args.cachedir)
    if args.dataset == 'core':
        argopy.set_options(src='gdac', gdac=args.gdac)
        ds, prefix = 'phy', '202501snap_yr'
    else:
        argopy.set_options(src='erddap') # Ifremer database, updated daily
        ds, prefix = 'bgc', 'ERDDAP_yr'

    savepath = os.path.join(args.outpath, args.dataset)
    os.makedirs(savepath, exist_ok=True)
    datetag = datetime.now().strftime('%Y%m%d')

    for yr in args.years:
        print(f'===> Fetching {yr} {args.dataset}: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))
        profiles, index = fetch_year(yr, ds)

        savetag = f'{prefix}{yr}_{args.dataset}'
        profiles_file = os.path.join(savepath, f'{savetag}_profiles_{datetag}.nc')
        profiles.to_netcdf(profiles_file)
        index.to_csv(os.path.join(savepath, f'{savetag}_index_{datetag}.csv'))
        print(f'Saved {profiles.sizes["N_PROF"]} profiles to {profiles_file}')


if __name__ == '__main__':
    main()
