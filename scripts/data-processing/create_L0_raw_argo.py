"""Download raw (L0) core and BGC Argo profiles with Argopy, one file per year.

Archive of the import steps in 0.2_argopy_import.ipynb (sections 3-4). The source is set per
dataset in the config ([L0.core] / [L0.bgc] source):
    gdac:   a local copy of the GDAC ([files] gdac)
    erddap: Ifremer ERDDAP (updated daily, so a rerun will not exactly reproduce older files)
core uses ds='phy' and bgc ds='bgc'. Both use the [L0.fetcher] options (expert mode: all data
and QC flags; QC is applied in create_L1_QC_argo.py) over the [region] box and pressure range,
<yr>-01-01 to <yr>-12-31, and are converted from points to profiles (N_PROF, N_LEVELS).
Writes [files] L0_profiles (netcdf) and L0_index (csv) for each year.

CRUSOE files (configs/southernOcean_CRUSOE.toml):
    core: GDAC January 2025 snapshot, accessed 20250207 (~1 hr per year)
    bgc:  ERDDAP, accessed 20250219 (~40 min per year)
To use the local GDAC, the index .txt files had to be copied out of the "dac" folder into
the snapshot root (202501-ArgoData/); otherwise the fetcher raises an error.

Cite: Argo (2025). Argo float data and metadata from Global Data Assembly Centre (Argo GDAC).
SEANOE. https://doi.org/10.17882/42182

Usage:
    python create_L0_raw_argo.py --config ../../configs/example.toml --dataset core
    python create_L0_raw_argo.py --config ../../configs/example.toml --dataset bgc --years 2020
"""
import sys
from datetime import datetime

import argopy
from argopy import DataFetcher

from argoclare import pipeline


def fetch_year(yr, ds, box, pres, fetcher_kwargs):
    """Fetch one year of profiles in the region box; returns (profiles Dataset, index DataFrame)."""
    # Format: [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max, datim_min, datim_max]
    BOX = box + pres + [f'{yr}-01-01', f'{yr}-12-31']
    fetcher = DataFetcher(ds=ds, **fetcher_kwargs).region(BOX).load()

    # Point observations -> profiles; simplify attributes so they can be written to netcdf
    profiles = fetcher.data.argo.point2profile()
    attrs = {'raw_attrs': '', 'Fetched_uri': ''}
    if ds == 'bgc':
        attrs['Fetched_constraints'] = str(profiles.Fetched_constraints).replace('/', '_')
    return profiles.assign_attrs(attrs), fetcher.index


def main():
    parser = pipeline.parser(__doc__)
    parser.add_argument('--dataset', choices=['core', 'bgc'], required=True)
    parser.add_argument('--years', type=int, nargs='+', default=None, help='Default: [run] years')
    args = parser.parse_args()
    run = pipeline.start(args, __file__)
    cfg = run.section(f'L0.{args.dataset}')
    years = run.years()

    # Check every output before the first (slow) download
    outputs = {yr: (run.output('L0_profiles', args.dataset, yr=yr), run.output('L0_index', args.dataset, yr=yr))
               for yr in years}

    # Fresh start
    argopy.reset_options()
    argopy.clear_cache()
    argopy.set_options(cachedir=run.path('argopy_cache'))
    if cfg['source'] == 'gdac':
        argopy.set_options(src='gdac', gdac=run.input('gdac'))
    else:
        argopy.set_options(src='erddap') # Ifremer database, updated daily
    ds = 'phy' if args.dataset == 'core' else 'bgc'
    fetcher_kwargs = dict(run.section('L0.fetcher'))
    if fetcher_kwargs.get('progress') and not sys.stdout.isatty():
        print('No terminal (batch job or redirected output): argopy progress display off')
        fetcher_kwargs['progress'] = False

    for yr in years:
        print(f'===> Fetching {yr} {args.dataset}: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))
        profiles, index = fetch_year(yr, ds, run.box, run.pres, fetcher_kwargs)
        profiles_file, index_file = outputs[yr]
        profiles.assign_attrs(run.attrs()).to_netcdf(profiles_file)
        index.to_csv(index_file)
        print(f'Saved {profiles.sizes["N_PROF"]} profiles to {profiles_file}')
    run.finish(source=cfg['source'], accessed=datetime.now().strftime('%Y%m%d'))


if __name__ == '__main__':
    main()
