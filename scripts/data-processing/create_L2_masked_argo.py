"""Create L2 open ocean profile indexes from the L1 QC'ed core or BGC Argo profiles, one file per year.

Same steps as 0.3_coreArgo_processing.ipynb and 0.4_bgcArgo_processing.ipynb ("Before
interpolation, bathymetry"):
    1. Read the L1 csv from create_L1_QC_argo.py and take the first non-null value of each
       variable per profid (the surfacemost QC'ed observation)
    2. Look up the nearest elevation in [files] bathy (CRUSOE: GEBCO 2024 sub-ice topography)
       for each profile and keep profiles where elevation < [L2] bathy_lim, with argoclare's
       get_bathymetry_DS()
    3. Drop the T/S/P QC flags and errors, and save as a Dataset along profid
Writes [files] L2 for each year.
Bathymetry masking runs before the L3 regridding because it is slow over many profiles; the
L3 scripts (create_L3_gridded_*.py) keep only the L1 profiles listed in these indexes.

Usage:
    python create_L2_masked_argo.py --config ../../configs/example.toml --dataset core
    python create_L2_masked_argo.py --config ../../configs/example.toml --dataset bgc --out-dir /tmp/test/
"""
import os
from datetime import datetime

import pandas as pd
import xarray as xr

import argoclare.oceantools as mod_ocean
from argoclare import pipeline


droplist = ['temperature_qc', 'salinity_qc', 'pressure_qc',
            'temp_error', 'psal_error', 'pres_error']


def first_by_profid(path, chunksize=5_000_000):
    """First non-null value of each column per profid, read in chunks (core L1 files are 2-5 GB).
    Same result as pd.read_csv(path, index_col=0).groupby('profid').first()."""
    chunks = [chunk.groupby('profid').first()
              for chunk in pd.read_csv(path, index_col=0, chunksize=chunksize)]
    return pd.concat(chunks).groupby(level='profid').first()


def main():
    parser = pipeline.parser(__doc__)
    parser.add_argument('--dataset', choices=['core', 'bgc'], required=True)
    parser.add_argument('--years', type=int, nargs='+', default=None, help='Default: [run] years')
    args = parser.parse_args()
    run = pipeline.start(args, __file__)
    bathy_lim = run.section('L2')['bathy_lim']
    label = 'Core' if args.dataset == 'core' else 'bgc'
    title = f'{label} profiles with valid QC data, masked to open ocean bathymetry where elevation < {bathy_lim:g}m'

    jobs = [(yr, run.input('L1', args.dataset, yr=yr), run.output('L2', args.dataset, yr=yr))
            for yr in run.years()]
    bathy_nc = xr.open_dataset(run.input('bathy')).elevation
    for yr, l1_file, savefile in jobs:
        print(f'===> Processing {yr} {args.dataset}: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))
        plat_index = first_by_profid(l1_file)
        INDEX = mod_ocean.get_bathymetry_DS(plat_index, bathy_nc=bathy_nc, dropvars=droplist,
                                            bathy_lim=bathy_lim,
                                            save_path=os.path.dirname(savefile) + os.sep,
                                            save_tag=os.path.splitext(os.path.basename(savefile))[0],
                                            nc_attrs={'title': title, 'bathy_file': os.path.basename(run.path('bathy')),
                                                      **run.attrs()})
        print(f'  kept {INDEX.sizes["profid"]} of {len(plat_index)} profiles')
    run.finish()


if __name__ == '__main__':
    main()
