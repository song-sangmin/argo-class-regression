"""Create L2 open ocean profile indexes from the L1 QC'ed core or BGC Argo profiles, one file per year.

Same steps as 0.3_coreArgo_processing.ipynb and 0.4_bgcArgo_processing.ipynb ("Before
interpolation, bathymetry"):
    1. Read the L1 csv from create_L1_QC_argo.py and take the first non-null value of each
       variable per profid (the surfacemost QC'ed observation)
    2. Look up the nearest GEBCO 2024 (sub-ice topography) elevation for each profile and keep
       profiles where elevation < -1000 m, with argoclare's get_bathymetry_DS()
    3. Drop the T/S/P QC flags and errors, and save as a Dataset along profid
Writes <type>_open_ocean_INDEX_<yr>.nc to <type>/L2-mask/.
Bathymetry masking runs before the L3 regridding because it is slow over many profiles; the
L3 scripts (create_L3_gridded_*.py) keep only the L1 profiles listed in these indexes.

Usage:
    python create_L2_masked_argo.py --dataset core                              # 2014-2024
    python create_L2_masked_argo.py --dataset bgc --years 2014 --outpath /tmp/test/
"""
import argparse
import os
from datetime import datetime

import pandas as pd
import xarray as xr

import argoclare.oceantools as mod_ocean

local_data_folder = '/Users/sangminsong/Documents/PROJECTS-PYTHON/CRUSOE-SOCarbon/x_localdata/'
default_bathy_path = '/Volumes/crusoe-repo/data/etopo/gebco_2024_sub_ice_topo/GEBCO_2024_sub_ice_topo.nc'

bathy_lim = -1000
droplist = ['temperature_qc', 'salinity_qc', 'pressure_qc',
            'temp_error', 'psal_error', 'pres_error']

# Titles as written to the files in use (the grid is GEBCO 2024, not ETOPO2)
settings = {
    'core': {'l1_file': 'core_QCflags1258_yr{yr}_{tag}.csv',
             'title': 'Core profiles with valid QC data, masked to open ocean bathymetry '
                      'where ETOPO2 elevation < -1000m'},
    'bgc':  {'l1_file': 'bgc_QCflags1258_wO2_yr{yr}_{tag}.csv',
             'title': 'bgc profiles with valid QC data, masked to open ocean bathymetry '
                      'where ETOPO2 elevation < -1000m'},
}
default_l1_tags = {'core': 'acc20260730', 'bgc': 'acc20260731'}


def first_by_profid(path, chunksize=5_000_000):
    """First non-null value of each column per profid, read in chunks (core L1 files are 2-5 GB).
    Same result as pd.read_csv(path, index_col=0).groupby('profid').first()."""
    chunks = [chunk.groupby('profid').first()
              for chunk in pd.read_csv(path, index_col=0, chunksize=chunksize)]
    return pd.concat(chunks).groupby(level='profid').first()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=['core', 'bgc'], required=True)
    parser.add_argument('--years', type=int, nargs='+', default=list(range(2014, 2025)))
    parser.add_argument('--l1-path', default=None,
                        help='Folder with L1 QC csv files (default: x_localdata/<dataset>/L1-qc/)')
    parser.add_argument('--l1-tag', default=None, help='Date tag in the L1 file names')
    parser.add_argument('--bathy-path', default=default_bathy_path, help='GEBCO elevation netcdf')
    parser.add_argument('--outpath', default=None,
                        help='Folder to save L2 files in (default: x_localdata/<dataset>/L2-mask/)')
    args = parser.parse_args()

    cfg = settings[args.dataset]
    l1_path = args.l1_path or os.path.join(local_data_folder, args.dataset, 'L1-qc')
    l1_tag = args.l1_tag or default_l1_tags[args.dataset]
    savepath = args.outpath or os.path.join(local_data_folder, args.dataset, 'L2-mask')
    os.makedirs(savepath, exist_ok=True)

    bathy_nc = xr.open_dataset(args.bathy_path).elevation
    for yr in args.years:
        print(f'===> Processing {yr} {args.dataset}: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))
        plat_index = first_by_profid(os.path.join(l1_path, cfg['l1_file'].format(yr=yr, tag=l1_tag)))
        INDEX = mod_ocean.get_bathymetry_DS(plat_index, bathy_nc=bathy_nc, dropvars=droplist,
                                            bathy_lim=bathy_lim,
                                            save_path=savepath + os.sep,
                                            save_tag=f'{args.dataset}_open_ocean_INDEX_{yr}',
                                            nc_attrs={'title': cfg['title']})
        print(f'  kept {INDEX.sizes["profid"]} of {len(plat_index)} profiles')


if __name__ == '__main__':
    main()
