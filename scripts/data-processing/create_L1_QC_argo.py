"""Create L1 QC'ed core or BGC Argo profiles from the raw (L0) Argopy files, one csv per year.

Same steps as 0.3_coreArgo_processing.ipynb and 0.4_bgcArgo_processing.ipynb (section 3):
    1. Open the L0 profiles from create_L0_raw_argo.py (0.2_argopy_import.ipynb)
    2. Per float, rename the *_ADJUSTED variables, add profid (wmoid + 'cyc' + cycle number)
       and SA, CT, sigma0, spice (gsw) with argoclare's process_argo_float()
    3. Keep observations with QC flags 1, 2, 5, 8 with argoclare's filter_qc_flags()
          core: temperature, salinity, pressure, time, position
          bgc:  same plus pH. Oxygen is kept with its QC flag but not filtered (diagnostic only),
                so profiles without valid pH are dropped
Writes, to <type>/L1-qc/:
    core: core_QCflags1258_yr<yr>_acc<datetag>.csv     (files in use: acc20260730)
    bgc:  bgc_QCflags1258_wO2_yr<yr>_acc<datetag>.csv  (files in use: acc20260731)

Usage:
    python create_L1_QC_argo.py --dataset core                                  # 2014-2024
    python create_L1_QC_argo.py --dataset bgc --years 2014 --floats 5904184 5904475 --outpath /tmp/test/
"""
import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

import argoclare.argotools as mod_argo

root_data_folder = '/Volumes/crusoe-repo/data/'
local_data_folder = '/Users/sangminsong/Documents/PROJECTS-PYTHON/CRUSOE-SOCarbon/x_localdata/'

use_flags = ['1', '2', '5', '8']
core_qcs = ['temperature_qc', 'salinity_qc', 'pressure_qc', 'time_qc', 'position_qc']

settings = {
    'core': {'l0_file': '202501snap_yr{yr}_core_profiles_20250207.nc',
             'bgc_list': [],
             'qc_vars': core_qcs,
             'savetag': 'core_QCflags1258_yr{yr}_acc{datetag}.csv'},
    'bgc':  {'l0_file': 'ERDDAP_yr{yr}_bgc_profiles_20250219.nc',
             'bgc_list': ['pH', 'oxygen'],
             'qc_vars': core_qcs + ['pH_qc'], # Do not drop on oxygen_qc, only for diagnostic
             'savetag': 'bgc_QCflags1258_wO2_yr{yr}_acc{datetag}.csv'},
}


def process_year(yr, dataset, l0_path, savepath, datetag, floats=None):
    """Process and QC one year of L0 profiles, then save to csv."""
    cfg = settings[dataset]
    print(f'===> Processing {yr} {dataset}: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))
    yearDS = xr.open_dataset(os.path.join(l0_path, cfg['l0_file'].format(yr=yr)))
    if floats:
        yearDS = yearDS.isel(N_PROF=np.isin(yearDS.PLATFORM_NUMBER.values, floats))
    temp = yearDS.to_dataframe()

    float_dict = {key: mod_argo.process_argo_float(group, bgc_list=cfg['bgc_list'])
                  for key, group in temp.groupby('PLATFORM_NUMBER')}
    yearDF = pd.concat(float_dict.values(), ignore_index=True).set_index(['profid', 'pressure'])

    yearDF_QC = mod_argo.filter_qc_flags(yearDF, qc_vars=cfg['qc_vars'], use_flags=use_flags)

    savefile = os.path.join(savepath, cfg['savetag'].format(yr=yr, datetag=datetag))
    yearDF_QC.to_csv(savefile)
    print('Saved to ' + savefile + '\n')
    return savefile


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=['core', 'bgc'], required=True)
    parser.add_argument('--years', type=int, nargs='+', default=list(range(2014, 2025)))
    parser.add_argument('--l0-path', default=None,
                        help='Folder with L0 Argopy files (default: crusoe-repo data/<dataset>/)')
    parser.add_argument('--outpath', default=None,
                        help='Folder to save L1 files in (default: x_localdata/<dataset>/L1-qc/)')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Years to process in parallel (core years need a lot of memory)')
    parser.add_argument('--floats', type=int, nargs='+', default=None,
                        help='Only process these WMO ids (for testing)')
    args = parser.parse_args()

    l0_path = args.l0_path or os.path.join(root_data_folder, args.dataset)
    savepath = args.outpath or os.path.join(local_data_folder, args.dataset, 'L1-qc')
    os.makedirs(savepath, exist_ok=True)
    datetag = datetime.now().strftime('%Y%m%d')

    Parallel(n_jobs=args.n_jobs)(
        delayed(process_year)(yr, args.dataset, l0_path, savepath, datetag, args.floats)
        for yr in args.years)


if __name__ == '__main__':
    main()
