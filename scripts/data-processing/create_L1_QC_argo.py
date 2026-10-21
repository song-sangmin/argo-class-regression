"""Create L1 QC'ed core or BGC Argo profiles from the raw (L0) Argopy files, one csv per year.

Same steps as 0.3_coreArgo_processing.ipynb and 0.4_bgcArgo_processing.ipynb (section 3):
    1. Open the L0 profiles from create_L0_raw_argo.py (0.2_argopy_import.ipynb)
    2. Per float, rename the *_ADJUSTED variables, add profid (wmoid + 'cyc' + cycle number)
       and SA, CT, sigma0, spice (gsw) with argoclare's process_argo_float()
    3. Keep observations with QC flags in [L1] use_flags (CRUSOE: 1, 2, 5, 8) with argoclare's
       filter_qc_flags(), on
          core: [L1] core_qc_vars (temperature, salinity, pressure, time, position)
          bgc:  the same plus [L1] bgc_qc_vars (pH). Oxygen is kept with its QC flag but not
                filtered (diagnostic only), so profiles without valid pH are dropped
Writes [files] L1 for each year.

Usage:
    python create_L1_QC_argo.py --config ../../configs/example.toml --dataset core
    python create_L1_QC_argo.py --config ../../configs/example.toml --dataset bgc --years 2020 \\
        --floats 5904184 5904475 --out-dir /tmp/test/
"""
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

import argoclare.argotools as mod_argo

import pipeline


def process_year(yr, l0_file, savefile, bgc_list, qc_vars, use_flags, floats=None):
    """Process and QC one year of L0 profiles, then save to csv."""
    print(f'===> Processing {yr}: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))
    yearDS = xr.open_dataset(l0_file)
    if floats:
        yearDS = yearDS.isel(N_PROF=np.isin(yearDS.PLATFORM_NUMBER.values, floats))
    temp = yearDS.to_dataframe()

    float_dict = {key: mod_argo.process_argo_float(group, bgc_list=bgc_list)
                  for key, group in temp.groupby('PLATFORM_NUMBER')}
    yearDF = pd.concat(float_dict.values(), ignore_index=True).set_index(['profid', 'pressure'])

    yearDF_QC = mod_argo.filter_qc_flags(yearDF, qc_vars=qc_vars, use_flags=use_flags)
    yearDF_QC.to_csv(savefile)
    print('Saved to ' + savefile + '\n')
    return savefile


def main():
    parser = pipeline.parser(__doc__)
    parser.add_argument('--dataset', choices=['core', 'bgc'], required=True)
    parser.add_argument('--years', type=int, nargs='+', default=None, help='Default: [run] years')
    parser.add_argument('--n-jobs', type=int, default=None,
                        help='Years to process in parallel (default: [L1] n_jobs; core years need a lot of memory)')
    parser.add_argument('--floats', type=int, nargs='+', default=None,
                        help='Only process these WMO ids (for testing)')
    args = parser.parse_args()
    run = pipeline.start(args, __file__)
    cfg = run.section('L1')

    if args.dataset == 'core':
        bgc_list, qc_vars = [], cfg['core_qc_vars']
    else: # do not drop on oxygen_qc, only for diagnostic
        bgc_list, qc_vars = cfg['bgc_vars'], cfg['core_qc_vars'] + cfg['bgc_qc_vars']

    jobs = [(yr, run.input('L0_profiles', args.dataset, yr=yr), run.output('L1', args.dataset, yr=yr))
            for yr in run.years()]
    Parallel(n_jobs=args.n_jobs or cfg.get('n_jobs', 1))(
        delayed(process_year)(yr, l0_file, savefile, bgc_list, qc_vars, cfg['use_flags'], args.floats)
        for yr, l0_file, savefile in jobs)
    run.finish(qc_vars=qc_vars, use_flags=cfg['use_flags'])


if __name__ == '__main__':
    main()
