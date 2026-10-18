"""Registry of the CRUSOE processed datasets in use, by processing level.

One place to look up (and open) the file each level should read, so P1/P2 runs point at the
current, corrected versions. Update the entry here when a dataset is regenerated.

    L1  QC'ed float profiles, one csv per year         (data-processing/create_L1_QC_argo.py)
    L2  open ocean indexes; resampled SOCAT            (create_L2_masked_argo.py, create_L2_resampled_masked_socat.py)
    L3  floats regridded to pressure levels + MLD      (create_L3_gridded_coreArgo.py, create_L3_gridded_bgcArgo.py)
    P1  one row per sample with regression features    (create_P1_core/bgc/socat_features.py)
    P2  clustered (to be added)

Usage (with argoclare installed, e.g. pip install -e . from the repo root):
    import argoclare.importer as importer
    importer.status()                                   # what is registered, and whether it exists
    path = importer.path_to('P1', 'socat')
    coreINDEX = importer.open_data('L3', 'coreINDEX')
    bgcL1 = importer.open_data('L1', 'bgc', yr=2018)
"""
import os

import pandas as pd
import xarray as xr

root_data_folder = '/Volumes/crusoe-repo/data/' # large files, but permanent data, all publicly available
local_data_folder = '/Users/sangminsong/Documents/PROJECTS-PYTHON/CRUSOE-SOCarbon/x_localdata/'
local_working_folder = local_data_folder + 'working-vars/'

YEARS = list(range(2014, 2025))

# path: file, or pattern with {yr} for one file per year
DATASETS = {
    'L1': {
        'core': {'path': local_data_folder + 'core/L1-qc/core_QCflags1258_yr{yr}_acc20260730.csv',
                 'script': 'create_L1_QC_argo.py --dataset core',
                 'notes': 'QC flags 1,2,5,8 on T/S/P/time/position'},
        'bgc':  {'path': local_data_folder + 'bgc/L1-qc/bgc_QCflags1258_wO2_yr{yr}_acc20260731.csv',
                 'script': 'create_L1_QC_argo.py --dataset bgc',
                 'notes': 'Also QC on pH; oxygen kept but not QC filtered'},
    },
    'L2': {
        'core':  {'path': local_data_folder + 'core/L2-mask/core_open_ocean_INDEX_{yr}.nc',
                  'script': 'create_L2_masked_argo.py --dataset core',
                  'notes': 'GEBCO 2024 elevation < -1000 m'},
        'bgc':   {'path': local_data_folder + 'bgc/L2-mask/bgc_open_ocean_INDEX_{yr}.nc',
                  'script': 'create_L2_masked_argo.py --dataset bgc',
                  'notes': 'GEBCO 2024 elevation < -1000 m'},
        'socat': {'path': local_data_folder + 'socat/SOCATv2025_SO_randomresample23p5h_seed0_open_ocean_INDEX_acc20261016.nc',
                  'script': 'create_L2_resampled_masked_socat.py',
                  'notes': 'Random 23.5 h resample (seed 0), 72 h gaps, GEBCO < -1000 m; dim sample'},
    },
    'L3': {
        'coreINDEX': {'path': local_data_folder + 'core/L3-interp/coreINDEX_byProfid_valid_interp_2014-2024_acc20261015.nc',
                      'script': 'create_L3_gridded_coreArgo.py',
                      'notes': 'Surfacemost obs, mlp/mld, prof_datetag; descending casts dropped'},
        'coreDATA':  {'path': local_data_folder + 'core/L3-interp/coreDATA_byProfid_valid_interp_2014-2024_acc20261015.nc',
                      'script': 'create_L3_gridded_coreArgo.py',
                      'notes': '0-1000 dbar every 5, 1050-1950 every 50'},
        'bgcINDEX':  {'path': local_data_folder + 'bgc/L3-interp/bgcINDEX_byProfid_valid_interp_2014-2024_acc20261015.nc',
                      'script': 'create_L3_gridded_bgcArgo.py',
                      'notes': 'Surfacemost obs (incl. pH), mlp/mld, prof_datetag'},
        'bgcDATA':   {'path': local_data_folder + 'bgc/L3-interp/bgcDATA_byProfid_valid_interp_2014-2024_acc20261015.nc',
                      'script': 'create_L3_gridded_bgcArgo.py',
                      'notes': 'T/S and pH on the same levels as coreDATA'},
    },
    'P1': {
        'core':  {'path': local_working_folder + 'P1-processed/P1_coreArgo_application_features_yr2014-2023_acc20261016.nc',
                  'script': 'create_P1_core_features.py',
                  'notes': 'One row per L3 core profile (dim profid); SLP and MBL 2024 fixes'},
        'bgc':   {'path': local_working_folder + 'P1-processed/P1_bgcArgo_soccom5m_features_yr2014-2024_acc20261016.nc',
                  'script': 'create_P1_bgc_features.py',
                  'notes': 'SOCCOM 5 m averages, pco2_ocean = pCO2_pHbias5_pK01; profid within 1 h ("" if none)'},
        'socat': {'path': local_working_folder + 'P1-processed/P1_socatv2025_features_yr2014-2024_acc20261016.nc',
                  'script': 'create_P1_socat_features.py --l2 <L2 socat>',
                  'notes': 'Nearest clusterable profile (CT/SA above 500 dbar) within 7 d ("" if none)'},
    },
    'P2': {
        'pcm_finder': {'path': local_working_folder + 'P2-clustered/pc8_gmm6_dbar500/PCM_finder_pc8_gmm6_dbar500_acc20261016.parquet',
                       'script': 'clustering/fit_argo_clustering.py',
                       'notes': 'pc8_gmm6_dbar500 on Oct 15 L3 core + bgc; classes north (1) to south; fit metadata in df.attrs'},
        'trainval':   {'path': local_working_folder + 'P2-clustered/pc8_gmm6_dbar500/P2_combined_trainval_socat_soccom_2014-2023_acc20261016.parquet',
                       'script': 'clustering/apply_P2_clustering.py',
                       'notes': 'bgcArgo (SOCCOM 5 m) + socatv2025 with classes, 2014-2023; unclassified dropped'},
        'test':       {'path': local_working_folder + 'P2-clustered/pc8_gmm6_dbar500/P2_combined_test_socat_soccom_2024_acc20261016.parquet',
                       'script': 'clustering/apply_P2_clustering.py',
                       'notes': 'bgcArgo + socatv2025 with classes, 2024'},
        'core':       {'path': local_working_folder + 'P2-clustered/pc8_gmm6_dbar500/P2_coreArgo_application_2014-2023_acc20261016.parquet',
                       'script': 'clustering/apply_P2_core_clustering.py --exclude-p2 <trainval> <test>',
                       'notes': 'Core application set 2014-2023; SOCCOM training/test profiles excluded'},
    },
    # Synthetic (E3SM) Argo, scripts in synthetic-processing/ (see its README)
    'SYN': {
        'L3_coreINDEX':  {'path': local_working_folder + 'synthetic/L3-interp/synthetic_coreINDEX_byProfid_2014-2023_acc20261016.nc',
                          'script': 'synthetic-processing/create_L3_synthetic_argo.py',
                          'notes': 'Synthetic floats following real core WMOs; CT/SA at 5 dbar, model surface variables'},
        'L3_coreDATA':   {'path': local_working_folder + 'synthetic/L3-interp/synthetic_coreDATA_byProfid_2014-2023_acc20261016.nc',
                          'script': 'synthetic-processing/create_L3_synthetic_argo.py',
                          'notes': 'CT, SA on the real L3 pressure levels'},
        'L3_bgcINDEX':   {'path': local_working_folder + 'synthetic/L3-interp/synthetic_bgcINDEX_byProfid_2014-2023_acc20261016.nc',
                          'script': 'synthetic-processing/create_L3_synthetic_argo.py',
                          'notes': 'Synthetic floats following real SOCCOM WMOs'},
        'L3_bgcDATA':    {'path': local_working_folder + 'synthetic/L3-interp/synthetic_bgcDATA_byProfid_2014-2023_acc20261016.nc',
                          'script': 'synthetic-processing/create_L3_synthetic_argo.py',
                          'notes': 'CT, SA on the real L3 pressure levels'},
        'P1_core':       {'path': local_working_folder + 'synthetic/P1-processed/P1_synArgo_core_application_features_yr2014-2023_acc20261016.nc',
                          'script': 'synthetic-processing/create_P1_synthetic_features.py',
                          'notes': 'Model features + time vars + max_solar_rad (no ADT)'},
        'P1_bgc':        {'path': local_working_folder + 'synthetic/P1-processed/P1_synArgo_bgc_features_yr2014-2023_acc20261016.nc',
                          'script': 'synthetic-processing/create_P1_synthetic_features.py',
                          'notes': 'Model features + time vars + max_solar_rad (no ADT)'},
        'P1_socat':      {'path': local_working_folder + 'synthetic/P1-processed/P1_synSOCAT_features_yr2014-2023_acc20261016.nc',
                          'script': 'synthetic-processing/create_P1_synthetic_socat.py',
                          'notes': 'Colocated to nearest clusterable synthetic profile (7 d); mld from that profile'},
        'pcm_finder':    {'path': local_working_folder + 'synthetic/P2-clustered/pc8_gmm6_dbar500/PCM_finder_pc8_gmm6_dbar500_acc20261016.parquet',
                          'script': 'clustering/fit_argo_clustering.py (synthetic L3 inputs)',
                          'notes': 'GMM refit on synthetic profiles; classes north to south (not the real class numbers)'},
        'P2_trainval':   {'path': local_working_folder + 'synthetic/P2-clustered/pc8_gmm6_dbar500/P2_synthetic_combined_trainval_socat_bgc_2014-2023_acc20261016.parquet',
                          'script': 'synthetic-processing/apply_P2_synthetic_clustering.py',
                          'notes': 'synBgcArgo + synSOCAT with classes; unclassified dropped'},
        'P2_core':       {'path': local_working_folder + 'synthetic/P2-clustered/pc8_gmm6_dbar500/P2_synArgo_core_application_2014-2023_acc20261016.parquet',
                          'script': 'synthetic-processing/apply_P2_synthetic_synCore_clustering.py',
                          'notes': 'synCoreArgo application set with classes; unclassified dropped'},
    },
}


def path_to(level, name, yr=None):
    """Registered path for a dataset; per-year datasets need yr."""
    entry = DATASETS[level][name]
    path = entry['path']
    if '{yr}' in path:
        if yr is None:
            raise ValueError(f'{level} {name} is stored per year; pass yr (one of {YEARS[0]}-{YEARS[-1]})')
        path = path.format(yr=yr)
    return path


def open_data(level, name, yr=None, **kwargs):
    """Open a registered dataset: netcdf with xarray, csv with pandas (index_col=0)."""
    path = path_to(level, name, yr)
    if path.endswith('.nc'):
        return xr.open_dataset(path, **kwargs)
    return pd.read_csv(path, index_col=0, **kwargs)


def status(levels=None):
    """Print each registered dataset and whether its file(s) exist."""
    for level, entries in DATASETS.items():
        if levels and level not in levels:
            continue
        print(f'== {level}' + ('' if entries else '  (none registered)'))
        for name, entry in entries.items():
            if '{yr}' in entry['path']:
                missing = [yr for yr in YEARS if not os.path.exists(entry['path'].format(yr=yr))]
                state = 'ok' if not missing else f'MISSING {missing}'
                shown = os.path.basename(entry['path'])
            else:
                state = 'ok' if os.path.exists(entry['path']) else 'MISSING'
                shown = os.path.basename(entry['path'])
            print(f'  {name:10s} {state:8s} {shown}')
            print(f'  {"":10s} {"":8s} {entry["notes"]}  [{entry["script"]}]')


if __name__ == '__main__':
    status()
