"""Combine NOAA GML marine boundary layer (MBL) CO2 text files into one netcdf.

Same structure as notebooks/data_access/noaa-mbl/marineboundarylayer_xco2.ipynb:
'mean' and 'sdev' (ppm, X2019 scale) on (decimalyr, sin_lat), with sin_lat = -1.00 to -0.50
in steps of 0.05 (90S to 30S) and a 'lat_deg' coordinate. The time step is 1/48 yr.
The old co2_mbl_2014-2023_dataset_combined.nc ended at decimalyr 2024.0, so every 2024
sample got the 2024-01-01 value; this one runs through 2025.0.

Reads [files] mbl_txt, downloaded from https://gml.noaa.gov/ccgg/GHGreference/ (surface, sine
latitude steps), and writes [files] mbl. Rows shared between files (e.g. 2021.0) are kept once.

Usage:
    python create_mbl_co2_dataset.py --config ../../configs/example.toml
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

import pipeline

sin_lat = np.round(np.arange(-1.00, -0.499, 0.05), 2)


def read_mbl_txt(path):
    """One GHGreference surface file -> DataFrame indexed by decimalyr, columns (sin_lat, mean/sdev)."""
    df = pd.read_csv(path, comment='#', sep=r'\s+', header=None)
    assert df.shape[1] == 1 + 2 * len(sin_lat), f'unexpected columns in {path}'
    df = df.set_index(0)
    df.index.name = 'decimalyr'
    df.columns = pd.MultiIndex.from_product([sin_lat, ['mean', 'sdev']])
    return df


def main():
    parser = pipeline.parser(__doc__)
    args = parser.parse_args()
    run = pipeline.start(args, __file__)
    files = run.input('mbl_txt')
    files = [files] if isinstance(files, str) else files
    out = run.output('mbl')

    df = pd.concat([read_mbl_txt(f) for f in files])
    dup = df.index.duplicated(keep='first')
    shared = df[df.index.isin(df.index[dup])]
    assert (shared.groupby(level=0).nunique() == 1).all().all(), 'files disagree on shared rows'
    df = df[~dup].sort_index()
    assert np.allclose(np.diff(df.index.values), 1 / 48, atol=1e-5), 'gap in the time steps'

    ds = xr.Dataset(
        {var: (('decimalyr', 'sin_lat'), df.xs(var, axis=1, level=1)[sin_lat].values)
         for var in ['mean', 'sdev']},
        coords={'decimalyr': df.index.values, 'sin_lat': sin_lat,
                'lat_deg': ('sin_lat', np.degrees(np.arcsin(sin_lat)))})
    yrs = f'{int(df.index.min())} to {df.index.max():g}'
    ds.attrs = {'description': f'NOAA Marine Boundary Layer CO2 data (surface), decimalyr {yrs}',
                'units': 'ppm (X2019 scale); sdev is the uncertainty',
                'source': 'https://gml.noaa.gov/ccgg/GHGreference/',
                'citation': 'Lan, X., Tans, P., Thoning, K., & NOAA Global Monitoring Laboratory (2024). '
                            'NOAA Greenhouse Gas Marine Boundary Layer Reference - CO2. '
                            'https://doi.org/10.15138/DVNP-F961',
                'files': ', '.join(os.path.basename(f) for f in files),
                **run.attrs(),
                'date_created': datetime.now().strftime('%Y-%m-%d')}
    ds.to_netcdf(out)
    print(f'Saved {ds.sizes["decimalyr"]} time steps ({yrs}) x {ds.sizes["sin_lat"]} latitudes to {out}')
    run.finish()


if __name__ == '__main__':
    main()
