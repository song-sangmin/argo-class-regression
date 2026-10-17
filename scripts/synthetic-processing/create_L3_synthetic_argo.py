"""Create synthetic Argo L3 datasets (core and bgc) from the E3SM synthetic floats.

Same steps as 4.2_synthetic_regridding_clustering.ipynb, with outputs laid out like the real
L3 files (create_L3_gridded_*.py) so fit_argo_clustering.py and the P1/P2 scripts can read them:
    1. Read the merged synthetic floats (fnum x profnum x nVertLevels). Each fnum is the WMO of
       the real float it follows: bgc = fnums that are SOCCOM floats in the real P1 bgc file,
       core = fnums in the real P1 core file that are not bgc. Other fnums are dropped.
    2. Surface (per profile) values from the model: wind stress = hypot(zonal, meridional) and
       wind speed from stress (argoclare oceantools.wind_speed(), solved for all profiles at
       once); pCO2surface == 0 marks missing pco2_ocean, pco2_atmos, delta_pco2 and
       atmospheric pressure; renamed to match the real data (atmPressure -> atmos_pres_Pa,
       dpCO2 -> delta_pco2, pCO2surface -> pco2_ocean, atmosphericCO2 -> pco2_atmos,
       ifrac -> sea_ice, MLD -> mld, lat/lon -> latitude/longitude, fnum -> wmoid)
    3. QC as in 4.2: keep profiles south of 35S with atmospheric pressure > 0.88 atm (JRA minimum)
    4. pressure = gsw.p_from_z(Depth, latitude); SA, CT from Salinity and Temperature
    5. Regrid CT and SA to the real L3 levels (0-1000 dbar every 5, 1050-1950 every 50) with
       argoclare's regrid_pressure_levels() (pchip, z_gap 1000, surface_fill 25), in parallel
profid = <fnum>cyc<profnum, 4 digits>; prof_datetag = <fnum>_<YYYYMMDDHH>.
Writes to working-vars/synthetic/L3-interp/:
    synthetic_<core|bgc>DATA_byProfid_<years>_acc<datetag>.nc   CT, SA on (profid, pressure)
    synthetic_<core|bgc>INDEX_byProfid_<years>_acc<datetag>.nc  one row per profid: CT/SA at
        5 dbar (from the regridded profile, as 4.3/4.4), mld, mlp, the model surface variables,
        time and position

Usage:
    python create_L3_synthetic_argo.py
    python create_L3_synthetic_argo.py --max-floats 20 --outpath /tmp/test/
"""
import argparse
import os
import sys
from datetime import datetime

import gsw
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

import argoclare.argotools as mod_argo

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import importer # noqa: E402

default_merged = importer.local_data_folder + 'LIGHT-bgcArgo/synArgo_realworld_matches_merged.nc'
default_outpath = importer.local_working_folder + 'synthetic/L3-interp/'
pres_regular = np.arange(0, 1001, 5).tolist() + np.arange(1050, 1951, 50).tolist()
surface_pressure = 5 # dbar level used for the surface CT/SA in INDEX

surface_vars = ['lat', 'lon', 'time', 'ifrac', 'MLD', 'atmPressure', 'windStressZonal', 'windStressMerid',
                'CO2_gas_flux', 'pCO2surface', 'dpCO2', 'atmosphericCO2']
rename = {'atmPressure': 'atmos_pres_Pa', 'dpCO2': 'delta_pco2', 'pCO2surface': 'pco2_ocean',
          'lat': 'latitude', 'lon': 'longitude', 'ifrac': 'sea_ice', 'fnum': 'wmoid', 'MLD': 'mld',
          'atmosphericCO2': 'pco2_atmos', 'time': 'datetime'}


def wind_speed_from_stress(tau, rho_air=1.2):
    """Vectorized argoclare.oceantools.wind_speed(): the real root u of
    a0 u^3 + a1 u^2 + a2 u = tau / rho_air (Large & Yeager 2004 drag). The cubic increases
    monotonically, so it has exactly one real root (Cardano). NaN where tau is 0 or missing."""
    a0, a1, a2 = 0.0000764, 0.000142, 0.0027
    tau = np.asarray(tau, dtype=float)
    b, c, d = a1 / a0, a2 / a0, -tau / rho_air / a0
    p = c - b**2 / 3
    q = 2 * b**3 / 27 - b * c / 3 + d
    disc = np.sqrt(q**2 / 4 + p**3 / 27)
    u = np.cbrt(-q / 2 + disc) + np.cbrt(-q / 2 - disc) - b / 3
    return np.where((tau == 0) | ~np.isfinite(tau), np.nan, u)


def process_surface(df):
    """Model surface variables (original names, one row per sample) -> wind speed, missing pCO2,
    real-data names, atmos_pres_atm and QC (south of 35S, atmos pressure > 0.88 atm).
    Shared with create_P1_synthetic_socat.py."""
    df = df.dropna(subset=['lat', 'time']).copy()
    df['windStress'] = np.hypot(df['windStressZonal'], df['windStressMerid'])
    df['wind_speed'] = wind_speed_from_stress(df['windStress'].values)
    zero_pco2 = df['pCO2surface'] == 0
    for var in ['pCO2surface', 'atmosphericCO2', 'atmPressure', 'dpCO2']:
        df[var] = df[var].where(~zero_pco2)
    df = df.rename(columns=rename)
    df['atmos_pres_atm'] = df['atmos_pres_Pa'] / 101325
    df = df[(df['latitude'] < -35) & (df['atmos_pres_atm'] > 0.88)].copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def add_ct_sa(df):
    """pressure from Depth; SA, CT from Salinity and Temperature (needs latitude, longitude)."""
    df['pressure'] = gsw.p_from_z(df['Depth'].values, df['latitude'].values)
    df['SA'] = gsw.SA_from_SP(df['Salinity'].values, df['pressure'].values, df['longitude'].values, df['latitude'].values)
    df['CT'] = gsw.CT_from_t(df['SA'].values, df['Temperature'].values, df['pressure'].values)
    return df


def regrid_ct_sa(df, levels):
    """Regrid CT and SA (df indexed by profid, with wmoid and pressure) as in 4.2/4.4."""
    df = df[df.groupby(level=0).pressure.transform('size') >= 2]
    regular = mod_argo.regrid_pressure_levels(df, levels, var_list=['CT', 'SA'], z_gap=1000, surface_fill=25)
    return regular[['CT', 'SA']]


def surface_table(ds, fnums):
    """One row per synthetic profile with the model surface variables, after QC."""
    sub = ds[surface_vars].drop_dims('nVertLevels', errors='ignore').sel(fnum=fnums).load()
    df = process_surface(sub.to_dataframe().reset_index())
    df['profid'] = df['wmoid'].astype(str) + 'cyc' + df['profnum'].astype(str).str.zfill(4)
    df['prof_datetag'] = df['wmoid'].astype(str) + '_' + df['datetime'].dt.strftime('%Y%m%d%H')
    df['mlp'] = gsw.p_from_z(-df['mld'].values, df['latitude'].values)
    return df.drop(columns=[c for c in ['nParticles_index'] if c in df.columns]).set_index('profid')


def grid_chunk(merged_path, fnums, keep_profids):
    """Regrid CT/SA for the kept profiles of a group of floats."""
    ds = xr.open_dataset(merged_path)
    prof = ds[['Temperature', 'Salinity', 'Depth', 'lat', 'lon']].sel(fnum=fnums).load()
    df = prof.to_dataframe().reset_index().dropna(subset=['Temperature', 'Salinity', 'Depth', 'lat'])
    df['profid'] = df['fnum'].astype(str) + 'cyc' + df['profnum'].astype(str).str.zfill(4)
    df = df[df['profid'].isin(keep_profids)].rename(columns={'fnum': 'wmoid', 'lat': 'latitude', 'lon': 'longitude'})
    if df.empty:
        return None
    return regrid_ct_sa(add_ct_sa(df).set_index('profid'), pres_regular)


def build_dataset(name, ds, fnums, args):
    """Surface table + regridded profiles for one set of synthetic floats."""
    print(f'===> {name}: {len(fnums)} floats: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))
    surf = surface_table(ds, fnums)
    print(f'  {len(surf)} profiles after QC (south of 35S, atmos pressure > 0.88 atm); regridding...')
    chunks = [fnums[i:i + args.chunk_floats] for i in range(0, len(fnums), args.chunk_floats)]
    keep = {int(f): set(g.index) for f, g in surf.groupby('wmoid')}
    parts = Parallel(n_jobs=args.n_jobs)(
        delayed(grid_chunk)(args.merged, chunk, set().union(*[keep.get(int(f), set()) for f in chunk]))
        for chunk in chunks)
    regular = pd.concat([p for p in parts if p is not None])

    DATA = xr.Dataset.from_dataframe(regular).reindex(pressure=pres_regular)
    profids = DATA.profid.values
    # Surface CT/SA at 5 dbar of the regridded profile (as 4.3/4.4 and synthetic SOCAT)
    index = surf.loc[profids].copy()
    index['pressure'] = float(surface_pressure)
    for var in ['CT', 'SA']:
        index[var] = DATA[var].sel(pressure=surface_pressure).values
    INDEX = xr.Dataset.from_dataframe(index.rename_axis('profid')).set_coords('prof_datetag')
    DATA = DATA.assign_coords(prof_datetag=('profid', INDEX['prof_datetag'].values))
    print(f'  regridded {DATA.sizes["profid"]} profiles: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))
    return DATA, INDEX


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--merged', default=default_merged, help='Merged synthetic float netcdf')
    parser.add_argument('--bgc-p1', default=importer.path_to('P1', 'bgc'), help='Real P1 bgc file (SOCCOM wmoids)')
    parser.add_argument('--core-p1', default=importer.path_to('P1', 'core'), help='Real P1 core file (core wmoids)')
    parser.add_argument('--outpath', default=default_outpath)
    parser.add_argument('--n-jobs', type=int, default=8, help='Parallel workers for regridding')
    parser.add_argument('--chunk-floats', type=int, default=25, help='Floats per regridding task')
    parser.add_argument('--max-floats', type=int, default=None, help='Only use the first N floats of each set (testing)')
    args = parser.parse_args()

    ds = xr.open_dataset(args.merged)
    bgc_w = set(pd.Series(xr.open_dataset(args.bgc_p1).wmoid.values).astype(int))
    core_w = set(xr.open_dataset(args.core_p1).wmoid.values.astype(int)) - bgc_w
    fnum = ds.fnum.values
    sets = {'bgc': fnum[np.isin(fnum, list(bgc_w))], 'core': fnum[np.isin(fnum, list(core_w))]}
    print(f'{len(fnum)} synthetic floats: {len(sets["bgc"])} bgc, {len(sets["core"])} core, '
          f'{len(fnum) - len(sets["bgc"]) - len(sets["core"])} matching neither (dropped)')

    os.makedirs(args.outpath, exist_ok=True)
    datetag = datetime.now().strftime('%Y%m%d')
    for name, fnums in sets.items():
        if args.max_floats:
            fnums = fnums[:args.max_floats]
        DATA, INDEX = build_dataset(name, ds, fnums, args)
        years = pd.to_datetime(INDEX['datetime'].values).year
        yrs = f'{years.min()}-{years.max()}' if years.max() > years.min() else str(years.min())
        attrs = {'source': f'E3SM synthetic Argo ({os.path.basename(args.merged)}), floats following '
                           f'real {name} Argo WMOs', 'script': os.path.basename(__file__), 'date': datetag,
                 'qc': 'latitude < -35, atmos_pres_atm > 0.88; pCO2surface == 0 treated as missing'}
        DATA = DATA.assign_attrs({**attrs, 'title': f'Synthetic {name} Argo profiles {yrs}, CT and SA regridded '
                                  f'(pchip) to n={len(pres_regular)} pressure levels as the real L3',
                                  'pressure_levels': 'Pressure levels in dbar: ' + str(pres_regular)})
        INDEX = INDEX.assign_attrs({**attrs, 'title': f'Synthetic {name} Argo profiles {yrs}: CT/SA at 5 dbar, '
                                    'CT/SA, model surface variables (sea_ice, mld, wind_speed, atmos_pres, '
                                    'pco2_ocean, pco2_atmos, delta_pco2, CO2_gas_flux), time and position'})
        for kind, dset in [('DATA', DATA), ('INDEX', INDEX)]:
            f = os.path.join(args.outpath, f'synthetic_{name}{kind}_byProfid_{yrs}_acc{datetag}.nc')
            dset.to_netcdf(f)
            print(f'  saved {f}')


if __name__ == '__main__':
    main()
