import os
from tqdm import tqdm
import numpy                 as np
import pandas                as pd
import xarray                as xr
from   datetime              import date, datetime, timedelta               
import scipy
import gsw

# ======== internal 
import crusoe.oceantools as mod_ocean
import crusoe.argotools as mod_argo
import crusoe.preprocessing as mod_prep
import crplot.figtools as mod_plot


# %% Import data saved from argopy-import.py 
path = '/Volumes/crusoe-repo/data/bgc/'

annual_bgc_profiles = {key:None for key in [str(x) for x in range(2014,2024)]}
annual_bgc_index = {key:None for key in [str(x) for x in range(2014,2024)]}

for x in os.listdir(path):
    if x.startswith('ERDDAP_yr') & x.endswith('.nc'):
        annual_bgc_profiles[x[9:13]] = xr.open_dataset(path +x)
        print('Imported data: ' + x)
    elif x.startswith('ERDDAP_yr') & x.endswith('.csv'):
        annual_bgc_index[x[9:13]] = pd.read_csv(path +x)
        print('Imported index: ' + x)


# %% (opt1) Joblib -- Convert to float dataframes, run QC filters 

# ===== Set save parameters
save=True
savepath = '/Users/sangminsong/Documents/PROJECTS-PYTHON/CRUSOE-SOCarbon/x_localdata/core/L1-qc/'
# savepath = '/Volumes/crusoe-repo/data/core/L1-qc/'
datetag = datetime.now().strftime('%Y%m%d')

# # ===== Initialize if running for first time
# coreDF = {k:None for k in annual_core_profiles.keys()}
# coreDF_QC = {k:None for k in annual_core_profiles.keys()}
# coreDF_regular = {k:None for k in annual_core_profiles.keys()}

# ===== Choose years to iterate over
# yrlist = [str(x) for x in range(2014,2025)]
yrlist = ['2021', '2023', '2024']

from joblib import Parallel, delayed

def process_argo_year(yr, save=False):
    print('===> Processing yr ' + yr)
    yearDS = xr.open_dataset('/Volumes/crusoe-repo/data/core/' + '202501snap_yr' + str(yr) + '_core_profiles_20250207.nc')
    temp = yearDS.to_dataframe()

    core_dict = {key:mod_argo.process_argo_float(group, bgc_list = []) 
                                    for key,group in temp.groupby('PLATFORM_NUMBER')}
    coreDF = pd.concat(core_dict.values(), ignore_index=True).set_index(['profid', 'pressure'])
    
    
    temp_qcs = ['temperature_qc', 'salinity_qc', 'pressure_qc', 'time_qc', 'position_qc'] # Do not drop oxygen_qc, only for diagnostic
    coreDF_QC = mod_argo.filter_qc_flags(coreDF, qc_vars=temp_qcs, use_flags=['1', '2', '5', '8'])

    if save:
        savetag = 'core_QCflags1258_yr'+ yr + '_acc' + datetag + '.csv'
        coreDF_QC.to_csv(savepath + savetag)
        print('Saved to ' + savepath + savetag + '\n')
    return yr, coreDF_QC

# coreDF_QC = dict(Parallel(n_jobs=3)(
#                                     delayed(process_argo_year)(yr, save=save) 
#                                     for yr in yrlist))


# %% (opt2) Sequential -- Convert to float dataframes, run QC filters

save=True
savepath = '/Volumes/crusoe-repo/data/core/L1-qc/'
datetag = datetime.datetime.now().strftime('%Y%m%d')

# ===== Initialize if running for first time

coreDF = {k:None for k in annual_core_profiles.keys()}
coreDF_QC = {k:None for k in annual_core_profiles.keys()}
coreDF_regular = {k:None for k in annual_core_profiles.keys()}

# ===== Choose years to iterate over

# yrlist = [str(x) for x in range(2014,2024)]
# yrlist = ['2019', '2020', '2022', '2023']
yrlist=['2024']

# =====  * LONG RUN TIME * =======
reload(mod_argo)
for yr in tqdm(yrlist):
    print('===> Processing yr ' + yr)
    temp = annual_core_profiles[yr].to_dataframe()
    core_dict = {key:mod_argo.process_argo_float(group, bgc_list = [], 
                                        ref_time= '2014-01-01') for key,group in temp.groupby('PLATFORM_NUMBER')}
    coreDF[yr] = pd.concat(core_dict.values(), ignore_index=True).set_index(['profid', 'pressure'])

# QC step 
for yr in tqdm(yrlist):
    coreDF_QC[yr] = mod_argo.filter_qc_flags(coreDF[yr], qc_vars='all', use_flags=['1', '2', '5', '8'])

    if save:
        savetag = 'core_QCflags1258_yr'+ yr + '_acc' + datetag + '.csv'
        coreDF_QC[yr].to_csv(savepath + savetag)
        print('Saved to ' + savepath + savetag + '\n')


# %% Bathymetric masking (slow)

bathy_path = '/Volumes/crusoe-repo/data/etopo/gebco_2024_sub_ice_topo/GEBCO_2024_sub_ice_topo.nc'


# == INITIALIZE EMPTY, COMMENT OUT TO AVOID REWRITE
coreINDEX_masked = {k:None for k in coreDF_QC.keys()}
coreDF_QC_masked= {k:None for k in coreDF_QC.keys()}

# == Run and save parameters
nc_title = 'Core profiles with valid QC data, masked to open ocean bathymetry where ETOPO2 elevation < -1000m'
nc_attributes = {'title': nc_title}
droplist = ['temperature_qc', 'salinity_qc', 'pressure_qc', 
             'temp_error', 'psal_error', 'pres_error']


# save_path = None # Doesn't save if None
# save_path = '/Volumes/crusoe-repo/data/core/L1-qc/'
save_path = '/Users/sangminsong/Documents/PROJECTS-PYTHON/CRUSOE-SOCarbon/x_localdata/bgc/L2-mask/'
save_tag = 'core_open_ocean_INDEX_'
yrlist = ['2024'] # ['2019', '2020', '2021', '2022', '2023']

# == LONG RUN TIME: 
for yr in tqdm(yrlist):
        print('===> Processing yr ' + yr + '\n')

        coreINDEX_masked[yr] = mod_ocean.get_bathymetry_DS(coreDF_QC[yr].groupby('profid').first(),
                                                bathy_nc = xr.open_dataset(bathy_path).elevation,
                                                dropvars = droplist,
                                                save_path = save_path, save_tag = save_tag + yr,
                                                nc_attrs = nc_attributes)
    

# CHOOSE WHETHER TO STORE BY profid OR PROFID

# Find valid profids (1) bathymetry and 2) valid data above 25 dbar)
validBathymetry_INDEX = xr.merge([coreINDEX_masked[yr] for yr in yrlist], join='outer')
temp = validBathymetry_INDEX.to_dataframe()
valid_surface_profids = temp[temp.pressure<=25].index # represents surface pressure

# Final index of surface profile locations (valid bathymetry, valid surface)
coreINDEX = validBathymetry_INDEX.sel(profid =[x for x in valid_surface_profids])
valid_profids = coreINDEX.profid.values

# Make corresponding dataframes
core_all = pd.concat([coreDF_QC[yr] for yr in yrlist])
# .set_index(['profid', 'pressure'])
core_valid = core_all.set_index('profid').loc[valid_profids]

# final.get_level_values('profid').nunique()
# print('Dropped ' + str(dropped) + ' profiles')

print('# open ocean profiles: ' + str(len(core_all.groupby('profid'))))
print('# with valid surface data (<25 dbar): ' + str(len(valid_profids)))

# %% Pressure interpolation

core_valid['linear_time'] = (core_valid['datetime'].apply(
                                        lambda x:mod_ocean.datetime2ytd(np.datetime64(x), 
                                        ref_time='2014-01-01'))
                            )
coreDF_regular_all = mod_argo.regrid_pressure_levels(core_valid ,#coreDF_QC_valid,
                                            pres_regular, bgc_list=[], 
                                            z_gap = 1000,       # 'dynamic',
                                            surface_fill=25,
                                            ref_time = '2014-01-01')
coreDF_regular_all['datetime'] = coreDF_regular_all['linear_time'].apply(lambda x: mod_ocean.ytd2datetime(x, ref_time='2014-01-01'))
print('Regridded.')

# %%   Convert back to xarray Dataset and save as netcdf

#  Make attributes for the xr dataset
datetag = str(datetime.now())[:10].replace('-',''); print(datetag)
title = 'Core float profiles with valid surface data (QC flags 1,2,5,8), interpolated to n=197 pressure levels between 0-980 dbar. Includes only profiles with valid pH observations above 25 dbar.'
pres_att = 'Pressure levels in dbar: ' + str(pres_regular)
source = 'Argopy, expert mode; GDAC snapshot Jan 2025'

nc_attrs = {'title': title, 'pressure_levels': pres_att, 'source': source, 'date': datetag}

# Make main dataset
coreDS = mod_argo.to_xr_dataset(coreDF_regular_all, nc_attrs)

coreDS
