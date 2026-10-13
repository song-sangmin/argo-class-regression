from argopy import DataFetcher  # This is the class to work with Argo data
from argopy import ArgoIndex  #  This is the class to work with Argo index
from argopy import ArgoNVSReferenceTables  # This is the class to retrieve data from Argo reference tables
from argopy import ArgoColors  # This is a class with usefull pre-defined colors
from argopy.plot import scatter_map, scatter_plot  # This is a function to easily make maps 

# Make a fresh start
import argopy
argopy.reset_options()
argopy.clear_cache()
argopy.set_options(cachedir='cache_bgc')

import logging
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("pyproj").setLevel(logging.ERROR)
logging.getLogger("fsspec").setLevel(logging.ERROR)
logging.getLogger("parso").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
DEBUGFORMATTER = '%(asctime)s [%(levelname)s] [%(name)s] %(filename)s:%(lineno)d: %(message)s'
logging.basicConfig(
    level=logging.DEBUG,
    format=DEBUGFORMATTER,
    datefmt='%I:%M:%S %p',
    handlers=[logging.FileHandler("nb-docs.log", mode='w')]

# Settings for data provider

argopy.set_options(src='erddap') # Ifremer database, updated daily
# argopy.set_options(src='gdac', gdac= 'https://...')

# To use local copy of the GDAC
# Using the January 2025 snapshot
# gdacroot = '/Volumes/ocean-repo/202501-ArgoData' 
# argopy.set_options(src='gdac', gdac = gdacroot)

#%% MAIN CORE FLOATS: Running for multiple years 
save = True
# save = False

yrlist=['2024']

for yrtag in yrlist:
    savetag = yrtag + '_core'
    print('processing ' + savetag)

    # Format: [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max, datim_min, datim_max]
    BOX = [-180, 180, -90, -35, 0, 2000, yrtag + '-01-01', yrtag + '-12-31']

    # Construct fetchers for core floats first
    # Expert mode returns all QC flags
    # Updated Feb 4 2025, use local copy of GDAC (Snapshot 01-09)
    fetcher = DataFetcher(ds='phy', mode='expert', params='all',
                    parallel=True, progress=True,
                    chunks_maxsize={'time': 30},
                )
    core_fetcher = fetcher.region(BOX).load()

    # Returns xr Dataset with point observations as dimension
    core_profiles = core_fetcher.data.argo.point2profile();
    core_profiles = core_profiles.assign_attrs(raw_attrs = '', Fetched_uri='') # simplify for save
    # Store index and domain
    core_index = core_fetcher.index
    core_domain = core_fetcher.domain 

    if save:
        core_profiles.to_netcdf(savepath + '202501snap_yr' + savetag + '_profiles_' + datetag + '.nc')
        core_index.to_csv(savepath + '202501snap_yr' + savetag + '_index_' + datetag + '.csv')
        print('Saved data to netcdf and csv')
        print(savepath + '202501snap_yr' + savetag + '_profiles_' + datetag + '.nc')

# %% # MAIN METHOD, bgc FLOATS: Running for multiple years 
# ~40 min per year 
save = True
# save = False
yrlist=['2024']

for yrtag in yrlist:
    savetag = yrtag + '_bgc'
    # Format: [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max, datim_min, datim_max]
    BOX = [-180, 180, -90, -35, 0, 2000, yrtag + '-01-01', yrtag + '-12-31']

    # Set which BGC variables are required for a float to be fetched
    # required_bgc = ['PH_IN_SITU_TOTAL']
    # fetcher = DataFetcher(ds='bgc', mode='expert', params='all',
    #                     measured = required_bgc,
    #                     parallel=True, progress=True,
    #                     chunks_maxsize={'time': 30},
    #                     )
    print('===> Processing ' + savetag)
    fetcher = DataFetcher(ds='bgc', mode='expert', params='all',
                    parallel=True, progress=True,
                    chunks_maxsize={'time': 30},
                )
    bgc_fetcher = fetcher.region(BOX).load()

    # Returns xr Dataset with point observations as dimension
    bgc_profiles = bgc_fetcher.data.argo.point2profile();
    bgc_profiles = bgc_profiles.assign_attrs(raw_attrs = '', Fetched_uri='', 
                                             Fetched_constraints=str(bgc_profiles.Fetched_constraints).replace('/','_')) # simplify for save
    # Store index and domain
    bgc_index = bgc_fetcher.index
    bgc_domain = bgc_fetcher.domain 

    if save:
        print('===> Saving data and index for yr ' + yrtag)
        bgc_profiles.to_netcdf(savepath + 'ERDDAP_yr' + savetag + '_profiles_' + datetag + '.nc')
        bgc_index.to_csv(savepath + 'ERDDAP_yr' + savetag + '_index_' + datetag + '.csv')
        print('Successfully saved to ' + savepath + 'ERDDAP_yr' + savetag + '_profiles_' + datetag + '.nc') 