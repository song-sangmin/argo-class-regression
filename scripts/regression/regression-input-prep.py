import data_loader as loader
from importlib import reload
import numpy as np
import pandas as pd
import xarray as xr
import optuna
from datetime import datetime
from tqdm import tqdm
import gsw

import matplotlib.pyplot as plt
from cmocean import cm as cmo
from cartopy import crs as ccrs
from __future__ import annotations


# From internal modules
from crusoe import pcm as mod_pcm
from crusoe import regression as mod_reg
from crusoe import preprocessing as mod_prep
from crusoe import crossvalidation as mod_cv
# from crusoe import argotools as mod_argo
# from crusoe import oceantools as mod_ocn

from crplot import figtools as mod_plot
from crplot import southpolar as sopo
from crplot import figs_pcm

datetag = datetime.now().strftime('%Y%m%d')


# Create dataframe to save and open in Perlmutter

# %% ==== Choose data run parameters
float_carbon_var= 'pHbias5_pK1'
pcm_params='pc8_gmm6_dbar500'
n_gmm = int(pcm_params.split('_')[1][-1]) 

desc = 'floatparam_' + float_carbon_var + '_' + pcm_params
print(desc + '; ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# ===== Import clustered data 
# datetag: 20260211 without 2024 data clustered; 20260327 with 2024 data clustered

# pc8_gmm6_dbar500
[bgcArgo_trainval, bgcArgo_test, 
        socat_trainval, socat_test, 
        coreArgo_application] = loader.import_p2_clustered(type = 'class-gapfilled', 
                                                                pcm_params=pcm_params)

print(len(bgcArgo_trainval), len(bgcArgo_test), len(socat_trainval), 
      len(socat_test), len(coreArgo_application))


# %% Define which clusters have sea ice profiles ==> include only in clusters 2, 3 
sea_ice_clusters = []

print('ice', '\t', 'cluster', '\t', 'proportion')
for ncluster in range(1, n_clusters+1):
    cluster_data = bgcArgo_trainval[bgcArgo_trainval.class_assign==ncluster]
    ice_profiles = cluster_data[cluster_data.sea_ice>0.1]
    print(len(ice_profiles), '\t', len(cluster_data),  '\t\t', len(ice_profiles)/len(cluster_data))
    if len(ice_profiles)>10: sea_ice_clusters.append(ncluster)

print('Sea ice clusters: ', sea_ice_clusters)


# Counts by platform and cluster
print('class', '\t', '#obs', '\t', 'float_obs', '\t', 'socat_obs')
combined_trainval = pd.concat([bgcArgo_trainval, socat_trainval])

for ncluster in range(1, n_clusters+1):
    cluster_data = combined_trainval[combined_trainval.class_assign==ncluster]
    float_data = bgcArgo_trainval[bgcArgo_trainval.class_assign==ncluster]
    ship_data = socat_trainval[socat_trainval.class_assign==ncluster]
    print(ncluster, '\t', len(cluster_data),  '\t', len(float_data), '\t\t', len(ship_data))

# %% Save combined trainval dataframe


# Initailize main dictionary to hold nestedCV containers for each class
# Updated Jun 10 to use methods 
soclass_list = ['cluster' + str(k) for k in range(1, n_clusters+1)]
n_outer = 5
n_inner = 2

feat_list = ['CT', 'SA', 'log_mld', 'adt', 'sea_ice', 'sea_ice_avg_7d', 'wind_speed', 'atmos_pres_atm', 'max_solar_rad']
combined_trainval = (mod_cv.align_platform_indexers([bgcArgo_trainval, socat_trainval], 
                                                    ['wmoid', 'cruiseid']))


combined_trainval = combined_trainval.dropna(subset=feat_list).drop_duplicates()
combined_trainval['datetime'] = pd.to_datetime(combined_trainval['datetime'])
# combined_trainval = combined_trainval.drop(columns=['wind_speed_avg_12h', 'wind_speed_avg_24h',
#        'wind_speed_avg_72h', 'sea_ice_avg_14d'])
combined_trainval['prof_datetag'] = combined_trainval['prof_datetag'].astype(str)
combined_trainval['prof_datetime'] = combined_trainval['prof_datetime'].astype('datetime64[ns]')
combined_trainval['nearest_profid'] = combined_trainval['nearest_profid'].astype(str)
save_cols = (['platform', 'platform_id', 
                'datetime', 'year', 'month', 'day', 'linear_time', 
                'platform_indexer', 
                'latitude', 'longitude', 
                'CT', 'SA',  'mld', 'log_mld', 'adt', 
                'sea_ice', 'sea_ice_avg_7d', 
                'wind_speed', 
                'max_solar_rad',
                'atmos_pres_Pa', 'atmos_pres_atm', 'atmos_co2_ppm', 'vapor_pres_atm',
                'pco2_ocean', 'pco2_atmos',
                'cluster',
                'cluster1_prob', 'cluster2_prob', 'cluster3_prob', 'cluster4_prob',
                'cluster5_prob', 'cluster6_prob', 'delta_pco2'])


from data_loader import local_working_folder
combined_trainval[save_cols].to_csv(local_working_folder + 'combined_trainval_acc20260928_seaiceclusters35.csv')
