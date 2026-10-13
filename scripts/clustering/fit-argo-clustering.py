import data_loader as loader
from importlib import reload
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from tqdm import tqdm

import matplotlib.pyplot as plt
from cmocean import cm as cmo
from cartopy import crs as ccrs

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

from data_loader import local_working_folder, root_data_folder
datetag = datetime.now().strftime('%Y%m%d')
print('Saving to ' + local_working_folder + ' with datetag ' + datetag)


### SET NOTEBOOK RUN PARAMETERS ### ===================

n_pca = 8               # PC components 
n_gmm = 6               # GMM classes 
dbar_limit = 500        # Depth limit for PCA 

# ================================================

# use_preprocessed = False # Set to True if you want to load pre-centered data 
save_run = True

pcm_params = 'pc' + str(n_pca) + '_gmm' + str(n_gmm) + '_dbar' + str(dbar_limit)
save_path = local_working_folder + 'P2-clustered/' + pcm_params + '/'
datetag = datetime.now().strftime('%Y%m%d') # For filenames

print(save_path, datetag)

# %% Import core and bgc datasets

[coreDS, coreINDEX] = loader.import_core_data(type='L3_extended') # now includes core 2024 data for potential testing
[bgcDS, bgcINDEX] = loader.import_bgc_data(type='L3_extended')


# Combine the ten-year and 2024 datasets with pressure data for pca
fullDS = xr.concat([coreDS, bgcDS], dim='profid')
_, index = np.unique(fullDS['profid'], return_index=True)
fullDS = fullDS.isel(profid=np.sort(index))

fullINDEX = pd.concat([coreINDEX.to_dataframe().reset_index(), 
                       bgcINDEX.to_dataframe().reset_index()],axis=0)
fullINDEX = (fullINDEX.dropna(subset=['profid']) #, 'pressure'])
            .set_index(['profid'])
            .to_xarray()
            .drop_duplicates('profid', keep='first'))
fullINDEX

# Make sure same profids in both datasets
shared_profids = set(fullINDEX.profid.values) & set(fullDS.profid.values)

# Finalize input index and dataset
input_INDEX = fullINDEX.sel(profid=[x for x in list(shared_profids)])
input_DS = fullDS.sel(profid=[x for x in list(shared_profids)])


# %%  Main PCA and GMM function Jul 2026 
# Updated Feb 2026  so re-index starting at 1 here instead of later in regression
print('==> Running with params ' + pcm_params + '; ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# inputDS = inputDS.drop_duplicates(dim='profid', keep='first') # not necessary anymore
# inputINDEX = coreINDEX.drop_duplicates(dim='profid', keep='first')

centered_DS = mod_pcm.center_byPressure(input_DS, vars_list= ['CT', 'SA'], with_scaling = False)

[Xtrain, Xtrans, pca_obj, PCdict, gmm, Y_gmm, class_locs, class_data] = mod_pcm.run_pca_and_gmm(
                                                                centered_DS, input_DS, input_INDEX,
                                                                n_pca=n_pca, 
                                                                n_gmm=n_gmm, 
                                                                dbar_limit=dbar_limit
                                                            )

# Calculate posterior probabilities for each class
probs, class_probs = mod_pcm.calc_postprobs(gmm, Xtrain, Xtrans, Y_gmm, class_locs)
ax = figs_pcm.sopolar_classes_paneled(class_locs, numpanels=[1,n_gmm], figsize=(10,5), dotsize=4)

# %% Create object to search classifications by profid / prof_datetag

# Modify the posterior prob dataframe to add class assignment and prof_datetag for searching (~13 min)
PCM_finder = probs.copy() # temp object to search
PCM_finder['class_assign'] = Y_gmm.loc[PCM_finder.index, 'class']
PCM_finder['prof_datetag'] = np.tile(np.nan, len(PCM_finder))
PCM_finder

# save_path = '../working-vars/regression/P2-clustered/' + pcm_params + '/'
save_run=True
if save_run:
    Y_gmm.to_csv(save_path + 'Y_gmm_' + pcm_params + '_' + datetag + '.csv') # PC components and class assignment
    PCM_finder.to_csv(save_path + 'PCM_finder_' + pcm_params + '_' + datetag + '.csv') # PC components and class assignment
    probs.to_csv(save_path + 'postprobs_' + pcm_params + '_' + datetag + '.csv')
    
    for i in range(1, n_gmm+1):
        class_probs[i].to_csv(save_path + 'class' + str(i) + '_probs_' + datetag + '.csv') # Dataframes for single class


    print('Saved run to ' + save_path + '; ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# %% BIC and AIC
[bic_mean, bic_std, aic_mean, aic_std] = mod_pcm.calc_bic_and_aic(np.asarray(Xtrans), max_N = 10, max_iter=10)

# Silhouette
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score


silhouette_avg_dict = {k:None for k in range_n_clusters}
fig = plt.figure(figsize=(10, 7))
ax = fig.gca()

for n_clusters in range_n_clusters[:1]:
    X = Xtrans_dict[n_clusters]
    cluster_labels = Y_gmm_dict[n_clusters]['class']
    # Create a subplot with 1 row and 2 columns
    # The 1st subplot is the silhouette plot
    ax.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg_dict[n_clusters] = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg_dict[n_clusters],
    )

# fig, ax = plt.subplots(3,1, figsize=(8, 5), layout='constrained')

# n = np.arange(2, 11)

# # BIC
# ax[0].plot(n, bic_mean, marker='o')
# ax[0].fill_between(
#     n,
#     bic_mean - bic_std,
#     bic_mean + bic_std,
#     alpha=0.2
# )
# ax[0].set_ylabel('BIC')
# ax[0].set_title('BIC')
# ax[0].grid(alpha=0.3)

# # AIC
# ax[1].plot(n, aic_mean, marker='o')
# ax[1].fill_between(
#     n,
#     aic_mean - aic_std,
#     aic_mean + aic_std,
#     alpha=0.2
# )
# # ax[1].set_xlabel('Number of clusters ')
# ax[1].set_ylabel('AIC')
# ax[1].set_title('AIC')
# ax[1].grid(alpha=0.3)


# ax[2].plot(n, silhouette_avg_dict.values(), marker='o')
# ax[2].set_xlabel('Number of clusters ')
# ax[2].set_ylabel('Silhouette score')

