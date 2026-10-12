import xarray as xr
import pandas as pd
import numpy as np

from datetime import datetime
import os

root_data_folder = '/Volumes/crusoe-repo/data/' # large files, but permanent data, all publicly available
local_data_folder = '/Users/sangminsong/Documents/PROJECTS-PYTHON/CRUSOE-SOCarbon/x_localdata/' 
local_working_folder = '/Users/sangminsong/Documents/PROJECTS-PYTHON/CRUSOE-SOCarbon/x_localdata/working-vars/'

# %% 
def import_p2_clustered(type = 'pcm_probs', pcm_params='pc8_gmm6_dbar500', match_profids=True): # , datetag = '20260327'
    """ 
    Updated clustering after preprocessing Feb 11 2026
    @return         PCM_finder: dataframe with profid, class probabilities, assignment, prof_datetag
    """
    filepath = local_working_folder + 'P2-clustered/' + pcm_params + '/'
    
    if type == 'pcm_probs': #output from 2.1_PCM
        if pcm_params =='pc8_gmm6_dbar500': use_datetag = '20260916'
        if pcm_params =='pc8_gmm6_dbar900': use_datetag = '20260624'

        print('Returning [PCM_components, PCM_finder (probabilities)] for params ' + pcm_params + ' and datetag ' + use_datetag + '.')
        PCM_components = pd.read_csv(filepath + 'Y_gmm_' + pcm_params + '_' + use_datetag + '.csv', index_col=0) # Results of GMM
        PCM_finder = pd.read_csv(filepath + 'PCM_finder_' + pcm_params + '_' + use_datetag + '.csv', index_col=0) # Results of GMM
        # allprobs = pd.read_csv(filepath + 'postprobs_' + pcm_params + '_' + datetag + '.csv')  # already reindexed at 1
        
        output = [PCM_components, PCM_finder]

    elif type == 'class-gapfilled': #output from 2.2_PCM
        if pcm_params =='pc8_gmm6_dbar500': 
            socat_datetag = '20260916' # '20260715' # '20260622'
            bgc_datetag = '20260928' #  0921 has no seaice7d or deltapco2. '20260715' # '20260622'
            core_datetag = '20260916' # '20260715' 
        if pcm_params =='pc8_gmm6_dbar900': use_datetag = '20260715' #'20260624'
        print('Importing processed Dataframes with gap-filled classes for params ' + pcm_params + ' and bgc datetag ' + bgc_datetag + '.')

        bgcArgo_trainval = pd.read_csv(filepath + 'P2_bgcArgo_trainval_class-gapfilled_' + pcm_params + '_acc' + bgc_datetag + '.csv', index_col=0)
        bgcArgo_test = pd.read_csv(filepath + 'P2_bgcArgo_test_class-gapfilled_' + pcm_params + '_acc' + bgc_datetag + '.csv', index_col=0)
        
        socat_trainval = pd.read_csv(filepath + 'P2_socatv2025_trainval_class-gapfilled_' + pcm_params + '_acc' + socat_datetag + '.csv', index_col=0)
        socat_test = pd.read_csv(filepath + 'P2_socatv2025_test_class-gapfilled_' + pcm_params + '_acc' + socat_datetag + '.csv', index_col=0)

        # current version doesn't have ADT or winds 
        coreArgo_application = pd.read_csv(filepath + 'P2_coreArgo_application_class-gapfilled_' + pcm_params + '_acc' + core_datetag + '.csv', index_col=0)

        if match_profids: 
            # The bgcArgo core profiles are still in coreArgo_application, but need to exclude ones that have associated bgc obs
            shared_profids = set(bgcArgo_trainval.index).intersection(set(coreArgo_application.index))
            core_only_profids = set(coreArgo_application.index) - set(shared_profids) # profids without associated 
            coreArgo_application = coreArgo_application.loc[[*core_only_profids]]

        output = [bgcArgo_trainval, bgcArgo_test, socat_trainval, socat_test, coreArgo_application]
        print('Returned [bgcArgo_trainval(2014-2023), bgcArgo_test(2024), ' \
                    'socat_trainval(2014-2023), socat_test(2024), coreArgo_application(2014-2023)]' \
                    ' for params ' + pcm_params + ' and bgc datetag ' + bgc_datetag + '.')

        # for df in output:
        #     df['datetime'] = pd.to_datetime(df['datetime'])
            # df.loc[df.sea_ice.isna(), 'sea_ice'] = 0
            # print('Filled nan sea ice with 0s.')
 
    return output  # [bgcArgo_trainval, bgcArgo_test, socat_trainval, socat_test, coreArgo_application]
