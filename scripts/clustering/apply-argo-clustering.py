# ====== New version from feb 11 with import p2 data
[bgcArgo_trainval, bgcArgo_test, socat_trainval, socat_test, coreArgo_application] = loader.import_p1_processed()

[PCM_components, PCM_finder] = loader.import_p2_clustered(type='pcm_probs', pcm_params = pcm_params)


# Add class assignments (will not drop nans yet)
floatDF = mod_pcm.add_class_assignments(bgcArgo_trainval, PCM_finder, n_gmm, profid_col='profid')
floatDF['prof_datetag'] = floatDF['prof_datetag'].astype(str)
floatDF_filled = mod_pcm.fill_missing_float_classes(floatDF.set_index('profid'), n_gmm=n_gmm)

testDF = mod_pcm.add_class_assignments(bgcArgo_test, PCM_finder, n_gmm, profid_col='profid')
testDF['prof_datetag'] = testDF['prof_datetag'].astype(str)
testDF_filled = mod_pcm.fill_missing_float_classes(testDF.set_index('profid'), n_gmm=n_gmm)

coreDF = mod_pcm.add_class_assignments(coreDF.reset_index(), PCM_finder, n_gmm, profid_col='profid')
coreDF = (mod_prep.index_by_datetag(coreDF, type='dataframe')
                .reset_index().set_index('profid') # add prof_datetag as column so you can fill in 
)
coreDF_filled = mod_pcm.fill_missing_float_classes(coreDF, n_gmm=n_gmm)


# %% Colocation for SOCAT

# Rerun colocation
fullINDEX = xr.open_dataset('/Users/sangminsong/Documents/PROJECTS-PYTHON/CRUSOE-SOCarbon/x_localdata/working-vars/P2-clustered/pc8_gmm6_dbar500/' +
                            'inputINDEX_coreBGC_2014–2024_acc' + '20260916' + '.nc')

import crusoe.oceantools as mod_ocean
fullINDEX["datetime"] = fullINDEX.datetime.astype('datetime64[ns]')
fullINDEX['linear_time'] = mod_ocean.datetime2ytd(fullINDEX['datetime'], ref_time='2014-01-01')

socat_test['linear_time'] = mod_ocean.datetime2ytd(socat_test['datetime'], ref_time='2014-01-01')
socat_trainval['linear_time'] = mod_ocean.datetime2ytd(socat_trainval['datetime'], ref_time='2014-01-01')


import crusoe.argotools as mod_argo

socat_test_colocated = mod_argo.find_nearest_float(socat_test, fullINDEX, yd_window=7, ref_time='2014-01-01')
socat_trainval_colocated = mod_argo.find_nearest_float(socat_trainval, fullINDEX, yd_window=7, ref_time='2014-01-01')

shipDF = mod_pcm.add_class_assignments(socat_trainval, PCM_finder, n_gmm, profid_col='nearest_profid')
shipDF['sample_datetag'] = shipDF['cruiseid'] + '_' + shipDF['datetime'].dt.strftime('%Y%m%d%H')
shipDF_filled = mod_pcm.fill_missing_ship_classes(shipDF.set_index('sample_datetag'), n_gmm)


shipDF_test = mod_pcm.add_class_assignments(socat_test, PCM_finder, n_gmm, profid_col='nearest_profid')
shipDF_test['sample_datetag'] = shipDF_test['cruiseid'] + '_' + shipDF_test['datetime'].dt.strftime('%Y%m%d%H')
shipDF_test_filled = mod_pcm.fill_missing_ship_classes(shipDF_test.set_index('sample_datetag'), n_gmm)

# %% optional save

save = True
if save:
    floatDF_filled.to_csv(save_path + 'P2_bgcArgo_trainval_class-gapfilled_' + pcm_params + '_acc' + datetag + '.csv')
    testDF_filled.to_csv(save_path + 'P2_bgcArgo_test_class-gapfilled_' + pcm_params + '_acc' + datetag + '.csv')
    # coreDF_filled.to_csv(save_path + 'P2_coreArgo_application_class-gapfilled_' + pcm_params + '_acc' + datetag + '.csv')
    # shipDF_filled.to_csv(save_path + 'P2_socat-trainval_class-gapfilled_' + pcm_params + '_acc' + datetag + '.csv')

    print('Saved floatDF_filled to ' + save_path + 'P2_bgcArgo_trainval_class-gapfilled_' + pcm_params + '_acc' + datetag + '.csv')
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    