from argoclare import crossvalidation
from argoclare import regression

import dask
from dask.distributed import LocalCluster, Client, wait
import os
from datetime import datetime
import pandas as pd
import numpy as np
from itertools import product

# %% Fold functions
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone

from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor

def run_dask_nested_cv(client, trainval_data, n_clusters, sea_ice_clusters,
                             feat_list, var_predict,
                             n_outer, n_inner, 
                            training_algorithm, tuning_grid,
                            cv_type, scoring_metric,
                            outpath, outfile,
                            seed=None):
    """ Main dask implementation 
    Inner fold version of run_dask_crossvalidation 
    
    :param trainval_data: pd Dataframe with combined training data, column "platform_id"
    :param n_clusters: # of Southern Ocean clusters
    :param sea_ice_clusters: which clusters should consider sea ice as a feature
    :param feat_list: feature list to train on 
    :param var_predict: variable name to predict (str)
    :param n_outer: # of outer folds to iterate over
    :param n_inner: # of inner folds to iterate over
    :param training_algorithm: (str) choose 'RFR', 'ERT', 'XGB'
    :param tuning_grid: dictionary of hyperparameter choices to test
    :param cv_type: choose "platform" or "kmeans" 
    :param seed: int seed for platform fold splits (default None, not reproducible)
    
    
    """

    start_time = datetime.now().strftime('%Y%m%d %H:%M:%S')
    ###################
    # 0. DATA SETUP 
    ###################

    print('Setting up data containers...')
    # Set up data containers for each cluster
    # soclass_list = ['cluster' + str(k) for k in range(1, n_clusters+1)]
    class_tags = ['cluster' + str(k) for k in range(1, n_clusters+1)]
    soclass_cvtainers = {soclass: None for soclass in class_tags} # used to be called soclass_cvtainers
    for classnum in range(1, n_clusters+1):
        outer_nest = (crossvalidation.setup_cv_container(
                                trainval_data[trainval_data.cluster==classnum],
                                n_outer = n_outer,
                                n_inner = n_inner,
                                spatial_cv = cv_type,
                                indexer = 'platform_id',
                                random_state = seed))
        soclass_cvtainers['cluster' + str(classnum)] = outer_nest
    cvtainers_future = client.scatter(soclass_cvtainers,broadcast=True)


    parameter_combinations = make_parameter_combinations(tuning_grid) 

    # Add more options as needed
    if training_algorithm == 'RFR':
        use_regressor = RandomForestRegressor(random_state=24, n_jobs=1) 
    elif training_algorithm == 'XGB':
        use_regressor = XGBRegressor(random_state=24, n_jobs=1)
    elif training_algorithm == 'ERT':
        use_regressor = ExtraTreesRegressor(random_state=24, n_jobs=1)

    ###################
    # 1. INNER CV 
    ###################

    print('Building dask tasks for inner CV...')   
    inner_futures = []

    for outer_foldnum in range(1, n_outer + 1): 
        for classnum in range(1, n_clusters+1):
            # Get cluster data within this outer fold
            cluster_data = soclass_cvtainers['cluster' + str(classnum)] # nestedCV object
            
            if classnum not in sea_ice_clusters: 
                class_feat_list = [x for x in feat_list if x != 'sea_ice']
            else: class_feat_list = feat_list.copy()

            # Run inner folds and store mean score for each parameter set
            for param_set in parameter_combinations:
                inner_future = client.submit(
                    fit_inner_folds, # returns pd Series
                    param_set,
                    'fold' + str(outer_foldnum), #outer_foldtag,
                    cvtainers_future,
                    'cluster' + str(classnum),
                    use_regressor,
                    class_feat_list,
                    var_predict,
                    scoring_metric)

                inner_futures.append(inner_future)
    
    # Compile inner results (each for one parameter set, one outer fold, one cluster)
    inner_results = client.gather(inner_futures)
    print('Dask computation for inner CV complete.')

    ###################
    # 2. PARAMETER SELECTION 
    ###################
    print('Aggregating results for each outer fold and class...')
    
    inner_summary = pd.DataFrame(inner_results)
    # inner_summary = inner_summary.set_index(['outer_foldtag', 'cluster'])

    best_idx = (inner_summary
                    .groupby(['outer_foldtag', 'cluster'])['mean_score']
                    .idxmin())
    foldxclass_best_params =  (inner_summary.loc[best_idx]
                                .set_index(['outer_foldtag', 'cluster']))[['params']]
    
    print('Selected best parameters for each outer fold and class.')
    
    ###################
    # 3. OUTER CV - FIT CLASS/FOLD MODELS
    ###################

    print('Building dask tasks for outer CV using tuned parameters...')
    outer_futures = []

    for outer_foldnum in range(1, n_outer + 1):
        for classnum in range(1, n_clusters + 1):
            fctag = ('fold' + str(outer_foldnum), 'cluster' + str(classnum))
            # tuned_params = foldxclass_best_params[fctag]
            tuned_params = foldxclass_best_params.loc[fctag,'params']

            if classnum not in sea_ice_clusters: 
                class_feat_list = [ x for x in feat_list if x != 'sea_ice' ] 
            else: class_feat_list = feat_list.copy()

            outer_future = client.submit(
                    fit_outer_class_model, # returns a dict with outer_foldtag, cluster, model, tuned_params
                    'fold' + str(outer_foldnum), 
                    tuned_params,
                    cvtainers_future,
                    'cluster' + str(classnum),
                    use_regressor,
                    class_feat_list,
                    var_predict)
            
            outer_futures.append(outer_future)
        
    outer_results = client.gather(outer_futures)
    print('Dask computation for fitting outer fold models complete.')

    ###################
    # 4. BUILD FOLD CMV + COMPUTE OUTER VALIDATION ERRORS
    ###################
    print('Evaluating outer fold models on validation data...')
    model_finder = organize_fold_results(outer_results)
    # print(model_finder)
    ClusteredModel_OuterFoldResults = {} # ClusteredModelVersion results
    for outer_foldnum in range(1, n_outer + 1):
        outer_foldtag = 'fold' + str(outer_foldnum)
        
        fold_CMV = regression.ClusteredModelVersion(ind_list = [x for x in range(1,n_clusters+1)], 
                                                 feat_list = feat_list)
        fold_valDF = pd.DataFrame()
        
        for classtag in class_tags:
            cluster_data = soclass_cvtainers[classtag] # nestedCV object
            fold_valDF = pd.concat([fold_valDF, cluster_data.get_validation_data(outer_foldtag)], axis=0) # this fold's validation data

        for classnum in range(1,n_clusters+1):
            classtag = 'cluster' + str(classnum)
            fctag = (outer_foldtag, classtag)
            fold_CMV.models[classnum] = model_finder.loc[outer_foldtag, classtag].values[0] #Regressor object
            fold_CMV.params[classnum] = foldxclass_best_params.loc[fctag,'params']

        fold_CMV = evaluate_outer_fold(fold_CMV, fold_valDF, feat_list, var_predict, sea_ice_clusters)

        # Run linear calibration
        cal_coeffs = regression.get_clustered_calibration_coeffs(fold_CMV.weighted_validation)
        fold_CMV.cal_coeffs = cal_coeffs
        fold_CMV.weighted_validation = regression.apply_clustered_calibration(cal_coeffs, fold_CMV.weighted_validation, n_clusters)
    
        ClusteredModel_OuterFoldResults[outer_foldtag] = fold_CMV # equivalent of ClusteredModel_OuterFoldResults = dict(client.gather(futures))
    
    print('Model evaluation by outer fold complete. Collapsing results and saving results...')
    
    #############
    # SAVE RESULTS
    ##############
    print('Saving results...')

    # Store weighted validation DF with estimates on all outer folds
    ClusteredModel_ValidationErrors = crossvalidation.collapse_folds(
        ClusteredModel_OuterFoldResults) # weighted validation DF
    ClusteredModel_InnerResults = (inner_summary
                                    .drop(columns=['params'])
                                    .set_index(['outer_foldtag', 'cluster']))  # all inner fold scores (mean of 5 MAE's per param set)
    ClusteredModel_BestParams = foldxclass_best_params.copy() # best params used to train each outer fold model

    # To save:
    datetag = datetime.now().strftime('%Y%m%d')
    ClusteredModel_ValidationErrors.weighted_validation.to_csv(outpath + outfile + '_outercvErrors_acc' + datetag + '.csv')
    ClusteredModel_InnerResults.to_csv(outpath + outfile + '_innerScores_acc' + datetag + '.csv')
    ClusteredModel_BestParams.to_csv(outpath + outfile + '_bestParams_acc' + datetag + '.csv')

    end_time = datetime.now().strftime('%Y%m%d %H:%M:%S')
    with open((outpath + outfile + '_runDescription_acc' + datetag + '.txt'), "w") as file:
        file.write('RUN_START: ' + start_time + '\n')
        file.write('RUN_END: ' + end_time + '\n')
        file.write('ALGORITHM: ' + training_algorithm + '\n')
        file.write('N_OUTER: ' + str(n_outer) + '\n')
        file.write('N_INNER: ' + str(n_inner) + '\n')
        file.write('CV_TYPE: ' + cv_type + '\n')
        file.write('SEED: ' + str(seed) + '\n')
        file.write('INNER_SCORE_METRIC: ' + scoring_metric + '\n')
        file.write('TUNING_GRID: ' + '\n')
        for key, value in tuning_grid.items():
            file.write(f"{key}: {value}\n")

    print('Saved validation errors to ' + outpath + outfile + '_outercvErrors_acc' + datetag + '.csv')
    print('Saved inner fold scores to ' + outpath + outfile + '_innerScores_acc' + datetag + '.csv')
    print('Saved best params to ' + outpath + outfile + '_bestParams_acc' + datetag + '.csv')
    print('Saved run description to ' + outpath + outfile + 'runDescription_acc' + datetag + '.txt')

    del inner_futures, outer_futures, cvtainers_future
    return 



def fit_inner_folds(hyperparams_set, 
                   outer_foldtag, 
                   data_cvtainers, cluster,
                   use_regressor, 
                   class_feat_list, var_predict,
                   scoring_metric='median_abs_error'):
    """
    Iterate over inner folds given one set of parameters for one cluster and one outer fold.
    """

    cluster_data = data_cvtainers[cluster] # NestedCV object
    tuning_data = cluster_data.inner_nests[outer_foldtag] # NestedCV object

    inner_scores = []

    # Iterate over inner folds (within one cluster x outer fold) for one parameter set
    # print('Evaluating pararmeter set for Cluster ' + str(cluster) + ', outer fold ' + str(outer_foldtag) + '...')
    for inner_foldtag in tuning_data.fold_list:
        inner_train = (cluster_data.get_inner_training_data(outer_foldtag, inner_foldtag)
                            .dropna(subset=class_feat_list + [var_predict]).copy())
        inner_val = (cluster_data.get_inner_validation_data(outer_foldtag, inner_foldtag)
                            .dropna(subset=class_feat_list + [var_predict]).copy())
        
        Mdl = clone(use_regressor)
        Mdl.set_params(**hyperparams_set)
        Mdl.fit(inner_train[class_feat_list],inner_train[var_predict])

        prediction = Mdl.predict(inner_val[class_feat_list])

        # Choose scoring metric, add others as desired
        if scoring_metric=='median_abs_error':
            score = np.median(np.abs(prediction - inner_val[var_predict])) #calculate_MAE(prediction,inner_val[var_predict])
            inner_scores.append(score) # 5 medAEs for that hyperparam set.
    mean_score = np.mean(inner_scores) # mean score for that hyp set amongst inner folds

    set_results = pd.Series(
            {'outer_foldtag': outer_foldtag, 'cluster': cluster, 
            'params': hyperparams_set, 
            'mean_score': mean_score} | hyperparams_set
            )
    # return hyperparams_set, np.mean(scores)
    return set_results

# def calculate_MAE(prediction, truth):
#     """ Calculate error between prediction and truth using specified metric"""
#     return np.median(np.abs(prediction - truth))

def select_best_params(list_inner_results):
    """ Return best hyperparameter set from inner fold tuning (as dict)
    For each clusterxfold.
    :param list_inner_results: list of pd.Series objects"""
    all_sets = pd.concat(list_inner_results, axis=1).T
    best_params = all_sets.loc[all_sets['mean_score'].idxmin(), 'params']
    return best_params

def fit_outer_class_model(outer_foldtag, 
                    tuned_params,
                    data_cvtainers, classtag,
                    use_regressor,
                    class_feat_list,
                    var_predict):
    """ Fit a class-dependent model for one outer fold using best hyperparameters from inner folds"""
    cluster_data = data_cvtainers[classtag]
    outer_train = (cluster_data.get_training_data(outer_foldtag)
                              .dropna(subset=class_feat_list + [var_predict])
                              .copy())
    Mdl = clone(use_regressor)
    Mdl.set_params(**tuned_params)
    Mdl.fit(outer_train[class_feat_list], outer_train[var_predict])

    return {'outer_foldtag': outer_foldtag, 'cluster': classtag, 'model': Mdl, 'tuned_params': tuned_params}

def evaluate_outer_fold(fold_CMV, fold_valDF,
                        feat_list,
                        var_predict,
                        sea_ice_clusters):
    temp_feat_list = [x for x in feat_list if x != 'sea_ice']
    fold_valDF = fold_valDF.dropna(subset=(temp_feat_list + [var_predict])).copy()
    fold_CMV.weighted_validation = regression.apply_cluster_RFR(feat_list, fold_CMV, fold_valDF, sample_tag='val', 
                                                             sea_ice_clusters=sea_ice_clusters,
                                                             target_var = var_predict)
    
    return fold_CMV

# def organize


def organize_fold_results(list_outer_results):
    """
    Organize outer fold results into a pd Dataframe with MultiIndex 
    for cluster and outer_foldtag, and columns for model and parameters.

    Access a RandomForestRegressor object with something like: model_finder.loc['cluster2', 'fold1'].values[0]
    """
    # Create a list of tuples for the MultiIndex
    index_tuples = [(res['outer_foldtag'], res['cluster']) for res in list_outer_results]
    multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['outer_foldtag','cluster'])

    # Create a DataFrame with the MultiIndex and model column
    model_finder = pd.DataFrame({'model': [res['model'] for res in list_outer_results]}, index=multi_index)

    return model_finder


    # print('break')
    # Calibrate each fold model
    # cal_coeffs = mod_reg.get_clustered_calibration_coeffs(fold_CMV.weighted_validation)
    # fold_CMV.weighted_validation = mod_reg.apply_clustered_calibration(cal_coeffs, fold_CMV.weighted_validation, len(soclass_list))

def make_parameter_combinations(tuning_grid): 
    """ Convert tuning grid to a list of parameter combinations (as dicts) for use in inner fold evaluation""" 
    keys = [x for x in tuning_grid.keys()]
    parameter_combinations = [dict(zip(keys, values)) for values in product( *(tuning_grid[key] for key in keys) ) ] 
    return parameter_combinations

