from argoclare import crossvalidation
from argoclare import regression
import procedures_cv

from datetime import datetime
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.base import clone
from xgboost import XGBRegressor


def run_model_application(client, trainval_data, application_data,
                          n_clusters, sea_ice_clusters,
                          feat_list, var_predict,
                          n_folds,
                          cluster_models,
                          cv_type, scoring_metric,
                          outpath, outfile,
                          save_models=False,
                          cluster_models_source=None):
    """ Main dask implementation
    Train one final clustered model on trainval_data and apply it to application_data.

    For each cluster: tune hyperparameters on n_folds (single-level CV), fit out-of-fold
    models with the best parameters to get calibration coefficients, then fit a final
    class-dependent model on all of that cluster's data.

    :param trainval_data: pd Dataframe with combined training data, column "platform_id"
    :param application_data: pd Dataframe to predict on (same features + cluster{k}_prob columns, no target needed)
    :param n_clusters: # of Southern Ocean clusters
    :param sea_ice_clusters: which clusters should consider sea ice as a feature
    :param feat_list: feature list to train on
    :param var_predict: variable name to predict (str)
    :param n_folds: # of folds for hyperparameter tuning
    :param cluster_models: dict of {classnum: (training_algorithm, tuning_grid)},
                           training_algorithm is 'RFR', 'ERT' or 'XGB', tuning_grid is a dict
    :param cv_type: "platform" (folds of whole floats/cruises, platform_id), "profile" (whole profiles,
                    profid column; crossvalidation platform splits by profid) or "kmeans"
    :param save_models: pickle the fitted final models with joblib (default False)
    :param cluster_models_source: (str) where cluster_models came from, written to the run description

    :return: application_data with columns cluster{k}_pred, weighted_pred, cluster{k}_calpred, lincal_pred
    """

    start_time = datetime.now().strftime('%Y%m%d %H:%M:%S')
    ###################
    # 0. DATA SETUP
    ###################

    print('Setting up data containers...')
    class_tags = ['cluster' + str(k) for k in range(1, n_clusters+1)]
    soclass_cvtainers = {soclass: None for soclass in class_tags}
    for classnum in range(1, n_clusters+1):
        soclass_cvtainers['cluster' + str(classnum)] = crossvalidation.setup_cv_container(
                                trainval_data[trainval_data.cluster==classnum],
                                n_outer = n_folds,
                                n_inner = None,
                                spatial_cv = 'platform' if cv_type == 'profile' else cv_type,
                                indexer = 'profid' if cv_type == 'profile' else 'platform_id')
    cvtainers_future = client.scatter(soclass_cvtainers, broadcast=True)

    # Regressor, parameter combinations and features for each cluster
    class_regressors = {}
    class_param_combinations = {}
    class_feat_lists = {}
    for classnum in range(1, n_clusters+1):
        training_algorithm, tuning_grid = cluster_models[classnum]
        class_regressors[classnum] = get_regressor(training_algorithm)
        class_param_combinations[classnum] = procedures_cv.make_parameter_combinations(tuning_grid)
        if classnum not in sea_ice_clusters:
            class_feat_lists[classnum] = [x for x in feat_list if x != 'sea_ice']
        else: class_feat_lists[classnum] = feat_list.copy()

    ###################
    # 1. HYPERPARAMETER TUNING
    ###################

    print('Building dask tasks for hyperparameter tuning...')
    tuning_futures = []

    for classnum in range(1, n_clusters+1):
        for param_set in class_param_combinations[classnum]:
            tuning_future = client.submit(
                fit_tuning_folds, # returns pd Series
                param_set,
                cvtainers_future,
                'cluster' + str(classnum),
                class_regressors[classnum],
                class_feat_lists[classnum],
                var_predict,
                scoring_metric)

            tuning_futures.append(tuning_future)

    tuning_results = client.gather(tuning_futures)
    print('Dask computation for hyperparameter tuning complete.')

    ###################
    # 2. PARAMETER SELECTION
    ###################
    print('Selecting best parameters for each class...')

    tuning_summary = pd.DataFrame(tuning_results)
    best_idx = tuning_summary.groupby('cluster')['mean_score'].idxmin()
    class_best_params = tuning_summary.loc[best_idx].set_index('cluster')[['params', 'mean_score']]

    ###################
    # 3. FIT OUT-OF-FOLD MODELS (CALIBRATION) + FINAL MODELS
    ###################

    print('Building dask tasks for out-of-fold and final models using tuned parameters...')
    fold_futures = []
    final_futures = []

    for classnum in range(1, n_clusters + 1):
        classtag = 'cluster' + str(classnum)
        tuned_params = class_best_params.loc[classtag, 'params']

        for foldnum in range(1, n_folds + 1):
            fold_future = client.submit(
                    procedures_cv.fit_outer_class_model, # returns a dict with outer_foldtag, cluster, model, tuned_params
                    'fold' + str(foldnum),
                    tuned_params,
                    cvtainers_future,
                    classtag,
                    class_regressors[classnum],
                    class_feat_lists[classnum],
                    var_predict)
            fold_futures.append(fold_future)

        final_future = client.submit(
                fit_final_class_model, # returns a dict with cluster, model, tuned_params
                tuned_params,
                cvtainers_future,
                classtag,
                class_regressors[classnum],
                class_feat_lists[classnum],
                var_predict)
        final_futures.append(final_future)

    fold_results = client.gather(fold_futures)
    final_results = client.gather(final_futures)
    print('Dask computation for fitting out-of-fold and final models complete.')

    ###################
    # 4. CALIBRATION COEFFICIENTS FROM OUT-OF-FOLD PREDICTIONS
    ###################
    print('Computing calibration coefficients from out-of-fold predictions...')
    model_finder = procedures_cv.organize_fold_results(fold_results)
    ClusteredModel_FoldResults = {}
    for foldnum in range(1, n_folds + 1):
        foldtag = 'fold' + str(foldnum)

        fold_CMV = regression.ClusteredModelVersion(ind_list = [x for x in range(1,n_clusters+1)],
                                                 feat_list = feat_list)
        fold_valDF = pd.concat([soclass_cvtainers[classtag].get_validation_data(foldtag)
                                for classtag in class_tags], axis=0)

        for classnum in range(1,n_clusters+1):
            classtag = 'cluster' + str(classnum)
            fold_CMV.models[classnum] = model_finder.loc[foldtag, classtag].values[0]
            fold_CMV.params[classnum] = class_best_params.loc[classtag, 'params']

        ClusteredModel_FoldResults[foldtag] = procedures_cv.evaluate_outer_fold(
                                                    fold_CMV, fold_valDF, feat_list, var_predict, sea_ice_clusters)

    oof_predictions = crossvalidation.collapse_folds(ClusteredModel_FoldResults).weighted_validation
    cal_coeffs = regression.get_clustered_calibration_coeffs(oof_predictions, target_var=var_predict)

    ###################
    # 5. APPLY FINAL MODEL
    ###################
    print('Applying final clustered model to application data...')
    final_CMV = regression.ClusteredModelVersion(ind_list = [x for x in range(1,n_clusters+1)],
                                                 feat_list = feat_list)
    for res in final_results:
        classnum = int(res['cluster'].replace('cluster', ''))
        final_CMV.models[classnum] = res['model']
        final_CMV.params[classnum] = res['tuned_params']
    final_CMV.cal_coeffs = cal_coeffs

    application_data = apply_final_model(final_CMV, application_data, feat_list, var_predict,
                                         sea_ice_clusters, n_clusters)

    #############
    # SAVE RESULTS
    ##############
    print('Saving results...')

    ClusteredModel_TuningResults = (tuning_summary
                                    .drop(columns=['params'])
                                    .set_index('cluster'))  # mean fold score per param set
    ClusteredModel_BestParams = class_best_params.copy()
    ClusteredModel_BestParams.insert(0, 'algorithm',
                                     [cluster_models[int(x.replace('cluster', ''))][0] for x in ClusteredModel_BestParams.index])
    ClusteredModel_CalCoeffs = pd.DataFrame.from_dict(
                                    {'cluster' + str(k): v for k, v in sorted(cal_coeffs.items())},
                                    orient='index', columns=['slope', 'intercept'])
    ClusteredModel_CalCoeffs.index.name = 'cluster'

    datetag = datetime.now().strftime('%Y%m%d')
    savefiles = {'predictions': outpath + outfile + '_predictions_acc' + datetag + '.csv',
                 'tuning scores': outpath + outfile + '_tuningScores_acc' + datetag + '.csv',
                 'best params': outpath + outfile + '_bestParams_acc' + datetag + '.csv',
                 'calibration coefficients': outpath + outfile + '_calCoeffs_acc' + datetag + '.csv'}
    application_data.to_csv(savefiles['predictions'])
    ClusteredModel_TuningResults.to_csv(savefiles['tuning scores'])
    ClusteredModel_BestParams.to_csv(savefiles['best params'])
    ClusteredModel_CalCoeffs.to_csv(savefiles['calibration coefficients'])

    if save_models:
        savefiles['final models'] = outpath + outfile + '_finalModels_acc' + datetag + '.joblib'
        joblib.dump({'models': final_CMV.models, 'params': final_CMV.params,
                     'cal_coeffs': cal_coeffs, 'feat_list': feat_list,
                     'sea_ice_clusters': sea_ice_clusters,
                     'algorithms': {k: v[0] for k, v in cluster_models.items()}},
                    savefiles['final models'])

    end_time = datetime.now().strftime('%Y%m%d %H:%M:%S')
    savefiles['run description'] = outpath + outfile + '_runDescription_acc' + datetag + '.txt'
    with open(savefiles['run description'], "w") as file:
        file.write('RUN_START: ' + start_time + '\n')
        file.write('RUN_END: ' + end_time + '\n')
        file.write('N_FOLDS: ' + str(n_folds) + '\n')
        file.write('CV_TYPE: ' + cv_type + '\n')
        file.write('TUNING_SCORE_METRIC: ' + scoring_metric + '\n')
        file.write('CLUSTER_MODELS_SOURCE: ' + str(cluster_models_source) + '\n')
        file.write('N_TRAINVAL: ' + str(len(trainval_data)) + '\n')
        file.write('N_APPLICATION: ' + str(len(application_data)) + '\n')
        file.write('N_PREDICTED: ' + str(application_data['weighted_pred'].notna().sum()) + '\n')
        for classnum in range(1, n_clusters+1):
            training_algorithm, tuning_grid = cluster_models[classnum]
            file.write(f"CLUSTER{classnum}: {training_algorithm}\n")
            for key, value in tuning_grid.items():
                file.write(f"    {key}: {value}\n")

    for label, savefile in savefiles.items():
        print('Saved ' + label + ' to ' + savefile)

    del tuning_futures, fold_futures, final_futures, cvtainers_future
    return application_data


def get_regressor(training_algorithm):
    """ Base regressor for each algorithm, add more options as needed """
    if training_algorithm == 'RFR':
        return RandomForestRegressor(random_state=24, n_jobs=1)
    elif training_algorithm == 'XGB':
        return XGBRegressor(random_state=24, n_jobs=1)
    elif training_algorithm == 'ERT':
        return ExtraTreesRegressor(random_state=24, n_jobs=1)
    raise ValueError('Unknown training_algorithm: ' + str(training_algorithm))


def fit_tuning_folds(hyperparams_set,
                     data_cvtainers, cluster,
                     use_regressor,
                     class_feat_list, var_predict,
                     scoring_metric='median_abs_error'):
    """
    Iterate over tuning folds given one set of parameters for one cluster.
    Single-level version of procedures_cv.fit_inner_folds.
    """
    cluster_data = data_cvtainers[cluster] # NestedCV object

    fold_scores = []
    for foldtag in cluster_data.fold_list:
        fold_train = (cluster_data.get_training_data(foldtag)
                            .dropna(subset=class_feat_list + [var_predict]).copy())
        fold_val = (cluster_data.get_validation_data(foldtag)
                            .dropna(subset=class_feat_list + [var_predict]).copy())

        Mdl = clone(use_regressor)
        Mdl.set_params(**hyperparams_set)
        Mdl.fit(fold_train[class_feat_list], fold_train[var_predict])

        prediction = Mdl.predict(fold_val[class_feat_list])

        # Choose scoring metric, add others as desired
        if scoring_metric=='median_abs_error':
            score = np.median(np.abs(prediction - fold_val[var_predict])) #procedures_cv.calculate_MAE(prediction, fold_val[var_predict])
            fold_scores.append(score)
    mean_score = np.mean(fold_scores) # mean score for that hyp set amongst folds

    return pd.Series({'cluster': cluster,
                      'params': hyperparams_set,
                      'mean_score': mean_score} | hyperparams_set)


def fit_final_class_model(tuned_params,
                          data_cvtainers, classtag,
                          use_regressor,
                          class_feat_list,
                          var_predict):
    """ Fit a class-dependent model on all of the cluster's data using best hyperparameters """
    cluster_data = data_cvtainers[classtag]
    class_train = (cluster_data.input_data
                              .dropna(subset=class_feat_list + [var_predict])
                              .copy())
    Mdl = clone(use_regressor)
    Mdl.set_params(**tuned_params)
    Mdl.fit(class_train[class_feat_list], class_train[var_predict])

    return {'cluster': classtag, 'model': Mdl, 'tuned_params': tuned_params}


def apply_final_model(final_CMV, application_data, feat_list, var_predict,
                      sea_ice_clusters, n_clusters):
    """
    Add cluster{k}_pred, weighted_pred, cluster{k}_calpred and lincal_pred columns to application_data.
    Rows missing a required feature are kept, with NaN predictions.
    """
    pred_cols = (['cluster' + str(k) + '_pred' for k in range(1, n_clusters+1)] + ['weighted_pred']
                 + ['cluster' + str(k) + '_calpred' for k in range(1, n_clusters+1)] + ['lincal_pred'])

    # Positional index so predictions map back even if the original index has duplicates
    applyDF = application_data.reset_index(drop=True)

    predDF = regression.apply_cluster_RFR(feat_list, final_CMV, applyDF,
                                          target_var = var_predict,
                                          sea_ice_clusters = sea_ice_clusters,
                                          target_known = False)
    predDF = regression.apply_clustered_calibration(final_CMV.cal_coeffs, predDF, n_clusters,
                                                    target_var = var_predict,
                                                    target_known = False)

    application_data = application_data.copy()
    application_data[pred_cols] = predDF[pred_cols].reindex(applyDF.index).to_numpy()
    return application_data
