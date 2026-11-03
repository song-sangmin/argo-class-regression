"""Alternate version of procedures_application.run_model_application with in-sample calibration.

Tuning and parameter selection are the same as procedures_application.py. The calibration
coefficients are fitted differently:
    procedures_application.py: out-of-fold predictions within trainval (one model per fold)
    this file:                 the final models' predictions on all of trainval (the data
                               they were trained on)
For each cluster, the slope and intercept are fitted to the decile means of weighted_pred
vs the target, over all of that cluster's trainval samples, as in procedures_cv.py. They are
stored with the final model (final_CMV.cal_coeffs, and the joblib file with --save-models)
and applied to the final model's predictions on the application and test data.
"""

from datetime import datetime

import joblib
import pandas as pd
import procedures_cv
from procedures_application import (
    apply_final_model,
    fit_final_class_model,
    fit_tuning_folds,
    get_regressor,
    get_test_metrics,
)

from argoclare import crossvalidation, regression


def run_model_application(
    client,
    trainval_data,
    application_data,
    n_clusters,
    sea_ice_clusters,
    feat_list,
    var_predict,
    n_folds,
    cluster_models,
    cv_type,
    scoring_metric,
    outpath,
    outfile,
    save_models=False,
    cluster_models_source=None,
    test_data=None,
):
    """Main dask implementation
    Train one final clustered model on trainval_data and apply it to application_data.

    For each cluster: tune hyperparameters on n_folds (single-level CV), fit a final
    class-dependent model on all of that cluster's data, then fit calibration coefficients
    on the final models' predictions on all of trainval.

    Same parameters and return value as procedures_application.run_model_application.
    """

    start_time = datetime.now().strftime('%Y%m%d %H:%M:%S')
    ###################
    # 0. DATA SETUP
    ###################

    print('Setting up data containers...')
    class_tags = ['cluster' + str(k) for k in range(1, n_clusters + 1)]
    soclass_cvtainers = {soclass: None for soclass in class_tags}
    for classnum in range(1, n_clusters + 1):
        soclass_cvtainers['cluster' + str(classnum)] = crossvalidation.setup_cv_container(
            trainval_data[trainval_data.cluster == classnum],
            n_outer=n_folds,
            n_inner=None,
            spatial_cv='platform' if cv_type == 'profile' else cv_type,
            indexer='profid' if cv_type == 'profile' else 'platform_id',
        )
    cvtainers_future = client.scatter(soclass_cvtainers, broadcast=True)

    # Regressor, parameter combinations and features for each cluster
    class_regressors = {}
    class_param_combinations = {}
    class_feat_lists = {}
    for classnum in range(1, n_clusters + 1):
        training_algorithm, tuning_grid = cluster_models[classnum]
        class_regressors[classnum] = get_regressor(training_algorithm)
        class_param_combinations[classnum] = procedures_cv.make_parameter_combinations(tuning_grid)
        if classnum not in sea_ice_clusters:
            class_feat_lists[classnum] = [x for x in feat_list if x != 'sea_ice']
        else:
            class_feat_lists[classnum] = feat_list.copy()

    ###################
    # 1. HYPERPARAMETER TUNING
    ###################

    print('Building dask tasks for hyperparameter tuning...')
    tuning_futures = []

    for classnum in range(1, n_clusters + 1):
        for param_set in class_param_combinations[classnum]:
            tuning_future = client.submit(
                fit_tuning_folds,  # returns pd Series
                param_set,
                cvtainers_future,
                'cluster' + str(classnum),
                class_regressors[classnum],
                class_feat_lists[classnum],
                var_predict,
                scoring_metric,
            )

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
    # 3. FIT FINAL MODELS
    ###################

    print('Building dask tasks for final models using tuned parameters...')
    final_futures = []

    for classnum in range(1, n_clusters + 1):
        classtag = 'cluster' + str(classnum)
        final_future = client.submit(
            fit_final_class_model,  # returns a dict with cluster, model, tuned_params
            class_best_params.loc[classtag, 'params'],
            cvtainers_future,
            classtag,
            class_regressors[classnum],
            class_feat_lists[classnum],
            var_predict,
        )
        final_futures.append(final_future)

    final_results = client.gather(final_futures)
    print('Dask computation for fitting final models complete.')

    final_CMV = regression.ClusteredModelVersion(ind_list=[x for x in range(1, n_clusters + 1)], feat_list=feat_list)
    for res in final_results:
        classnum = int(res['cluster'].replace('cluster', ''))
        final_CMV.models[classnum] = res['model']
        final_CMV.params[classnum] = res['tuned_params']

    ###################
    # 4. CALIBRATION COEFFICIENTS FROM FINAL MODEL PREDICTIONS ON TRAINVAL
    ###################
    print('Computing calibration coefficients from final model predictions on all trainval data...')
    trainval_predictions = regression.apply_cluster_RFR(
        feat_list,
        final_CMV,
        trainval_data.reset_index(drop=True),  # positional index in case of duplicate labels
        sample_tag='train',
        target_var=var_predict,
        sea_ice_clusters=sea_ice_clusters,
    )
    cal_coeffs = regression.get_clustered_calibration_coeffs(trainval_predictions, target_var=var_predict)
    final_CMV.cal_coeffs = cal_coeffs  # stored with the final model

    ###################
    # 5. APPLY FINAL MODEL
    ###################
    print('Applying final clustered model to application data...')
    application_data = apply_final_model(
        final_CMV, application_data, feat_list, var_predict, sea_ice_clusters, n_clusters
    )

    # In-sample predictions and errors on trainval, with the calibration fitted on them
    trainval_data = apply_final_model(final_CMV, trainval_data, feat_list, var_predict, sea_ice_clusters, n_clusters)
    trainval_data['train_error'] = trainval_data['weighted_pred'] - trainval_data[var_predict]
    trainval_data['calibrated_train_error'] = trainval_data['lincal_pred'] - trainval_data[var_predict]

    if test_data is not None:
        print('Evaluating the final clustered model on the test data...')
        test_data = apply_final_model(final_CMV, test_data, feat_list, var_predict, sea_ice_clusters, n_clusters)
        test_data['test_error'] = test_data['weighted_pred'] - test_data[var_predict]
        test_data['calibrated_test_error'] = test_data['lincal_pred'] - test_data[var_predict]
        test_metrics = get_test_metrics(test_data)
        print(test_metrics.round(3).to_string())

    #############
    # SAVE RESULTS
    ##############
    print('Saving results...')

    ClusteredModel_TuningResults = tuning_summary.drop(columns=['params']).set_index(
        'cluster'
    )  # mean fold score per param set
    ClusteredModel_BestParams = class_best_params.copy()
    ClusteredModel_BestParams.insert(
        0, 'algorithm', [cluster_models[int(x.replace('cluster', ''))][0] for x in ClusteredModel_BestParams.index]
    )
    ClusteredModel_CalCoeffs = pd.DataFrame.from_dict(
        {'cluster' + str(k): v for k, v in sorted(cal_coeffs.items())}, orient='index', columns=['slope', 'intercept']
    )
    ClusteredModel_CalCoeffs.index.name = 'cluster'

    datetag = datetime.now().strftime('%Y%m%d')
    savefiles = {
        'predictions': outpath + outfile + '_predictions_acc' + datetag + '.csv',
        'trainval predictions': outpath + outfile + '_trainvalPredictions_acc' + datetag + '.csv',
        'tuning scores': outpath + outfile + '_tuningScores_acc' + datetag + '.csv',
        'best params': outpath + outfile + '_bestParams_acc' + datetag + '.csv',
        'calibration coefficients': outpath + outfile + '_calCoeffs_acc' + datetag + '.csv',
    }
    application_data.to_csv(savefiles['predictions'])
    trainval_data.to_csv(savefiles['trainval predictions'])
    ClusteredModel_TuningResults.to_csv(savefiles['tuning scores'])
    ClusteredModel_BestParams.to_csv(savefiles['best params'])
    ClusteredModel_CalCoeffs.to_csv(savefiles['calibration coefficients'])
    if test_data is not None:
        savefiles['test predictions'] = outpath + outfile + '_testPredictions_acc' + datetag + '.csv'
        savefiles['test metrics'] = outpath + outfile + '_testMetrics_acc' + datetag + '.csv'
        test_data.to_csv(savefiles['test predictions'])
        test_metrics.to_csv(savefiles['test metrics'])

    if save_models:
        savefiles['final models'] = outpath + outfile + '_finalModels_acc' + datetag + '.joblib'
        joblib.dump(
            {
                'models': final_CMV.models,
                'params': final_CMV.params,
                'cal_coeffs': cal_coeffs,
                'feat_list': feat_list,
                'sea_ice_clusters': sea_ice_clusters,
                'algorithms': {k: v[0] for k, v in cluster_models.items()},
            },
            savefiles['final models'],
        )

    end_time = datetime.now().strftime('%Y%m%d %H:%M:%S')
    savefiles['run description'] = outpath + outfile + '_runDescription_acc' + datetag + '.txt'
    with open(savefiles['run description'], 'w') as file:
        file.write('RUN_START: ' + start_time + '\n')
        file.write('RUN_END: ' + end_time + '\n')
        file.write('N_FOLDS: ' + str(n_folds) + '\n')
        file.write('CV_TYPE: ' + cv_type + '\n')
        file.write('TUNING_SCORE_METRIC: ' + scoring_metric + '\n')
        file.write('CALIBRATION: final model predictions on all trainval data (procedures_application_alternate.py)\n')
        file.write('CLUSTER_MODELS_SOURCE: ' + str(cluster_models_source) + '\n')
        file.write('N_TRAINVAL: ' + str(len(trainval_data)) + '\n')
        file.write('N_APPLICATION: ' + str(len(application_data)) + '\n')
        file.write('N_PREDICTED: ' + str(application_data['weighted_pred'].notna().sum()) + '\n')
        if test_data is not None:
            overall = test_metrics.loc['all']
            file.write('N_TEST: ' + str(len(test_data)) + '\n')
            file.write(f'TEST_RMSE: {overall["rmse"]:.4f} (calibrated {overall["calibrated_rmse"]:.4f})\n')
            file.write(f'TEST_MAE: {overall["mae"]:.4f} (calibrated {overall["calibrated_mae"]:.4f})\n')
        for classnum in range(1, n_clusters + 1):
            training_algorithm, tuning_grid = cluster_models[classnum]
            file.write(f'CLUSTER{classnum}: {training_algorithm}\n')
            for key, value in tuning_grid.items():
                file.write(f'    {key}: {value}\n')

    for label, savefile in savefiles.items():
        print('Saved ' + label + ' to ' + savefile)

    del tuning_futures, final_futures, cvtainers_future
    return application_data
