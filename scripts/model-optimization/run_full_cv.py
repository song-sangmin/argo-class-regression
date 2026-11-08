"""
Nested CV with full-mode calibration: version of run_dask_nested_cv.py whose calibration matches
run_model_application.py --calibration full (procedures_application_test.py).

Tuning, parameter selection and outer fold models are the same as procedures_cv.run_dask_nested_cv.
The calibration coefficients are fitted differently:
    procedures_cv.py: on each outer fold's validation rows (the rows they are then scored on)
    this file:        on each outer fold model's predictions on its own outer training rows, then
                      applied to the outer validation rows, so the calibrated errors are held out
For each outer fold and cluster k, the slope and intercept are fitted to the decile means of
cluster{k}_pred vs the target over cluster k's outer training rows, the same fit
procedures_application_test.py makes with the final model on all of trainval.

Takes the same arguments as run_dask_nested_cv.py. Outputs are the same files (default run name
ending in _fullcal), plus <run name>_calCoeffs with the slope and intercept for each outer fold and cluster.

Usage:
    python run_full_cv.py RFR rfr_full
    python run_full_cv.py RFR rfr_full --run-name crusoe_RFR_platformCV_fullcal --seed 24
"""

import os
from datetime import datetime

import pandas as pd
import run_dask_nested_cv
from dask.distributed import Client, LocalCluster
from procedures_cv import (
    evaluate_outer_fold,
    fit_inner_folds,
    get_regressor,
    make_parameter_combinations,
    organize_fold_results,
)
from sklearn.base import clone
from tuning_grids import TUNING_GRIDS

from argoclare import crossvalidation, regression


def fit_outer_class_model_fullcal(
    outer_foldtag, tuned_params, data_cvtainers, classtag, use_regressor, class_feat_list, var_predict
):
    """Fit a class-dependent model for one outer fold using best hyperparameters from inner folds
    (same as procedures_cv.fit_outer_class_model), and fit its calibration coefficients on its
    predictions on the outer training rows it was fitted on."""
    cluster_data = data_cvtainers[classtag]
    outer_train = cluster_data.get_training_data(outer_foldtag).dropna(subset=class_feat_list + [var_predict]).copy()
    Mdl = clone(use_regressor)
    Mdl.set_params(**tuned_params)
    Mdl.fit(outer_train[class_feat_list], outer_train[var_predict])

    # In-sample calibration: decile line of cluster{k}_pred vs the target over this cluster's outer training rows
    classnum = int(classtag.replace('cluster', ''))
    pred_col = 'cluster' + str(classnum) + '_pred'
    trainDF = pd.DataFrame(
        {
            'cluster': classnum,
            pred_col: Mdl.predict(outer_train[class_feat_list]),
            var_predict: outer_train[var_predict].to_numpy(),
        }
    )
    cal_coeffs = regression.get_clustered_calibration_coeffs(
        trainDF, target_var=var_predict, pred_col='cluster{k}_pred'
    )

    return {
        'outer_foldtag': outer_foldtag,
        'cluster': classtag,
        'model': Mdl,
        'tuned_params': tuned_params,
        'cal_coeffs': cal_coeffs[classnum],
    }


def run_full_cv(
    client,
    trainval_data,
    n_clusters,
    sea_ice_clusters,
    feat_list,
    var_predict,
    n_outer,
    n_inner,
    training_algorithm,
    tuning_grid,
    cv_type,
    scoring_metric,
    outpath,
    outfile,
    seed=None,
):
    """Main dask implementation
    Same as procedures_cv.run_dask_nested_cv, except that each outer fold's calibration coefficients
    are fitted on the outer fold models' predictions on their own training rows.

    Same parameters as procedures_cv.run_dask_nested_cv.
    """

    start_time = datetime.now().strftime('%Y%m%d %H:%M:%S')
    tune = bool(tuning_grid)  # no inner CV without hyperparameters to tune
    if not tune:
        n_inner = None
    ###################
    # 0. DATA SETUP
    ###################

    print('Setting up data containers...')
    class_tags = ['cluster' + str(k) for k in range(1, n_clusters + 1)]
    soclass_cvtainers = {soclass: None for soclass in class_tags}
    for classnum in range(1, n_clusters + 1):
        outer_nest = crossvalidation.setup_cv_container(
            trainval_data[trainval_data.cluster == classnum],
            n_outer=n_outer,
            n_inner=n_inner,
            spatial_cv='platform' if cv_type == 'profile' else cv_type,
            indexer='profid' if cv_type == 'profile' else 'platform_id',
            random_state=seed,
        )
        soclass_cvtainers['cluster' + str(classnum)] = outer_nest
    cvtainers_future = client.scatter(soclass_cvtainers, broadcast=True)

    parameter_combinations = make_parameter_combinations(tuning_grid) if tune else []
    use_regressor = get_regressor(training_algorithm)

    ###################
    # 1. INNER CV
    ###################

    print('Building dask tasks for inner CV...' if tune else 'No tuning grid, skipping inner CV...')
    inner_futures = []

    for outer_foldnum in range(1, n_outer + 1):
        for classnum in range(1, n_clusters + 1):
            if classnum not in sea_ice_clusters:
                class_feat_list = [x for x in feat_list if x != 'sea_ice']
            else:
                class_feat_list = feat_list.copy()

            # Run inner folds and store mean score for each parameter set
            for param_set in parameter_combinations:
                inner_future = client.submit(
                    fit_inner_folds,  # returns pd Series
                    param_set,
                    'fold' + str(outer_foldnum),  # outer_foldtag,
                    cvtainers_future,
                    'cluster' + str(classnum),
                    use_regressor,
                    class_feat_list,
                    var_predict,
                    scoring_metric,
                )

                inner_futures.append(inner_future)

    # Compile inner results (each for one parameter set, one outer fold, one cluster)
    inner_results = client.gather(inner_futures)
    if tune:
        print('Dask computation for inner CV complete.')

    ###################
    # 2. PARAMETER SELECTION
    ###################
    print('Aggregating results for each outer fold and class...')

    inner_summary = pd.DataFrame(inner_results)

    if tune:
        best_idx = inner_summary.groupby(['outer_foldtag', 'cluster'])['mean_score'].idxmin()
        foldxclass_best_params = (inner_summary.loc[best_idx].set_index(['outer_foldtag', 'cluster']))[['params']]
    else:  # default parameters ({}) for every outer fold and class
        fold_index = pd.MultiIndex.from_product(
            [['fold' + str(k) for k in range(1, n_outer + 1)], class_tags], names=['outer_foldtag', 'cluster']
        )
        foldxclass_best_params = pd.DataFrame({'params': [{} for _ in fold_index]}, index=fold_index)

    print('Selected best parameters for each outer fold and class.')

    ###################
    # 3. OUTER CV - FIT CLASS/FOLD MODELS AND CALIBRATION COEFFICIENTS
    ###################

    print('Building dask tasks for outer CV using tuned parameters...')
    outer_futures = []

    for outer_foldnum in range(1, n_outer + 1):
        for classnum in range(1, n_clusters + 1):
            fctag = ('fold' + str(outer_foldnum), 'cluster' + str(classnum))
            tuned_params = foldxclass_best_params.loc[fctag, 'params']

            if classnum not in sea_ice_clusters:
                class_feat_list = [x for x in feat_list if x != 'sea_ice']
            else:
                class_feat_list = feat_list.copy()

            outer_future = client.submit(
                fit_outer_class_model_fullcal,  # also returns cal_coeffs fitted on the outer training rows
                'fold' + str(outer_foldnum),
                tuned_params,
                cvtainers_future,
                'cluster' + str(classnum),
                use_regressor,
                class_feat_list,
                var_predict,
            )

            outer_futures.append(outer_future)

    outer_results = client.gather(outer_futures)
    print('Dask computation for fitting outer fold models and calibration coefficients complete.')

    ###################
    # 4. BUILD FOLD CMV + COMPUTE OUTER VALIDATION ERRORS
    ###################
    print('Evaluating outer fold models on validation data...')
    model_finder = organize_fold_results(outer_results)
    cal_finder = {(res['outer_foldtag'], res['cluster']): res['cal_coeffs'] for res in outer_results}
    ClusteredModel_OuterFoldResults = {}  # ClusteredModelVersion results
    for outer_foldnum in range(1, n_outer + 1):
        outer_foldtag = 'fold' + str(outer_foldnum)

        fold_CMV = regression.ClusteredModelVersion(ind_list=[x for x in range(1, n_clusters + 1)], feat_list=feat_list)
        fold_valDF = pd.concat(
            [soclass_cvtainers[classtag].get_validation_data(outer_foldtag) for classtag in class_tags], axis=0
        )  # this fold's validation data

        for classnum in range(1, n_clusters + 1):
            classtag = 'cluster' + str(classnum)
            fctag = (outer_foldtag, classtag)
            fold_CMV.models[classnum] = model_finder.loc[outer_foldtag, classtag].values[0]  # Regressor object
            fold_CMV.params[classnum] = foldxclass_best_params.loc[fctag, 'params']

        fold_CMV = evaluate_outer_fold(fold_CMV, fold_valDF, feat_list, var_predict, sea_ice_clusters)

        # Linear calibration: coefficients fitted on each cluster model's outer training rows (step 3),
        # applied to this fold's validation rows
        fold_CMV.cal_coeffs = {
            classnum: cal_finder[(outer_foldtag, 'cluster' + str(classnum))] for classnum in range(1, n_clusters + 1)
        }
        fold_CMV.weighted_validation = regression.apply_clustered_calibration(
            fold_CMV.cal_coeffs, fold_CMV.weighted_validation, n_clusters, target_var=var_predict
        )

        ClusteredModel_OuterFoldResults[outer_foldtag] = fold_CMV

    print('Model evaluation by outer fold complete. Collapsing results and saving results...')

    #############
    # SAVE RESULTS
    ##############
    print('Saving results...')

    # Store weighted validation DF with estimates on all outer folds
    ClusteredModel_ValidationErrors = crossvalidation.collapse_folds(ClusteredModel_OuterFoldResults)
    if tune:
        ClusteredModel_InnerResults = inner_summary.drop(columns=['params']).set_index(['outer_foldtag', 'cluster'])
    ClusteredModel_BestParams = foldxclass_best_params.copy()  # best params used to train each outer fold model
    ClusteredModel_CalCoeffs = pd.DataFrame.from_dict(cal_finder, orient='index', columns=['slope', 'intercept'])
    ClusteredModel_CalCoeffs.index = pd.MultiIndex.from_tuples(
        ClusteredModel_CalCoeffs.index, names=['outer_foldtag', 'cluster']
    )

    datetag = datetime.now().strftime('%Y%m%d')
    savefiles = {
        'validation errors': outpath + outfile + '_outercvErrors_acc' + datetag + '.csv',
        'best params': outpath + outfile + '_bestParams_acc' + datetag + '.csv',
        'calibration coefficients': outpath + outfile + '_calCoeffs_acc' + datetag + '.csv',
        'run description': outpath + outfile + '_runDescription_acc' + datetag + '.txt',
    }
    if tune:
        savefiles['inner fold scores'] = outpath + outfile + '_innerScores_acc' + datetag + '.csv'

    ClusteredModel_ValidationErrors.weighted_validation.to_csv(savefiles['validation errors'])
    if tune:
        ClusteredModel_InnerResults.to_csv(savefiles['inner fold scores'])
    ClusteredModel_BestParams.to_csv(savefiles['best params'])
    ClusteredModel_CalCoeffs.to_csv(savefiles['calibration coefficients'])

    end_time = datetime.now().strftime('%Y%m%d %H:%M:%S')
    with open(savefiles['run description'], 'w') as file:
        file.write('RUN_START: ' + start_time + '\n')
        file.write('RUN_END: ' + end_time + '\n')
        file.write('ALGORITHM: ' + training_algorithm + '\n')
        file.write('N_OUTER: ' + str(n_outer) + '\n')
        file.write('N_INNER: ' + (str(n_inner) if tune else '0 (no tuning grid)') + '\n')
        file.write('CV_TYPE: ' + cv_type + '\n')
        file.write('SEED: ' + str(seed) + '\n')
        file.write('INNER_SCORE_METRIC: ' + scoring_metric + '\n')
        file.write(
            'CALIBRATION: full, cluster{k}_pred decile fit on each outer fold model predictions on its '
            'outer training rows\n'
        )
        file.write('TUNING_GRID: ' + '\n')
        for key, value in tuning_grid.items():
            file.write(f'{key}: {value}\n')

    for name, path in savefiles.items():
        print(f'Saved {name} to {path}')

    del inner_futures, outer_futures, cvtainers_future
    return


def main():
    args = run_dask_nested_cv.parse_args()  # same arguments as run_dask_nested_cv.py
    tuning_grid = TUNING_GRIDS[args.grid]
    run_name = args.run_name or f'crusoe_{args.algorithm}_{args.cv_type}CV_grid_{args.grid}_fullcal'
    savepath = os.path.join(args.savepath, '')  # outfile is appended directly to savepath
    os.makedirs(savepath, exist_ok=True)
    run_feat_list = [
        x for x in (getattr(args, 'features', None) or run_dask_nested_cv.feat_list) if x not in args.drop_features
    ]
    target = getattr(args, 'target', None) or run_dask_nested_cv.var_predict

    trainval_data = pd.read_csv(args.data, index_col=0)  # for platformID
    if args.cv_type == 'profile' and 'profid' not in trainval_data.columns:
        raise SystemExit(
            f'--cv-type profile needs a profid column in {args.data}: rerun '
            'regression-input-prep.py with [regression] cv_type = "profile"'
        )
    n_clusters = int(trainval_data.columns.str.fullmatch(r'cluster\d+_prob').sum())  # GMM classes
    print(f'Loaded {len(trainval_data)} samples from {args.data} ({n_clusters} clusters)')
    print(f'Algorithm: {args.algorithm}, grid: {args.grid} = {tuning_grid}')
    inner_folds = f'{args.n_inner} inner' if tuning_grid else 'no inner (no tuning)'
    print(f'Folds: {args.n_outer} outer x {inner_folds}, seed: {args.seed}, run name: {run_name}')
    print(f'Features: {run_feat_list}, target: {target}, sea ice clusters: {args.sea_ice_clusters}')
    print('Calibration: full (fitted on outer training rows)')

    with LocalCluster(dashboard_address=None) as cluster, Client(cluster) as client:
        print(client)
        print('Starting dask training: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))
        run_full_cv(
            client,
            trainval_data,
            n_clusters,
            args.sea_ice_clusters,
            run_feat_list,
            target,
            args.n_outer,
            args.n_inner,
            args.algorithm,
            tuning_grid,
            args.cv_type,
            args.scoring,
            savepath,
            run_name,
            seed=args.seed,
        )
        print('Finished dask cross validation at: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))


if __name__ == '__main__':
    main()
