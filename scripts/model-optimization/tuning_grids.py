"""Hyperparameter tuning grids for run_dask_nested_cv.py, selected by name.

Add new grids here, then pass the key on the command line, e.g.
    sbatch -J RFR_full submit_nested_cv.sbatch RFR rfr_full
"""

TUNING_GRIDS = {
    # ===== RFR =====
    'rfr_full': {'max_features': range(1, 9),
                 'min_samples_leaf': range(1,10,2),
                 'n_estimators': [100, 250, 500, 750, 1000]},
    # Small grid for checking the job runs (use with -q debug)
    'rfr_test': {'max_features': range(1, 3),
                 'min_samples_leaf': [2, 5],
                 'n_estimators': [100, 500]},

    # ===== ERT =====
    'ert_full': {'max_features': range(1, 9),
                 'min_samples_leaf': range(1,10,2),
                 'n_estimators': [100, 250, 500, 750, 1000]},
    
    'ert_test': {'max_features': range(1, 3),
                 'min_samples_leaf': [2, 5],
                 'n_estimators': [100, 500]},

    # ===== XGB =====
    'xgb_full': {'eta': [0.01, 0.025, 0.05], # shrinkage
                 'max_depth': [2, 4, 6, 8],
                 'min_child_weight': [1, 5, 10, 20], # larger = more conservative
                 'colsample_bytree': [0.25, 0.5, 0.75], # prop. of features for each tree
                 'n_estimators': [100, 250, 500]},

    'xgb_test': {'eta': [0.01, 0.05],
                'max_depth': [2, 3]
                }
}
