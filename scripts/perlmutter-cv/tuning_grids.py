"""Hyperparameter tuning grids for run_dask_nested_cv.py, selected by name.

Add new grids here, then pass the key on the command line, e.g.
    sbatch -J RFR_full submit_nested_cv.sbatch RFR rfr_full
"""

TUNING_GRIDS = {
    # ===== RFR =====
    'rfr_full': {'max_features': range(1, 9),
                 'min_samples_split': range(2, 11, 2),
                 'min_samples_leaf': range(2, 11, 2),
                 'n_estimators': [100, 250, 500, 750, 1000]},
    # Small grid for checking the job runs (use with -q debug)
    'rfr_test': {'max_features': range(1, 3),
                 'min_samples_split': [5, 10],
                 'n_estimators': [100, 500]},

    # ===== ERT =====
    'ert_full': {'max_features': range(1, 6),
                 'min_samples_split': range(2, 11),
                 'n_estimators': [100, 250, 500, 750, 1000]},

    # ===== XGB =====
    'xgb_full': {'eta': [0.01, 0.02, 0.03, 0.04, 0.05], # shrinkage
                 'max_depth': [2, 3, 4, 5, 6],
                 'min_child_weight': [1, 2, 5, 10], # larger = more conservative
                 'colsample_bytree': [0.5, 0.625, 0.75, 0.875, 1.0], # prop. of features for each tree
                 'n_estimators': [100, 250, 500, 750, 1000]},
}
