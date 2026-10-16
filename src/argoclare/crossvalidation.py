# Helper functions f
import pandas as pd
import xarray as xr
import numpy as np
import scipy
from scipy.stats import iqr
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# %% Fold functions
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut

from argoclare import regression as mod_reg


# %% Main object to store cross-validation data

class NestedCrossValContainer: 
    """ Nested cross-validation version (Schratz 2019) with an
    outer loop for error estimation and inner loop for hyperparameter tuning
    """
    def __init__(self, input_data, nfolds=5, skip_storage=False):
        """ 
        @ param     nfolds: number of folds (outer or inner)
        """
        fold_list = ['fold' + str(k) for k in range(1, nfolds+1)]
        
        # Outer folds for main model comparison
        if skip_storage: self.input_data = None
        else: self.input_data = input_data
        self.fold_list = fold_list

        # rename from trainDF, valDF
        # self.training_data = {fold: None for fold in fold_list} # Collapsed across classes, analogous to trainDF_all in CrossValContainer
        # self.validation_data = {fold: None for fold in fold_list}

        self.training_inds = {fold: None for fold in fold_list} # 
        self.validation_inds = {fold: None for fold in fold_list}

        # Later, populate this with other NestedCrossValContainer objects for the inner loop of tuning hyperparameters
        self.inner_nests = {fold:None for fold in fold_list} # ['fold' + str(k) for k in range(1, n_outer+1)]} 
    
    def copy(self):
        return self
    
    def to_labeled_dataframe(self):
        """ Collapse training and validation data across folds into a single dataframe """
        fullDF = pd.DataFrame()
        for nfold in self.fold_list:
            temp = self.validation_data[nfold]
            temp['fold'] = np.tile(nfold, len(temp))
            fullDF = pd.concat([fullDF, temp], axis=0)
        return fullDF
    
    def get_validation_data(self, foldtag):
        """ Return validation data for a given fold tag """
        return self.input_data.loc[self.validation_inds[foldtag]]
    
    def get_training_data(self, foldtag):
        """ Return training data for a given fold tag """
        return self.input_data.loc[self.training_inds[foldtag]]

    def get_inner_validation_data(self, outer_foldtag, inner_foldtag):
        """ Retrieve inner validation data to be used for hyperparameter tuning. 
        Ex: cluster_data.get_inner_validation_data(outer_foldtag, inner_foldtag)"""
        tuning_data = self.inner_nests[outer_foldtag]
        infold_data = (self.get_training_data(outer_foldtag)
                            .loc[tuning_data.validation_inds[inner_foldtag]])
        return infold_data
    
    def get_inner_training_data(self, outer_foldtag, inner_foldtag):
        """ Retrieve inner training data to be used for hyperparameter tuning. """
        tuning_data = self.inner_nests[outer_foldtag]
        infold_data = (self.get_training_data(outer_foldtag)
                           .loc[tuning_data.training_inds[inner_foldtag]])
        return infold_data
    

# %% Containers for storing data cross-validation containers

def create_cv_nest(trainval_data,
                    n_split: int,
                    spatial_cv = 'platform',
                    latitude_scaler = 1,
                    indexer='platform_id',
                  skip_storage = False,
                  random_state = None): 
    """ 
    Internal function in setup_cv_container.
    Equivalent to calling setup_cv_container with n_inner = None

    :param trainval_data: dataframe with data to split into training/validation folds
    :param n_splits
    :param spatial_cv: whether to use spatial clustering for fold splits 
                    default False, i.e. random splits by platform)
                    if 'k-means', use clustering on lat/lon coordinates
    :param latitude_scaler: float, scaling ratio for latitude/longitude in k-means
                    default of 1 gives range [-1.1] to match sinusoidal longitude range [-1,1]
                    changing to >1 gives more weight to latitude in clustering (ex: 3 yields range [-3,3])

    :param indexer: column name to use for identifying unique platforms for splitting 
                    default 'platform_id' if running .align_platform_indexers() on combined dataset
    :param random_state: int seed for platform fold splits (default None, not reproducible)

    """
    # Set up folds for cross-validation, with optional nesting
    cvtainer = NestedCrossValContainer(trainval_data, n_split, skip_storage=skip_storage) # Store in soclass_cvtainers[soclass]
    # [cvtainer.training_inds, cvtainer.validation_inds] = subset_folds(trainval_data, 
    #                                                                       type='platform',
    #                                                                         indexer=indexer, 
    #                                                                         nfolds=n_split)
    if spatial_cv == 'platform':
        [cvtainer.training_inds, cvtainer.validation_inds] = subset_folds(trainval_data, 
                                                                          type='platform',
                                                                            indexer=indexer, 
                                                                            nfolds=n_split,
                                                                            random_state=random_state)
    elif spatial_cv == 'kmeans':
        [cvtainer.training_inds, cvtainer.validation_inds] = subset_folds(trainval_data, 
                                                                          type='kmeans',
                                                                            indexer=indexer, 
                                                                            nfolds=n_split,
                                                                            latitude_scaler = latitude_scaler)

    return cvtainer

def setup_cv_container(trainval_data,
                       n_outer: int,
                       n_inner = None, 
                       spatial_cv = False,
                       latitude_scaler = 1,
                       indexer='platform_id',
                       random_state = None) -> NestedCrossValContainer: 
    """ 
    Sets up nested cross-validation containers for each class

    :param trainval_data: dataframe with data to split into training/validation folds
    :param n_outer: number of outer folds; performance estimation level
    :param n_inner: number of inner folds; hyperparameter tuning level  
                    default None, equivalent to non-nested CV) 
    :param spatial_cv: whether to use spatial clustering for fold splits 
                    default False, i.e. random splits by platform)
                    if 'k-means', use clustering on lat/lon coordinates
    :param latitude_scaler: float, scaling ratio for latitude/longitude in k-means
                    default of 1 gives range [-1.1] to match sinusoidal longitude range [-1,1]
                    changing to >1 gives more weight to latitude in clustering (ex: 3 yields range [-3,3])
    :param indexer: column name to use for identifying unique platforms for splitting 
                    default 'platform_id' if running .align_platform_indexers() on combined dataset
    :param random_state: int seed for platform fold splits (default None, not reproducible)
                    the same seed gives identical folds across runs on the same data

    :return: NestedCrossValContainer with outer folds for performance estimation
                        and optional inner folds for tuning
    """
    # Set up outer folds, for both ship and float
    outer_nest = create_cv_nest(trainval_data, n_outer, 
                                spatial_cv=spatial_cv, latitude_scaler = latitude_scaler, indexer=indexer,
                                random_state=random_state)
    
    # Optional inner folds for hyperparameter tuning (nested CV)
    if n_inner is not None:
        for outer_fold_tag in outer_nest.fold_list: 
            # Use only the training data from the outer fold to make a new set of splits 
            # training_data = outer_nest.get_training_data[outer_fold_tag]
            inner_nest = create_cv_nest(outer_nest.get_training_data(outer_fold_tag), 
                                        n_inner, 
                                        spatial_cv=spatial_cv, latitude_scaler = latitude_scaler, indexer=indexer,
                                        skip_storage=True,
                                        random_state=random_state)
            # inner_nest = create_cv_nest(outer_nest.training_data[outer_fold_tag], n_inner, 
            #                             spatial_cv=spatial_cv, latitude_scaler = latitude_scaler, indexer=indexer)
            outer_nest.inner_nests[outer_fold_tag] = inner_nest
    else: outer_nest.inner_nests = None

    return outer_nest



# %% Data splitting functions for training and validation sets
def align_platform_indexers(plat_data : list[pd.DataFrame], 
                            plat_indexers : list[str]) -> pd.DataFrame:
    """ 
    Renames unique platform indexers (e.g. wmoid for floats) to standard "platform_id")

    :param plat_data: list of dataframes for each platform (e.g. floatDF, socatDF)
    :param plat_indexers: list of column names to use as indexers for fold splitting
                        ('wmoid' or 'cruiseid')
    
    :return None
    """
    output = []
    for ind, platDF in enumerate(plat_data): # floatDF, socatDF
        current_indexer = plat_indexers[ind]
        renamed_platDF = platDF.rename(columns={current_indexer: 'platform_id'})
        renamed_platDF['platform_id'] = renamed_platDF['platform_id'].astype(str) # make sure all are strings
        renamed_platDF['platform_indexer'] = np.tile(current_indexer, len(platDF)) # add column to keep track of original indexer type
        output.append(renamed_platDF)

    return pd.concat(output, axis=0)


def subset_folds(platDF, type= 'platform', indexer='wmoid', nfolds=5,
                            latitude_scaler=1, random_state=None) -> list[dict[str, pd.DataFrame]]:
    """ 
    Returns dictionary of training and validation dataframes for each fold.

    :param platDF: dataframe for either floatDF or shipDF
    :param type: options by 'platform', 'kmeans', or 'random' 
    :param indexer: choose whether to split by 'wmoid' (float) or 'profid' (profile)
                    only used if type = 'platform'
    :param nfolds: (int) number of folds for cross-validation
    :param latitude_scaler: float, scaling ratio for latitude/longitude in k-means
                            default of 1 gives range [-1.1] to match sinusoidal longitude range [-1,1]
                            changing to >1 gives more weight to latitude in clustering (ex: 3 yields range [-3,3])
    :param random_state: int seed for shuffling platforms (type='platform' only), default None
    :return: training_inds: dictionary with keys 'fold1', 'fold2', ... with training DF integer locs
             validation_inds: dictionary with keys 'fold1', 'fold2', ... with validation DF integer locs

    """
    training_inds = {('fold'+str(k+1)):None for k in range(nfolds)}
    validation_inds = {('fold'+str(k+1)):None for k in range(nfolds)}
    
    if type == 'random': # not recommended!
        kf = KFold(n_splits=nfolds, random_state=42, shuffle=True)
        for ind, (train_index, val_index) in enumerate(kf.split(platDF)):
            #returns integer locs
            # training_inds['fold' + str(ind+1)] = np.isin(np.arange(len(platDF)), train_index)
            # validation_inds['fold' + str(ind+1)] = np.isin(np.arange(len(platDF)), val_index)
            training_inds['fold' + str(ind+1)] = platDF.index[train_index]
            validation_inds['fold' + str(ind+1)] = platDF.index[val_index]

    elif type == 'platform': 
        # platDF[indexer] = platDF[indexer].astype(str) #catch
        ids = platDF[indexer].astype(str).unique().to_numpy()
        np.random.default_rng(random_state).shuffle(ids)
        holdout_ids = np.array_split(ids, nfolds)
        for k in range(nfolds):
            # training_inds[('fold'+str(k+1))] = ~platDF[indexer].isin(holdout_ids[k])
            # validation_inds[('fold'+str(k+1))] = platDF[indexer].isin(holdout_ids[k])
            training_inds['fold' + str(k+1)] = platDF.index[
                ~platDF[indexer].isin(holdout_ids[k])]

            validation_inds['fold' + str(k+1)] = platDF.index[
                platDF[indexer].isin(holdout_ids[k])]

    elif type == 'kmeans': 
        kmeans = KMeans(n_clusters=nfolds, random_state=42)
        platDF['sin_longitude'] = np.sin(np.radians(platDF['longitude']))
        platDF['cos_longitude'] = np.cos(np.radians(platDF['longitude']))

        latitude_0to1 = (platDF['latitude'] - platDF['latitude'].min()) / platDF['latitude'].max() #range 0 to 1
        platDF['scaled_latitude'] = -latitude_scaler + latitude_scaler*2*(latitude_0to1) 
        platDF['fold'] = kmeans.fit_predict(platDF[['scaled_latitude', 'sin_longitude', 'cos_longitude']])

        for fnum in range(nfolds):
            # training_inds['fold' + str(fnum+1)] = platDF['fold'] != fnum
            # validation_inds['fold' + str(fnum+1)] = platDF['fold'] == fnum
            training_inds['fold' + str(fnum+1)] = platDF.index[
                platDF['fold'] != fnum]

            validation_inds['fold' + str(fnum+1)] = platDF.index[
                platDF['fold'] == fnum]

    return training_inds, validation_inds

def subset_training_validation(platDF, indexer = 'wmoid', split_frac = 0.8):
    """  
    Data subsetting for simple holdout validation;
    withholds a fraction of the unique platform IDS for validation
    and uses rest to train a single model.

    :param platDF: dataframe for either floatDF or shipDF
    :param indexer: column whether to split by 'wmoid' or 'cruiseid' 
    :param split_frac: fraction of unique platforms to use for training (default 0.8)
    
    :return: trainingDF: data for training, approx. 80% by default
            validationDF: data to withhold for validation
    """
    # Create list of unique profile ID's and select random 80% for training
    ids = platDF[indexer].unique()
    np.random.seed(42) 
    training_ids = np.random.choice(ids, int(np.floor(split_frac*len(ids))), replace=False)
    trainingDF = platDF[platDF[indexer].isin(training_ids)]

    validation_ids = [x for x in ids if x not in training_ids]
    validationDF = platDF[platDF[indexer].isin(validation_ids)]

    return trainingDF, validationDF, training_ids


        
# %% Main run function

def single_fold_nested_CV(soclass_cvtainers, soclass_list, sea_ice_clusters, # data
                            use_regressor, 
                            tuning_grid, 
                            feat_list, outer_foldtag, 
                            var_predict = 'delta_pco2'):
    """
    Main function to run nested CV for overall performance estimation (outer fold)
    Each fold run will populate fields in fold_CMV, which will be combined in storedRuns

    @ param
    foldkey: str 'fold1' key for dictionary
        soclass_cvtainers: dictionary of NestedCrossValContainer objects for each class, with inner and outer folds already set up
        soclass_list: list of integers correspnding to class numbers
        sea_ice_clusters: list of class numbers that have sea ice profiles (include sea ice in feature list) 
        
    """
    print('Training outer fold ' + str(outer_foldtag) + '...')
    # Each fold will be associated with a ClusteredModelVersion object (separate models for each cluster)
    fold_CMV = mod_reg.ClusteredModelVersion(ind_list = soclass_list, feat_list = feat_list) #singleRun
    fold_valDF = pd.DataFrame()

    for k in soclass_list: # For each class, train a separate regression 
        # print(k)
        cluster_data = soclass_cvtainers['cluster' + str(k)] # nestedCV object

        fold_valDF = pd.concat([fold_valDF, cluster_data.get_validation_data(outer_foldtag)], axis=0) # this fold's validation data

        if k not in sea_ice_clusters: # Remove sea ice as feature where not needed
            class_feat_list = [x for x in feat_list if x != 'sea_ice']
            # print('No sea ice in class ' + str(k) + ', removing from features for this class')
        else: class_feat_list = feat_list.copy()

        if tuning_grid is not None:
            # ==== Inner leave-one-group-out CV for hyperparameter tuning 
            # print('\t\tTraining cluster ' + str(k) + ' inner folds for hyperparameter tuning...')
            logo_cv = LeaveOneGroupOut()
            logo_X = pd.DataFrame()
            tuning_data = cluster_data.inner_nests[outer_foldtag] # NestedCrossValContainer object for one outer fold
            
            # Make input array for GridSearchCV with groups for LeaveOneGroupOut splitting
            # for fnum, inner_foldtag in enumerate(tuning_data.validation_data.keys()):
            #     infold = tuning_data.validation_data[inner_foldtag].dropna(subset=(class_feat_list + [var_predict])).copy()
            print('Using inner folds for tuning...')
            for inner_foldtag in tuning_data.fold_list:
                # tuning_input_data = outer_nest.get_training_data[outer_fold_tag].loc[tuning_data.validation_inds[inner_foldtag]]
                infold = (cluster_data.get_inner_data(outer_foldtag, inner_foldtag)
                            .dropna(subset=class_feat_list + [var_predict])
                            .copy())
                # infold = tuning_data.get_validation_data(inner_foldtag).dropna(subset=class_feat_list + [var_predict]).copy()
                infold['inner_fold'] = inner_foldtag

                logo_X = pd.concat([logo_X, infold], axis=0)    
                
                # infold.loc[:,'inner_fold'] = np.tile(inner_foldtag, len(infold)) # assign group number for cv splitting
                # logo_X = pd.concat([logo_X, infold], axis=0)
            
            # Iterating over inner folds with GridSearchCV to find best hyperparameters
            clf = GridSearchCV(estimator=use_regressor, param_grid=tuning_grid, cv=logo_cv, n_jobs=1)
            inner_clf = clf.fit(X=logo_X[class_feat_list], y=logo_X[var_predict], groups=logo_X['inner_fold'])
            # ========= end inner CV  

            # Use tuned hyperparameters for best estimator for this particular outer fold, store for each class
            # didn't store clf.cv_results but could do this 
            # also did not calculate training error or "simple" validation error which would be without the weighted averaging
            fold_CMV.models[k] = inner_clf.best_estimator_ # GridSearchCV function already refits on all tuning_data
            fold_CMV.params[k] = inner_clf.best_params_
        
        else: # No inner tuning
            # print('test')
            fold_trainDF = cluster_data.get_training_data(outer_foldtag)
            fold_CMV.models[k] = mod_reg.fit_single_RFR(class_feat_list, 
                                    fold_trainDF, 
                                    var_predict = var_predict)

# == 

    fold_valDF = fold_valDF.dropna(subset=(feat_list + [var_predict])).copy()
    fold_CMV.weighted_validation = mod_reg.apply_cluster_RFR(feat_list, fold_CMV, fold_valDF, sample_tag='val', 
                                                             sea_ice_clusters=sea_ice_clusters)
    # print('break')
    # Calibrate each fold model
    cal_coeffs = mod_reg.get_clustered_calibration_coeffs(fold_CMV.weighted_validation)
    fold_CMV.weighted_validation = mod_reg.apply_clustered_calibration(cal_coeffs, fold_CMV.weighted_validation, len(soclass_list))

    return fold_CMV

def collapse_folds(RunTagResults, feat_list=None):
    #  ===  Collapse errors across folds so you can treat as storedRuns for comparison and visualization functions
    fold_list = RunTagResults.keys()
    runtag_collapsed = mod_reg.ClusteredModelVersion(ind_list = fold_list, feat_list=feat_list)

    singleRun_errors_across_folds = pd.DataFrame()
    for nfold in RunTagResults.keys():
        tempDF = RunTagResults[nfold].weighted_validation
        tempDF['fold'] = np.tile(nfold, len(tempDF))
        singleRun_errors_across_folds = pd.concat([singleRun_errors_across_folds, tempDF], axis=0)

    runtag_collapsed.weighted_validation = singleRun_errors_across_folds    

    return runtag_collapsed

# def summarize_fold_params(OuterFoldResults, n_clusters, n_outer, tuning_grid, print_results=False):
def summarize_fold_params(OuterFoldResults, n_clusters, n_outer, tuning_grid, print_results=False):
    arrays = [[x for x in range(1, n_clusters+1)],
        [x for x in range(1, n_outer+1)]]
    index = pd.MultiIndex.from_product(arrays, names=["cluster", "outfold"])

    # Best parameters from each fold, for each cluster
    
    Model_FoldParams = pd.DataFrame(index = index, columns= [x for x in tuning_grid.keys()])
    for outfold in range(1, n_outer+1):
        outer_foldtag = 'fold' + str(outfold)
        for classnum, best_params in OuterFoldResults[outer_foldtag].params.items():
            Model_FoldParams.loc[classnum, outer_foldtag] = best_params

    if print_results: print(Model_FoldParams)

    return Model_FoldParams

# %% Useful functions for data diagnostics

def get_cvtainer_counts(soclass_cvtainers, n_clusters, n_outer, n_inner=None):
    """ function for getting # of observations in nested architecture
    Works with either nested or non-nested (n_inner=None) CV objects

    @param: soclass_cvtainers: dictionary of mod_reg.NestedCrossValContainers, one for each class
            n_clusters: number of southern ocean classes
            n_outer: number of outer folds
            n_inner: number of inner folds (if None, will just get outer fold counts)
    """
    arrays = [['cluster'+str(x) for x in range(1, n_clusters+1)],
       ['fold'+str(x) for x in range(1, n_outer+1)]]
    colnames = ["cluster", "outfold"]

    if n_inner is not None:
       arrays = arrays + [['total'] + ['fold' + str(x) for x in range(1, n_inner+1)]]
       colnames = colnames + ['infold']

    index = pd.MultiIndex.from_product(arrays, names=colnames)
    NestedCvtainerCounts = pd.DataFrame(index = index, columns= ['train_counts', 'val_counts'])
    for classnum in ['cluster'+str(x) for x in range(1, n_clusters+1)]:
        if n_inner is None:
            for foldnum in ['fold'+str(x) for x in range(1, n_outer+1)]:
                trainDF = soclass_cvtainers[classnum].get_training_data(foldnum)
                valDF = soclass_cvtainers[classnum].get_validation_data(foldnum)
                NestedCvtainerCounts.loc[(classnum, foldnum), 'train_counts'] = len(trainDF)
                NestedCvtainerCounts.loc[(classnum, foldnum), 'val_counts'] = len(valDF)
        
                # NestedCvtainerCounts.loc[(classnum, foldnum), 'train_counts'] = len(soclass_cvtainers[classnum].training_data[foldnum])
                # NestedCvtainerCounts.loc[(classnum, foldnum), 'val_counts'] = len(soclass_cvtainers[classnum].validation_data[foldnum])
        
        if n_inner is not None:
            for foldnum in ['fold'+str(x) for x in range(1, n_outer+1)]:
                trainDF = soclass_cvtainers[classnum].get_training_data(foldnum)
                valDF = soclass_cvtainers[classnum].get_validation_data(foldnum)
                NestedCvtainerCounts.loc[(classnum, foldnum, 'total'), 'train_counts'] = len(trainDF)
                NestedCvtainerCounts.loc[(classnum, foldnum, 'total'), 'val_counts'] = len(valDF)
                
                # NestedCvtainerCounts.loc[(classnum, foldnum, 'total'), 'train_counts'] = len(soclass_cvtainers[classnum].training_data[foldnum])
                # NestedCvtainerCounts.loc[(classnum, foldnum, 'total'), 'val_counts'] = len(soclass_cvtainers[classnum].validation_data[foldnum])
                
                inner_nest = soclass_cvtainers[classnum].inner_nests[foldnum] 
                for tuningnum in ['fold'+str(x) for x in range(1, n_inner+1)]:
                    tuning_data = soclass_cvtainers[classnum].inner_nests[foldnum]
                    tuning_trainDF = soclass_cvtainers[classnum].get_inner_data(foldnum, tuningnum, type='training')
                    tuning_valDF = soclass_cvtainers[classnum].get_inner_data(foldnum, tuningnum, type='validation')

                    NestedCvtainerCounts.loc[(classnum, foldnum, tuningnum), 'train_counts'] = len(tuning_trainDF)
                    NestedCvtainerCounts.loc[(classnum, foldnum, tuningnum), 'val_counts'] = len(tuning_valDF)

                    # NestedCvtainerCounts.loc[(classnum, foldnum, tuningnum), 'train_counts'] = len(soclass_cvtainers[classnum].inner_nests[foldnum].training_data[tuningnum])
                    # NestedCvtainerCounts.loc[(classnum, foldnum, tuningnum), 'val_counts'] = len(soclass_cvtainers[classnum].inner_nests[foldnum].validation_data[tuningnum])

    return NestedCvtainerCounts

