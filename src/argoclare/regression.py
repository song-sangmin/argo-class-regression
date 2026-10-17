import pandas as pd
import xarray as xr
import numpy as np
import scipy
import math
from scipy.stats import iqr

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# %% Setup run tags
def make_feat_dict(feat_lists):
    """ 
    @param    feat_lists: list of lists of feature names for each cluster
    """
    ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    labels = ['feat'+ ascii_uppercase[i] for i in range(len(feat_lists))]
    feat_options = {labels[i]: feat_lists[i] for i in range(len(feat_lists))}

    return feat_options 

def make_run_tags(feat_options, data_options, target_options):
    """ 
    @ param   feat_options: dict of feature lists (make_feat_dict())
               data_options: dict of datasets for training, validation
                    data_options = {'float': [trainClasses_float, valClasses],
                        'ship': [trainClasses_ship, valClasses],
                        'combined': [trainClasses, valClasses]}


               target_options: list of target variable names"""
    # Automatically generate run tag combinations
    run_tags = []
    for key1 in feat_options.keys():
        for key2 in data_options.keys():
            for target in target_options:
                run_tags.append(key1 + '-' + key2 + '-' + target)
    return run_tags


# %% Training and validation data

def get_trainval_counts(training_classes, validation_classes, n_gmm=8):
    """  
    originally from 3.1_clustered_rfr.ipynb 
    @param  training_classes: dict of training dataframes for each cluster

            validation_classes: dict of validation dataframes for each cluster
    
    """
    countobs = pd.DataFrame({'cluster':range(1,n_gmm+1), 
                        'train':[len(k) for k in training_classes.values()],
                        'validation':[len(k) for k in validation_classes.values()]}
                         )

    countobs['train_float'] = [len(k[~k['wmoid'].isna()]) for k in training_classes.values()]
    countobs['train_ship'] = [len(k[k.wmoid.isna()]) for k in training_classes.values()]

    countobs['val_float'] = [len(k[~k['wmoid'].isna()]) for k in validation_classes.values()]
    countobs['val_ship'] = [len(k[k.wmoid.isna()]) for k in validation_classes.values()]

    # countobs['train_pc_float'] = countobs.apply(lambda row: row['train_float']/row['train']*100, axis=1)
    countobs['train_pc_float'] = countobs.apply(lambda row: row['train_float']/row['train']*100, axis=1)
    countobs['val_pc_float'] = countobs.apply(lambda row: row['val_float']/row['validation']*100, axis=1)

    countobs['TOTAL_FLOAT'] = countobs['train_float'] + countobs['val_float']
    countobs['TOTAL_SHIP'] = countobs['train_ship'] + countobs['val_ship']

    countobs['TOTAL_OBS'] = countobs['train'] + countobs['validation']

    # print('percent split into training/validation: ' + str(len(trainDF_all)/(len(trainDF_all)+len(valDF_all))*100))
    countobs.set_index('cluster', inplace=True)

    return countobs

def find_trainval_split(floatDF, shipDF, pc_thresh=0.3, n_gmm=6):
    """ 
    Main splitting function used in 3.1_clsutered_RFR
    Updated Jan 19 2026
    Instead of doing probability-based assignment, do signle assignment
    Iteratively search for a training/validation split that gives at least X% float

    @ param     pc_thresh : proportion of float data in training and validation sets
    """

    platDF_all = pd.concat([floatDF, shipDF], axis=0)
    trainClasses = {k:None for k in range(1, n_gmm+1)}
    valClasses = {k:None for k in range(1, n_gmm+1)}

    # pc_thresh = 0.3
    for soclass in range(1, n_gmm+1):
        print('==> Processing class ' + str(soclass))
        classDF = platDF_all[platDF_all['class_assign'] == soclass]
        classDF_float = classDF[~classDF.wmoid.isna()]
        classDF_ship = classDF[classDF.wmoid.isna()]

        pc_train_float = 0; pc_val_float = 0
        print('Searching for valid splits...')
        while (pc_train_float < pc_thresh) or (pc_val_float < pc_thresh):
            # Iteratively search for split until both training and validation have at least pc_thresh float
            [holdTrain_float, holdVal_float, _] = subset_training_validation(classDF_float, indexer='wmoid', split_frac=0.8)
            [holdTrain_ship, holdVal_ship, _] = subset_training_validation(classDF_ship, indexer='cruiseid', split_frac=0.8)
            # print('trying split')
            pc_train_float = len(holdTrain_float) / (len(holdTrain_float) + len(holdTrain_ship))
            pc_val_float = len(holdVal_float) / (len(holdVal_float) + len(holdVal_ship))

        trainClasses[soclass] = pd.concat([holdTrain_float, holdTrain_ship], axis=0)
        valClasses[soclass] = pd.concat([holdVal_float, holdVal_ship], axis=0)
        
    trainClasses_float = { ind : k[~k.wmoid.isna()] for ind, k in trainClasses.items()}
    trainClasses_ship = { ind : k[k.wmoid.isna()] for ind, k in trainClasses.items()}

    trainDF_all = pd.concat(trainClasses.values(), axis=0)
    valDF_all = pd.concat(valClasses.values(), axis=0)
    print('Resulting split for training/validation observations: ' + str(len(trainDF_all)/(len(trainDF_all)+len(valDF_all))*100))

    countobs = get_trainval_counts(trainClasses, valClasses, n_gmm=n_gmm)
    countobs

    return [trainClasses, valClasses, trainDF_all, valDF_all, countobs]

def separate_platforms(platDF_combined):
    """ 
    Split combined dataframe into float and ship dataframes
   :param platDF_combined: combined dataframe with both float and ship data
   :return: floatDF, shipDF
    """
    
    # floatDF = platDF_combined[~platDF_combined.wmoid.isna()].copy()
    # shipDF = platDF_combined[~platDF_combined.cruiseid.isna()].copy()

    floatDF = platDF_combined[platDF_combined.platform_indexer == 'wmoid']
    shipDF = platDF_combined[platDF_combined.platform_indexer == 'cruiseid']

    return [floatDF, shipDF]


# %% Simple model version object, without clustering
class ModelVersion: 
    """ 
    Object holding different model runs
    @ param     ind_list = class indices e.g. range(1,9)
                
                models - scikit learn RandomForestRegressor objects
                errorstats - maybe get rid of. holds error summary stats
                DF_err - validation DataFrame with estimation errors
    """
    def __init__(self, ind_list):
        self.ind_list = ind_list
        self.models = {model: None for model in ind_list}
        self.errorstats = {model: None for model in ind_list}
        self.DF_err = {model: None for model in ind_list}
    
    def add(self, model_name):
        self.models[model_name] = None
        self.errorstats[model_name] = None
        self.DF_err[model_name] = None

    def copy(self):
        return self
  
def fit_single_RFR(feat_list, 
              trainingDF, 
              var_predict = 'delta_fco2',
              algorithm ='RFR',
              loss_criterion = 'squared_error',
              ntrees=1000, max_feats = 1/3, min_samples_split=5,
              booster= 'gbtree', tree_method='approx'):
    """ 
    Fit single RFR for a class. 
    Compute validation errors later 
    """

    if algorithm == 'RFR':
        Mdl = RandomForestRegressor(ntrees, max_features = max_feats, random_state = 0, 
                                    criterion = loss_criterion,
                                    bootstrap=True, min_samples_split=min_samples_split)
    elif algorithm == 'XGB':
        Mdl = XGBRegressor(n_estimators=ntrees, 
                          booster = booster, 
                          tree_method=tree_method)

    trainingDF= trainingDF.dropna(subset=feat_list).copy()

    if 'mld' in feat_list:
        trainingDF['log_mld'] = np.log(trainingDF['mld'])
        feat_list = [x if x != 'mld' else 'log_mld' for x in feat_list]

    # Train the model 
    X_training = trainingDF[feat_list].to_numpy()
    Y_training = trainingDF[var_predict].to_numpy().flatten()
    Mdl.fit(X_training, Y_training)

    return Mdl

def apply_single_RFR(skmodel, feat_list, sampleDF, sample_tag='val', 
                      target_var = 'delta_pco2', target_known = True): 
    """ 
    @param      feat_list list of features in sampleDF to use
                ClusteredModVer: already trained ClusteredModelVersion object
                sampleDF: dataframe with dataset to get errors on 
                sample_tag: specify a string (default "val" returns DF with cols 'val_error')
                            only need if target_known = True
    """
    sampleDF = sampleDF.dropna(axis=0, subset=feat_list).copy()
    
    if 'mld' in feat_list:
        sampleDF['log_mld'] = np.log(sampleDF['mld'])
        feat_list = [x if x != 'mld' else 'log_mld' for x in feat_list]

    # k_list = ClusteredModVer.ind_list
    # skmodel = ClusteredModVer.models[k]

    pred_col = sample_tag + '_pred'
    sampleDF[pred_col] = skmodel.predict(sampleDF[feat_list].to_numpy())

    if target_known:
        sampleDF[sample_tag + '_error'] = sampleDF[pred_col] - sampleDF[target_var]
        sampleDF[sample_tag + '_relative_error'] = sampleDF[sample_tag + '_error'] / sampleDF[target_var]
    
    return sampleDF

# %%  Class dependent regression 
class ClusteredModelVersion:
    """ 
    Object holding different model runs
    @ param     ind_list = class indices e.g. range(1,9)
                models - scikit-learn objects
    """
    def __init__(self, ind_list, feat_list=None):
        self.ind_list = ind_list
        self.feat_list = feat_list
        self.models = {model: None for model in ind_list}
        self.params = {model: None for model in ind_list} # hyperparams by class
        self.weighted_training = None
        self.weighted_validation = None
        self.cal_coeffs = None

    def copy(self):
        return self

def apply_cluster_RFR(feat_list, ClusteredModVer, sampleDF, sample_tag='val', 
                      target_var = 'delta_pco2', 
                      sea_ice_clusters = None,
                      target_known = True): 
    """ 
    @param      feat_list list of features in sampleDF to use
                ClusteredModVer: already trained ClusteredModelVersion object
                sampleDF: dataframe with dataset to get errors on 
                sample_tag: specify a string (default "val" returns DF with cols 'val_error')
                            only need if target_known = True
                target_known: if False, target_var is not required in sampleDF (e.g. application data)
    """
    temp_feat_list = [x for x in feat_list if x != 'sea_ice']
    required_cols = temp_feat_list + [target_var] if target_known else temp_feat_list
    sampleDF = sampleDF.dropna(axis=0, subset=required_cols).copy()
    
    if 'mld' in feat_list: # should already be log_mld in feat_list, but catch any missed
        print('Warning: feature_list includes mld instead of log(mld), transforming...')
        sampleDF['log_mld'] = np.log(sampleDF['mld'])
        feat_list = [x if x != 'mld' else 'log_mld' for x in feat_list]

    k_list = ClusteredModVer.ind_list
    for k in k_list:
        if k not in sea_ice_clusters: 
            class_feat_list = [x for x in feat_list if x != 'sea_ice']
            # print('No sea ice in class ' + str(k) + ', removing from features for this class')
        else: class_feat_list = feat_list.copy()
        pred_col = 'cluster' + str(k) + '_pred'
        sampleDF[pred_col] = ClusteredModVer.models[k].predict(sampleDF[class_feat_list]) #.to_numpy())
    sampleDF['weighted_pred'] = sampleDF.apply(lambda row: weighted_prediction(row, k_list), axis=1)

    if target_known:
        sampleDF[sample_tag + '_error'] = sampleDF['weighted_pred'] - sampleDF[target_var]
        sampleDF[sample_tag + '_relative_error'] = sampleDF[sample_tag + '_error'] / sampleDF[target_var]
    
    return sampleDF


# %% Calibration function s
# originally from 3.2_kfold
from scipy import stats

def apply_clustered_calibration(cal_coeffs, valDF, n_clusters, target_var = 'delta_pco2', target_known = True):
    """ @param  target_known: if False, skip calibrated error columns (target_var not required) """
    
    # New version as of Mar 1 2026
    calibrated_valDF = valDF.copy()
    k_list = range(1, n_clusters+1)

    for ncluster in k_list:

        slope, intercept = cal_coeffs[ncluster]
        new_col = 'cluster' + str(ncluster) + '_calpred'
        calibrated_valDF[new_col] = calibrated_valDF['cluster' + str(ncluster) + '_pred'] * slope + intercept

    calibrated_valDF['lincal_pred'] = calibrated_valDF.apply(lambda row: weighted_prediction(row, k_list, pred_col_tag = '_calpred'),
                                                                      axis=1)

    if target_known:
        calibrated_valDF['calibrated_error'] = calibrated_valDF['lincal_pred'] - calibrated_valDF[target_var]
        calibrated_valDF['calibrated_relative_error'] = calibrated_valDF['calibrated_error'] / calibrated_valDF[target_var]

    return calibrated_valDF

def get_clustered_calibration_coeffs(valDF):
    cal_coeffs = {ncluster:None for ncluster in valDF['cluster'].unique()}
    for ncluster, class_valDF in valDF.groupby('cluster'):
        class_valDF['n_decile'] = pd.qcut(class_valDF['weighted_pred'], 10, labels=list(range(1, 11))) #
        cal_pred = class_valDF.groupby('n_decile', observed=True)['weighted_pred'].agg(['mean', 'min', 'max', 'count'])
        cal_obs = class_valDF.groupby('n_decile', observed=True)['delta_pco2'].agg(['mean', 'min', 'max', 'count'])

        stat_var = 'mean'
        lincal = stats.linregress(cal_pred[stat_var].values, cal_obs[stat_var].values)

        cal_coeffs[ncluster] = [lincal.slope, lincal.intercept]
    
    return cal_coeffs

def weighted_prediction(row, ind_list, pred_col_tag = '_pred'):
    """ 
    Make final predictions by weighting by class probabilities
    ind_list or k_list is range(1,n_gmm+1)

    @ param: pred_col_tag = '_pred' by default
                            for calibration, pass '_calpred'
                    
    """
    total = 0
    for k in ind_list:
        prob_col = 'cluster' + str(k) + '_prob'
        pred_col = 'cluster' + str(k) + pred_col_tag
        total += row[prob_col] * row[pred_col]
    return total

def summarize_errors(platDF, error_param = 'val_error', pd_format=False):
        """ 
        Summarize the chosen error metric with median absolute error, 
        mean absolute error, bias, and RMSE.

        :param platDF: pandas dataframe 
        :param error_param: string name of error column to summarize
        :param pd_format: boolean, if True returns results as a pandas dataframe, 
                        if False returns as a list, e.g. for storedRuns_comparison()
        
        """
        platDF = platDF.copy() #self.collapse_errors()

        err = platDF[error_param]
        median_abs_error = np.abs(err).median()
        mean_abs_error = np.abs(err).mean()
        bias = (err.mean())

        platDF[error_param + '_sq'] = platDF[error_param]**2
        mse = np.sum(platDF[error_param + '_sq']) / len(platDF[error_param])
        rmse = np.sqrt(mse)

        result = [median_abs_error, mean_abs_error, bias, rmse]
        if pd_format:
            result = pd.DataFrame(index=[''], data={'median_abs_error': [median_abs_error], 
                                                    'mean_abs_error': [mean_abs_error],
                                                    'bias': [bias], 'rmse': [rmse]})
        return result

# %% Comparing multiple runs 

def storedRuns_comparison(storedRuns, run_tags = None, error_param='val_error', 
                          target_var='delta_pco2',
                          show_platforms = ''): # float, ship, or combined
    """ 
    storedRuns: dictionary of ModelVersion objects, runtag as keys"""
    total_MAEs = pd.DataFrame()

    if run_tags is None: run_tags = [x for x in storedRuns.keys()]

    for run_tag in run_tags[:]:
        singleRun = storedRuns[run_tag]
        # print('==> Results for ' + run_tag)
        # print('\t features ', feat_options[run_tag.split('-')[0]])

        runResults = singleRun.weighted_validation.copy()
        [run_median_abs_error, run_mean_abs_error, run_bias, run_rmse] = summarize_DF_errors(runResults, error_param=error_param)

        total_MAEs.loc[run_tag, 'median_AE'] = run_median_abs_error
        total_MAEs.loc[run_tag, 'mean_AE'] = run_mean_abs_error
        total_MAEs.loc[run_tag, 'bias'] = run_bias
        total_MAEs.loc[run_tag, 'RMSE'] = run_rmse

    if show_platforms in ['float', 'ship', 'combined']:
        show_rows = [x for x in run_tags if show_platforms in x]
        total_MAEs = total_MAEs.loc[show_rows, :]

    # total_MAEs['labels'] = feat_list_labels
    # total_MAEs.set_index('labels', inplace=True)
    return total_MAEs


# %% Single run summaries (single feature list)

def singleRun_collapse_errors(singleRun):
    """ 
    note: used to be collapse_DF_errors() 
    Collapse DF_error from each class (within a singleRun) into single dataframe"""
    ind_list = [x for x in singleRun.ind_list]
    combined = pd.concat([singleRun.DF_err[x] for x in ind_list], axis=0)
    return combined

def singleRun_class_summary(singleRun, error_param = 'val_error', n_clusters=6):
    """ 
    Summarize error statistics for each class, and also overall
    """
    ind_list = [x for x in range(1, n_clusters+1)] # singleRun.ind_list]
    full_validation = singleRun.weighted_validation.copy()
    result = pd.DataFrame(index=ind_list, columns = ['median_AE', 'mean_AE', 'bias', 'rmse'])
    for ind in ind_list:
        # result.loc[ind, :] = summarize_DF_errors(singleRun.DF_err[ind]) # old
        class_error = full_validation[full_validation['cluster'] == ind].copy()
        result.loc[ind, :] = summarize_errors(class_error, error_param = error_param)

    # result.loc['overall', :] = summarize_DF_errors(singleRun_collapse_errors(singleRun))
    result.loc['overall', :] = summarize_errors(full_validation, error_param = error_param)
    return result

def singleRun_platform_summary(singleRun, error_param = 'val_error'):
    """ 
    note: used to be summarize_DF()
    Summarize error statistics for a single run, split by platform type
    
    runResults is the collapsed DF_err from a ModelVersion object
    run_tag = 'featD-combined-delta_pco2'
    runResults = singleRun_collapse_errors(storedRuns_withk4[run_tag])
    """

    if isinstance(singleRun, pd.DataFrame):
        runResults = singleRun.copy()
    else:
        runResults = singleRun.weighted_validation.copy() # singleRun_collapse_errors(singleRun)

    # print('Validation errors within single run: ')
    result = pd.DataFrame(index=['median_AE', 'mean_AE', 
                                # 'median_ape', 'mean_ape',
                                'bias', 'rmse'])
    result['overall'] = summarize_errors(runResults, error_param = error_param)

    [errors_float, errors_ship] = separate_platforms(runResults)
    result['float_component'] = summarize_errors(errors_float)
    result['ship_component'] = summarize_errors(errors_ship)

    return result.T

# %% Ancillary functions  


# def get_95_bounds(data):
#     mean = np.mean(data); std_dev = np.std(data)
#     low = mean - 2 * std_dev
#     high = mean + 2 * std_dev
#     print('95% bounds: \t' + str(low.round(5)) + ' to ' + str(high.round(5)) )
#     return [low, high]

def get_depth_bias(data, ranges, var='val_error'):
    """ Get validation errors in 100m depth bins. """
    # Example range to pass as @param: ranges
    # pressure_ranges = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500),
    #                 (500, 600), (600, 700), (700, 800), (800, 900), (900, 1000)]
    return {f"{start}-{end}": data[(data["pressure"] >= start) & (data["pressure"] < end)][var].values
            for start, end in ranges}

def print_errors_restrictdepth(data, var ='test_relative_error', pres_lim= [0,1000]):
    """ 
    @ param: data:  dataset that has "test_relative_error" in it
    """
    data = data[(data.pressure > pres_lim[0]) & (data.pressure < pres_lim[1])]
    err = data[var]
    print('Error metric: ' + var)
    print('Restricted to depths ' + str(pres_lim[0]) + ' to ' + str(pres_lim[1]) + ':')
    print('median abs error: \t' + str(np.abs(err).median()))
    print('mean abs error \t\t' + str(np.abs(err).mean()))

    # Bounds 95
    [low, high] = get_95_bounds(err)
    print('\n95% of errors fall between:')
    print(str(low.round(5)) + ' to ' + str(high.round(5)) )

    err = data[data.yearday <210][var]
    [low, high] = get_95_bounds(err)
    print("\nDuring SOGOS between depths " + str(pres_lim[0]) + ' to ' + str(pres_lim[1]) + ':')
    print('95% of errors fall between:')
    print(str(low.round(5)) + ' to ' + str(high.round(5)) )

