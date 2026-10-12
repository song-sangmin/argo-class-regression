import numpy as np
import pandas as pd
import xarray as xr
import scipy



def get_80_bounds(data):
    mean = np.mean(data); std_dev = np.std(data)
    low = mean - 1.282 * std_dev
    high = mean + 1.282 * std_dev
    print('80% bounds: \t' + str(low.round(5)) + ' to ' + str(high.round(5)) )
    return [low, high]


def get_95_bounds(data):
    mean = np.mean(data); std_dev = np.std(data)
    low = mean - 2 * std_dev
    high = mean + 2 * std_dev
    print('95% bounds: \t' + str(low.round(5)) + ' to ' + str(high.round(5)) )
    return [low, high]

def summarize_errors(platDF):
    err = platDF['val_error']
    median_abs_error = np.abs(err).median()
    mean_abs_error = np.abs(err).mean()
    bias = (err.mean())

    platDF['val_error_sq'] = platDF['val_error']**2
    mse = np.sum(platDF['val_error_sq']) / len(platDF.val_error)
    rmse = np.sqrt(mse)

    # absolute percentage error
    platDF['ape'] = np.abs(platDF['val_relative_error'])*100
    median_ape = platDF['ape'].median()
    mean_ape = platDF['ape'].mean()

    result = ([median_abs_error, mean_abs_error, 
            #   median_ape, mean_ape, 
              bias, rmse])
    return result