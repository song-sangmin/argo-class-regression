"""Print a summary table of error metrics for the nested CV run(s) in a folder.

Reads every *_outercvErrors_*.csv in the folder (written by run_dask_nested_cv.py) and prints,
for each run, the out-of-fold errors (prediction - truth) of
    raw:        weighted_pred (cluster-probability weighted prediction)
    calibrated: lincal_pred (cluster linear calibration)
as n, RMSE, MAE, median absolute error, bias and R2:
    1. overall, with the spread of RMSE across the outer folds
    2. by assigned cluster, by platform, and by gmm_cluster if the file has it (the GMM class
       kept by regression-trainval-no-clustering.py, to compare a single-cluster run with the
       clustered one class by class)
Errors by cluster here use weighted_pred; compare_algorithms.py instead uses each cluster's own
cluster{k}_pred on its rows, so its per-cluster numbers differ slightly.
With --lat-range, only samples with latitude in [south, north] (degrees, negative south,
inclusive) are summarized, e.g. --lat-range -70 -35 for 70S to 35S.

Usage:
    python summarize_cv_errors.py /path/to/argoclare-runs/RFR_realworld_noclusters
    python summarize_cv_errors.py /path/to/run_folder --by cluster platform year --save
    python summarize_cv_errors.py /path/to/run_folder --lat-range -70 -35
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

PREDICTIONS = {'raw': 'weighted_pred', 'cal': 'lincal_pred'}


def error_metrics(pred, truth):
    """n, RMSE, MAE, median absolute error, bias and R2 of pred against truth (rows with both)."""
    ok = pred.notna() & truth.notna()
    pred, truth = pred[ok], truth[ok]
    error = pred - truth
    ss_tot = np.sum((truth - truth.mean()) ** 2)
    return {
        'n': int(ok.sum()),
        'rmse': np.sqrt(np.mean(error**2)),
        'mae': np.mean(np.abs(error)),
        'medae': np.median(np.abs(error)),
        'bias': np.mean(error),
        'r2': 1 - np.sum(error**2) / ss_tot if ss_tot > 0 else np.nan,
    }


def summary_row(data, target):
    """Raw and calibrated metrics for one group of rows, as one flat dict."""
    row = {}
    for tag, col in PREDICTIONS.items():
        if col not in data.columns:
            continue
        metrics = error_metrics(data[col], data[target])
        if tag == 'cal':
            metrics.pop('n')
        row |= {(k if tag == 'raw' else f'{k}_cal'): v for k, v in metrics.items()}
    return row


def fold_spread(data, target):
    """Mean and standard deviation across outer folds of the per-fold RMSE."""
    spread = {}
    for tag, col in PREDICTIONS.items():
        if col not in data.columns or 'fold' not in data.columns:
            continue
        fold_rmse = data.groupby('fold').apply(lambda d: error_metrics(d[col], d[target])['rmse'])
        suffix = '' if tag == 'raw' else '_cal'
        spread |= {f'fold_rmse_mean{suffix}': fold_rmse.mean(), f'fold_rmse_std{suffix}': fold_rmse.std()}
    return spread


def summarize(data, target, by):
    """Overall table (one row) and one table per grouping column."""
    overall = pd.DataFrame([summary_row(data, target) | fold_spread(data, target)], index=['all'])
    groups = {
        col: pd.DataFrame({key: summary_row(group, target) for key, group in data.groupby(col)}).T.rename_axis(col)
        for col in by
        if col in data.columns
    }
    for table in groups.values():
        table['n'] = table['n'].astype(int)
    return overall, groups


def run_name(path):
    """Run name from the file name: <run>_outercvErrors_acc<date>.csv -> <run>."""
    return os.path.basename(path).split('_outercvErrors')[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folder', help='Folder with the *_outercvErrors_*.csv file(s)')
    parser.add_argument('--target', default='delta_pco2', help='Target column (default: delta_pco2)')
    parser.add_argument(
        '--by',
        nargs='+',
        default=['cluster', 'gmm_cluster', 'platform'],
        help='Columns to break the errors down by, when present (default: cluster gmm_cluster platform)',
    )
    parser.add_argument(
        '--save', action='store_true', help='Also save the tables to <run>_errorSummary.csv in the folder'
    )
    parser.add_argument(
        '--lat-range',
        type=float,
        nargs=2,
        default=None,
        metavar=('SOUTH', 'NORTH'),
        help='Only samples with latitude in [SOUTH, NORTH], degrees with south negative (e.g. -70 -35)',
    )
    args = parser.parse_args()
    if args.lat_range:
        south, north = sorted(args.lat_range)
        lat_tag = f'_lat{abs(south):g}{"S" if south < 0 else "N"}-{abs(north):g}{"S" if north < 0 else "N"}'

    files = sorted(glob.glob(os.path.join(args.folder, '*_outercvErrors_*.csv')))
    if not files:
        raise SystemExit(f'No *_outercvErrors_*.csv file in {args.folder}')

    for path in files:
        data = pd.read_csv(path, index_col=0)
        if args.target not in data.columns:
            raise SystemExit(f'{os.path.basename(path)} has no {args.target} column (pass --target)')
        n_all = len(data)
        if args.lat_range:
            data = data[data['latitude'].between(south, north)]
        by = [col for col in args.by if col in data.columns and data[col].nunique() > 1]
        overall, groups = summarize(data, args.target, by)

        n_folds = data['fold'].nunique() if 'fold' in data.columns else 'unknown'
        print(f'\n===== {run_name(path)} =====')
        print(f'{os.path.basename(path)}: {len(data)} samples, {n_folds} outer folds, target {args.target}')
        if args.lat_range:
            print(f'Latitude {south:g} to {north:g}: {len(data)} of {n_all} samples')
        print('Errors are prediction - truth; raw: weighted_pred, *_cal: lincal_pred\n')
        print('Overall:')
        print(overall.round(3).to_string())
        for col, table in groups.items():
            print(f'\nBy {col}:')
            print(table.round(3).to_string())

        if args.save:
            tables = [overall.rename_axis('group').reset_index().assign(by='all')] + [
                table.rename_axis('group').reset_index().assign(by=col) for col, table in groups.items()
            ]
            savefile = os.path.join(
                args.folder, run_name(path) + '_errorSummary' + (lat_tag if args.lat_range else '') + '.csv'
            )
            summary = pd.concat(tables, ignore_index=True)
            summary[['by', 'group'] + [c for c in summary.columns if c not in ('by', 'group')]].to_csv(
                savefile, index=False
            )
            print(f'\nSaved to {savefile}')


if __name__ == '__main__':
    main()
