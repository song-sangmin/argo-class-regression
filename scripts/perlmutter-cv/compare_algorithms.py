"""Choose the best regression algorithm for each cluster from nested CV results.

Reads the outercvErrors files written by the runs that submit_algorithm_comparison.sh
submits (run names crusoe_<ALGORITHM>_<grid>_<tag>), and for each cluster k scores
that cluster's own model on its own rows: cluster{k}_pred vs delta_pco2 where cluster == k.
The best algorithm per cluster is the one with the lowest raw RMSE. Calibrated
(cluster{k}_calpred) errors are stored alongside for reference; the nested CV
calibration is fitted on the same validation rows, so they are partly in-sample.

Light pandas work, run on a login node once all runs have finished:
    module load python; conda activate dask-cv
    python compare_algorithms.py --tag cmp20261015

Outputs (in --outpath, default same as --savepath):
    <tag>_algorithmComparison.csv   per-cluster errors for every algorithm
    <tag>_overallComparison.csv     overall weighted_pred errors, each run vs the combined best choice
    <tag>_bestAlgorithms.json       {cluster: [algorithm, grid]}, for run_model_application.py --cluster-models
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

# ==== Standard run parameters =======
n_clusters = 6 # SO classes
var_predict = 'delta_pco2'

# Folder the nested CV runs save results to
default_savepath = '/pscratch/sd/s/sangsong/dask_cv/results/'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', required=True,
                        help='Comparison tag given to submit_algorithm_comparison.sh')
    parser.add_argument('--algorithms', nargs='+', default=['RFR', 'ERT', 'XGB'])
    parser.add_argument('--savepath', default=default_savepath,
                        help='Folder with the outercvErrors files')
    parser.add_argument('--outpath', default=None,
                        help='Folder to save comparison outputs (default: --savepath)')
    return parser.parse_args()


def find_run_file(savepath, algorithm, tag):
    """ Find the outercvErrors file for one algorithm, return (path, grid name).
    If the run was saved more than once, use the latest acc date. """
    pattern = re.compile(rf'^crusoe_{algorithm}_(.+)_{re.escape(tag)}_outercvErrors_acc(\d{{8}})\.csv$')
    matches = []
    for path in glob.glob(os.path.join(savepath, f'crusoe_{algorithm}_*_{tag}_outercvErrors_acc*.csv')):
        found = pattern.match(os.path.basename(path))
        if found:
            matches.append((found.group(2), found.group(1), path)) # (datetag, grid, path)

    if not matches:
        return None, None
    grids = {grid for _, grid, _ in matches}
    if len(grids) > 1:
        raise ValueError(f'{algorithm}: runs with tag {tag} found for more than one grid {sorted(grids)}')
    matches.sort()
    if len(matches) > 1:
        print(f'Warning: {len(matches)} {algorithm} files for tag {tag}, using latest: {matches[-1][2]}')
    datetag, grid, path = matches[-1]
    return path, grid


def error_stats(error, suffix=''):
    """ Summary statistics for an error series (prediction - truth) """
    return {'n' + suffix: int(error.notna().sum()),
            'rmse' + suffix: np.sqrt(np.mean(error**2)),
            'mae' + suffix: np.mean(np.abs(error)),
            'bias' + suffix: np.mean(error)}


def compare_clusters(run_data, run_grids):
    """ Per-cluster errors of each cluster's own model on its own rows, for each algorithm """
    rows = []
    for classnum in range(1, n_clusters + 1):
        for algorithm, data in run_data.items():
            class_data = data[data['cluster'] == classnum]
            raw_error = class_data['cluster' + str(classnum) + '_pred'] - class_data[var_predict]
            cal_error = class_data['cluster' + str(classnum) + '_calpred'] - class_data[var_predict]
            stats_cal = error_stats(cal_error, suffix='_cal')
            stats_cal.pop('n_cal')
            rows.append({'cluster': classnum, 'algorithm': algorithm, 'grid': run_grids[algorithm]}
                        | error_stats(raw_error) | stats_cal)
    comparison = pd.DataFrame(rows)

    # Best = lowest raw RMSE within each cluster
    best_idx = comparison.groupby('cluster')['rmse'].idxmin()
    best_cal_idx = comparison.groupby('cluster')['rmse_cal'].idxmin()
    comparison['best'] = comparison.index.isin(best_idx)
    comparison['best_by_cal'] = comparison.index.isin(best_cal_idx)
    return comparison


def compare_overall(run_data, best_algorithms):
    """ Overall weighted_pred and lincal_pred errors for each run and for the combined best choice.
    The combined prediction takes cluster{k}_pred from the best algorithm's run for each cluster.
    Every prediction is out-of-fold, so this works even if runs used different folds. """
    rows = [{'model': algorithm}
            | error_stats(data['weighted_pred'] - data[var_predict])
            | error_stats(data['lincal_pred'] - data[var_predict], suffix='_cal')
            for algorithm, data in run_data.items()]

    # Rows must line up across runs to build the combined prediction
    indexes = [data.index for data in run_data.values()]
    if any(index.has_duplicates for index in indexes):
        print('Warning: duplicate row labels in outercvErrors, skipping combined best-choice check')
        return pd.DataFrame(rows)
    common_index = indexes[0]
    for index in indexes[1:]:
        common_index = common_index.intersection(index)
    if any(len(index) != len(common_index) for index in indexes):
        print(f'Warning: runs have different rows, using the {len(common_index)} rows they share for the combined check')

    reference = next(iter(run_data.values())).loc[common_index] # target and cluster probs are the same in every run
    combined_pred = pd.Series(0.0, index=common_index)
    combined_calpred = pd.Series(0.0, index=common_index)
    for classnum, algorithm in best_algorithms.items():
        data = run_data[algorithm].loc[common_index]
        prob = reference['cluster' + str(classnum) + '_prob']
        combined_pred += prob * data['cluster' + str(classnum) + '_pred']
        combined_calpred += prob * data['cluster' + str(classnum) + '_calpred']

    rows.append({'model': 'combined_best'}
                | error_stats(combined_pred - reference[var_predict])
                | error_stats(combined_calpred - reference[var_predict], suffix='_cal'))
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    outpath = args.outpath or args.savepath
    os.makedirs(outpath, exist_ok=True)

    # Find all runs first, fail with the full list of missing ones
    run_files, run_grids, missing = {}, {}, []
    for algorithm in args.algorithms:
        path, grid = find_run_file(args.savepath, algorithm, args.tag)
        if path is None:
            missing.append(f'crusoe_{algorithm}_<grid>_{args.tag}')
        else:
            run_files[algorithm], run_grids[algorithm] = path, grid
    if missing:
        raise FileNotFoundError(f'No outercvErrors file in {args.savepath} for: {", ".join(missing)}. '
                                'Check the jobs finished (sacct) before comparing.')

    run_data = {}
    for algorithm, path in run_files.items():
        run_data[algorithm] = pd.read_csv(path, index_col=0)
        print(f'Loaded {algorithm} ({run_grids[algorithm]}): {len(run_data[algorithm])} rows from {path}')

    comparison = compare_clusters(run_data, run_grids)
    best = comparison[comparison['best']].set_index('cluster')
    best_algorithms = best['algorithm'].to_dict()
    overall = compare_overall(run_data, best_algorithms)

    print('\nPer-cluster errors (cluster{k}_pred on cluster k rows; *_cal uses cluster{k}_calpred):')
    print(comparison.round(3).to_string(index=False))
    print('\nOverall errors (weighted_pred; *_cal uses lincal_pred):')
    print(overall.round(3).to_string(index=False))

    print('\nBest algorithm by cluster (lowest raw RMSE):')
    for classnum, row in best.iterrows():
        print(f'  Cluster {classnum}: {row["algorithm"]} ({row["grid"]}), RMSE {row["rmse"]:.3f}')
    best_cal = comparison[comparison['best_by_cal']].set_index('cluster')['algorithm']
    for classnum, algorithm in best_algorithms.items():
        if best_cal[classnum] != algorithm:
            print(f'Warning: cluster {classnum} would pick {best_cal[classnum]} by calibrated RMSE instead of {algorithm}')

    savefiles = {'per-cluster comparison': os.path.join(outpath, args.tag + '_algorithmComparison.csv'),
                 'overall comparison': os.path.join(outpath, args.tag + '_overallComparison.csv'),
                 'best algorithms': os.path.join(outpath, args.tag + '_bestAlgorithms.json')}
    comparison.to_csv(savefiles['per-cluster comparison'], index=False)
    overall.to_csv(savefiles['overall comparison'], index=False)
    with open(savefiles['best algorithms'], 'w') as file:
        json.dump({str(classnum): [row['algorithm'], row['grid']] for classnum, row in best.iterrows()},
                  file, indent=2)

    print()
    for label, savefile in savefiles.items():
        print('Saved ' + label + ' to ' + savefile)


if __name__ == "__main__":
    main()
