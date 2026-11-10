"""Print the nested CV and test errors of the latest model run, side by side, overall and by cluster.

Reads, from the results folder of --config ([files] model_results):
    nested CV:  the newest <run>_outercvErrors_acc<date>.csv (run_dask_nested_cv.py)
    test:       the newest <run>_testPredictions_acc<date>.csv (run_model_application.py)
For results kept elsewhere (e.g. copied back from a cluster), set [files] cv_results and/or
test_results in the config to their folders; each defaults to model_results. Then it prints n, RMSE, median absolute error and bias (prediction - truth) of the calibrated
prediction (lincal_pred), with the uncalibrated one (weighted_pred) in parentheses, by assigned
cluster. Both use summarize_cv_errors.error_metrics, so the CV and test columns compare directly.
--algorithm and --grid pick the CV run of that algorithm and grid (as `pixi run modeling` does).

Usage (from the repo root):
    python scripts/analysis/error_summary.py --config configs/example_drake.toml
    python scripts/analysis/error_summary.py --config configs/example_drake.toml --algorithm RFR --grid rfr_test
    python scripts/analysis/error_summary.py --cv-file <outercvErrors.csv> --test-file <testPredictions.csv>
"""

import argparse
import glob
import json
import os
import re

import pandas as pd
from summarize_cv_errors import PREDICTIONS, error_metrics

from argoclare import pipeline

METRICS = {'rmse': 'RMSE', 'medae': 'MedAE', 'bias': 'Bias'}


def newest(folder, pattern):
    """Newest file matching pattern in folder, by its acc<date> tag, then modification time."""

    def key(path):
        found = re.search(r'_acc(\d{8})', os.path.basename(path))
        return (found.group(1) if found else '', os.path.getmtime(path))

    files = glob.glob(os.path.join(folder, pattern))
    return max(files, key=key) if files else None


def results_folder(run, key):
    """[files] key (cv_results or test_results) if the config sets it, else [files] model_results."""
    files = run.section('files')
    pcm_files = files.get(run.pcm) if isinstance(files.get(run.pcm), dict) else {}
    return run.path(key if key in files or key in pcm_files else 'model_results')


def run_description(path):
    """KEY: value lines of the <run>_runDescription_acc<date>.txt next to an output file, if any."""
    description = re.sub(r'_(outercvErrors|testPredictions)_', '_runDescription_', path)
    description = os.path.splitext(description)[0] + '.txt'
    if not os.path.exists(description):
        return {}
    with open(description) as file:
        pairs = [line.split(':', 1) for line in file if re.match(r'^[A-Z_]+:', line)]
    return {key.strip(): value.strip() for key, value in pairs}


def group_metrics(data, target):
    """{group: {(metric, 'cal' or 'raw'): value, 'n': n}} for 'All' and each assigned cluster."""
    groups = {'All': data} | {int(classnum): group for classnum, group in data.groupby('cluster')}
    table = {}
    for key, group in groups.items():
        table[key] = {}
        for tag, col in PREDICTIONS.items():
            values = error_metrics(group[col], group[target])
            table[key]['n'] = values['n']
            table[key] |= {(metric, tag): values[metric] for metric in METRICS}
    return table


def fmt_cell(values, metric):
    """'cal (raw)', one decimal, no -0.0"""
    text = [f'{values[(metric, tag)]:.1f}' for tag in ('cal', 'raw')]
    text = ['0.0' if t == '-0.0' else t for t in text]
    return f'{text[0]} ({text[1]})'


def print_table(tables, target):
    """One row per group, a block of n + metrics for each of the tables (CV, test)."""
    keys = ['All'] + sorted({key for table in tables.values() for key in table if key != 'All'})
    header = ['cluster'] + [f'{name} {col}' for name in tables for col in ['n', *METRICS.values()]]
    rows = []
    for key in keys:
        row = [str(key)]
        for table in tables.values():
            if key in table and table[key]['n']:
                row += [f'{table[key]["n"]:,}'] + [fmt_cell(table[key], metric) for metric in METRICS]
            else:
                row += ['0'] + ['-'] * len(METRICS)
        rows.append(row)
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]

    def line(cells):
        return '  '.join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(cells, widths)))

    print(f'\nErrors in {target} (prediction - observation): calibrated (uncalibrated)\n')
    print(line(header))
    print(line(['-' * w for w in widths]))
    for i, row in enumerate(rows):
        print(line(row))
        if i == 0:
            print(line(['-' * w for w in widths]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--algorithm', default=None, help='Algorithm of the CV run to show (default: newest run)')
    parser.add_argument('--grid', default=None, help='Tuning grid of the CV run to show (default: newest run)')
    parser.add_argument('--cv-file', default=None, help='outercvErrors csv to use instead of the newest one')
    parser.add_argument('--test-file', default=None, help='testPredictions csv to use instead of the newest one')
    parser.add_argument('--target', default=None, help='Target column (default: from regression_info, with --config)')
    pipeline.add_model_args(parser)
    args = parser.parse_args()

    if args.algorithm:
        print(f'Showing results for {args.algorithm}...')
    cv_folder = test_folder = None
    target = args.target
    if args.config:
        run = pipeline.load(
            args.config,
            machine=args.machine,
            n_pca=args.n_pca,
            n_gmm=args.n_gmm,
            dbar_limit=args.dbar_limit,
            exclude_years=args.exclude_years,
        )
        cv_folder, test_folder = results_folder(run, 'cv_results'), results_folder(run, 'test_results')
        info_file = run.path('reg_info')
        if target is None and os.path.exists(info_file):
            with open(info_file) as file:
                target = json.load(file).get('target')
    elif not args.cv_file and not args.test_file:
        raise SystemExit('Pass --config, or --cv-file and/or --test-file')
    target = target or 'delta_pco2'

    cv_file, test_file = args.cv_file, args.test_file
    if cv_folder and not cv_file:
        algorithm = f'_{args.algorithm}_' if args.algorithm else '_'
        grid = f'_grid_{args.grid}' if args.grid else ''
        cv_file = newest(cv_folder, f'*{algorithm}*{grid}_outercvErrors_acc*.csv')
    if test_folder and not test_file:
        test_file = newest(test_folder, '*_testPredictions_acc*.csv')
    if not cv_file and not test_file:
        folders = cv_folder if cv_folder == test_folder else f'{cv_folder} or {test_folder}'
        raise SystemExit(
            f'No outercvErrors or testPredictions file in {folders}. '
            'For results copied elsewhere, set [files] cv_results and test_results in the config'
        )

    tables = {}
    if cv_file:
        description = run_description(cv_file)
        folds = f'{description.get("N_OUTER", "?")} outer x {description.get("N_INNER", "?")} inner'
        print(f'Nested CV: {cv_file}')
        print(f'    {description.get("ALGORITHM", "")} {description.get("CV_TYPE", "")} CV, {folds} folds')
        tables['CV'] = group_metrics(pd.read_csv(cv_file, index_col=0), target)
    else:
        print('Nested CV: no outercvErrors file' + (' for that algorithm and grid' if args.algorithm else ''))
    if test_file:
        test_data = pd.read_csv(test_file, index_col=0)
        years = ', '.join(str(y) for y in sorted(test_data['year'].unique())) if 'year' in test_data else '?'
        print(f'Test:      {test_file}')
        print(f'    final models on the test set, years {years}')
        tables['test'] = group_metrics(test_data, target)
    else:
        print('Test:      no testPredictions file (no test years, or --no-test)')
    print_table(tables, target)


if __name__ == '__main__':
    main()
