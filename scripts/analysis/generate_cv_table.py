"""Save a LaTeX table of outer fold validation errors by cluster for each algorithm in a folder.

Reads the crusoe_<ALGORITHM>_*_outercvErrors_*.csv files in the folder (written by
run_dask_nested_cv.py) and writes a sidewaystable with RMSE, median absolute error and bias
(prediction - truth) by assigned cluster and overall, one column per algorithm. Each cell is the
calibrated prediction (lincal_pred) with the uncalibrated one (weighted_pred) in parentheses,
the same errors as summarize_cv_errors.py.

Usage:
    python generate_cv_table.py /path/to/argoclare-runs/cmp20261109_pK014
    python generate_cv_table.py /path/to/run_folder --algorithms RFR ERT XGB MLR --out table.tex
"""

import argparse
import glob
import os

import pandas as pd
from summarize_cv_errors import PREDICTIONS, error_metrics, run_name

METRICS = {'rmse': 'RMSE', 'medae': 'MedAE', 'bias': 'Bias'}


def find_runs(folder, algorithms):
    """{algorithm: outercvErrors path}, from file names crusoe_<ALGORITHM>_<grid>_<tag>_outercvErrors_*.csv"""
    files = {}
    for path in sorted(glob.glob(os.path.join(folder, '*_outercvErrors_*.csv'))):
        algorithm = run_name(path).split('_')[1]
        if algorithm not in algorithms:
            continue
        if algorithm in files:
            raise SystemExit(f'More than one {algorithm} run in {folder}: {files[algorithm]} and {path}')
        files[algorithm] = path
    missing = [algorithm for algorithm in algorithms if algorithm not in files]
    if missing:
        raise SystemExit(f'No outercvErrors file in {folder} for: {", ".join(missing)}')
    return files


def group_metrics(data, target):
    """{group: {(metric, 'cal' or 'raw'): value}} by cluster plus 'All', and n per group."""
    groups = {classnum: group for classnum, group in data.groupby('cluster')} | {'All': data}
    metrics, counts = {}, {}
    for key, group in groups.items():
        metrics[key] = {}
        for tag, col in PREDICTIONS.items():
            values = error_metrics(group[col], group[target])
            counts[key] = values['n']
            metrics[key] |= {(metric, tag): values[metric] for metric in METRICS}
    return metrics, counts


def fmt_number(value):
    """One decimal, LaTeX minus sign, no -0.0"""
    text = f'{value:.1f}'
    if text == '-0.0':
        text = '0.0'
    return text.replace('-', '$-$')


def fmt_cell(values, metric):
    return f'{fmt_number(values[(metric, "cal")])} ({fmt_number(values[(metric, "raw")])})'


def join_names(names):
    """'A', 'A and B', 'A, B, and C'"""
    if len(names) <= 2:
        return ' and '.join(names)
    return ', '.join(names[:-1]) + ', and ' + names[-1]


def latex_table(results, counts, algorithms, label):
    n_alg = len(algorithms)
    blocks = ['c' * n_alg] * len(METRICS)
    header = ' & '.join(
        f'\\multicolumn{{{n_alg}}}{{c{"|" if i < len(METRICS) - 1 else ""}}}{{{name}}}'
        for i, name in enumerate(METRICS.values())
    )
    clines = ' '.join(f'\\cline{{{3 + i * n_alg}-{2 + (i + 1) * n_alg}}}' for i in range(len(METRICS)))
    caption = (
        f'Outer fold validation errors ($\\mu$atm) from {join_names(algorithms)} by cluster: '
        'root mean square error (RMSE), median absolute error (MedAE) and mean bias '
        '(prediction $-$ observation). Values are for the linearly calibrated prediction, '
        'with the uncalibrated prediction in parentheses.'
    )

    def row(key):
        cells = [fmt_cell(results[algorithm][key], metric) for metric in METRICS for algorithm in algorithms]
        return ' & '.join([str(key), f'{counts[key]:,}'.replace(',', '{,}')] + cells) + ' \\\\'

    clusters = [key for key in counts if key != 'All']
    lines = [
        '\\begin{sidewaystable}',
        f'\\caption{{{caption}}}',
        f'\\label{{{label}}}',
        '\\centering',
        f'\\begin{{tabular}}{{lr{"|".join(blocks)}}}',
        '\\hline',
        f' & & {header} \\\\',
        clines,
        ' & '.join(['Cluster', '$n$'] + algorithms * len(METRICS)) + ' \\\\',
        '\\hline',
        *[row(key) for key in clusters],
        '\\hline',
        row('All'),
        '\\hline',
        '\\end{tabular}',
        '\\end{sidewaystable}',
    ]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folder', help='Folder with the *_outercvErrors_*.csv files')
    parser.add_argument(
        '--algorithms',
        nargs='+',
        default=['RFR', 'ERT', 'XGB'],
        help='Algorithms in column order (default: RFR ERT XGB)',
    )
    parser.add_argument('--target', default='delta_pco2', help='Target column (default: delta_pco2)')
    parser.add_argument('--label', default='tab:cverror_algorithms', help='LaTeX label')
    parser.add_argument('--out', default=None, help='Output .tex file (default: <folder>/<folder name>_cvTable.tex)')
    args = parser.parse_args()

    folder = os.path.normpath(args.folder)
    results, counts = {}, None
    for algorithm, path in find_runs(folder, args.algorithms).items():
        data = pd.read_csv(path, index_col=0)
        results[algorithm], run_counts = group_metrics(data, args.target)
        if counts is not None and run_counts != counts:
            print(f'Warning: {algorithm} has different sample counts per cluster; n column uses the first run')
        counts = counts or run_counts
        print(f'Loaded {algorithm}: {len(data)} rows from {os.path.basename(path)}')

    table = latex_table(results, counts, args.algorithms, args.label)
    out = args.out or os.path.join(folder, os.path.basename(folder) + '_cvTable.tex')
    with open(out, 'w') as file:
        file.write(table)
    print(f'\n{table}\nSaved to {out}')


if __name__ == '__main__':
    main()
