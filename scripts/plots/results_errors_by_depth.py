"""Box plots of the model errors by depth (100 dbar bins) for a depth-resolved run.

Reads the newest nested CV (outercvErrors) and test (testPredictions) files of --config, as
analysis/error_summary.py does, and plots the calibrated errors (lincal_pred - target) of each,
binned by pressure: one CV box and one test box per bin, depth increasing downwards. Boxes are the
quartiles, whiskers 1.5 IQR, outliers not shown; the legend gives n and RMSE of each set. Bins
include their top edge, and the deepest bin also its bottom (e.g. 900-1000 holds 1000 dbar).
Saves <run_dir>/plots/errors_by_depth_<pcm>_<CV run>.png (or --out).

Usage (from the repo root):
    python scripts/plots/results_errors_by_depth.py --config configs/example_drake.toml
    python scripts/plots/results_errors_by_depth.py --config configs/example_drake.toml --algorithm RFR --grid rfr_test
"""

import argparse
import json
import math
import os
import re
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.colors as mpcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'analysis'))
from error_summary import is_depth_resolved, newest, results_folder, run_description  # noqa: E402
from summarize_cv_errors import error_metrics  # noqa: E402

from argoclare import pipeline  # noqa: E402

UNITS = {'oxygen': 'µmol/kg', 'nitrate': 'µmol/kg', 'delta_pco2': 'µatm', 'pco2': 'µatm'}
COLORS = {'CV': 'C0', 'test': 'C3'}


def get_depth_bias(data, ranges, var='error'):
    """Values of var in each pressure range {'start-end': array}; ranges are (start, end, last)
    tuples, each holding start <= pressure < end, or <= end for the last one."""
    return {
        f'{start:g}-{end:g}': data[
            (data['pressure'] >= start) & ((data['pressure'] <= end) if last else (data['pressure'] < end))
        ][var].values
        for start, end, last in ranges
    }


def depth_ranges(pmax, size):
    """(start, end, last) pressure ranges of `size` dbar from 0 to pmax (rounded up to size)."""
    n = max(1, math.ceil(pmax / size))
    return [(i * size, (i + 1) * size, i == n - 1) for i in range(n)]


def boxplot_depth_binned(ax, binned, positions, color, width=0.35, lw=1.5):
    """Horizontal box plots of the binned errors at the given y positions, in one color."""
    style = {'color': color, 'linewidth': lw}  # copied for each: matplotlib edits boxprops in place
    bplot = ax.boxplot(
        list(binned.values()),
        positions=positions,
        widths=width,
        orientation='horizontal',
        showfliers=False,
        patch_artist=True,
        medianprops=dict(style),
        capprops=dict(style),
        whiskerprops=dict(style),
        boxprops=dict(style),
        zorder=2,
    )
    for patch in bplot['boxes']:
        patch.set_facecolor(mpcolors.to_rgba(color, alpha=0.4))
    return bplot


def add_results_args(parser, out_name):
    """--algorithm, --grid, --units, --out and the config options of the results plots."""
    parser.add_argument('--algorithm', default=None, help='Algorithm of the CV run to plot (default: newest run)')
    parser.add_argument('--grid', default=None, help='Tuning grid of the CV run to plot (default: newest run)')
    parser.add_argument('--units', default=None, help='Error units for the axis label (default: from the target)')
    parser.add_argument('--out', default=None, help=f'Output png (default: <run_dir>/plots/{out_name}_*.png)')
    return pipeline.add_model_args(parser)


def load_results(args, names=('CV', 'test')):
    """The run, target and {name: results} of --config for names CV (newest outercvErrors of
    --algorithm/--grid) and test (newest testPredictions), as analysis/error_summary.py finds them,
    each with an error column (lincal_pred - target); names without a file are left out."""
    if not args.config:
        raise SystemExit('Pass --config')
    run = pipeline.load(
        args.config,
        machine=args.machine,
        n_pca=args.n_pca,
        n_gmm=args.n_gmm,
        dbar_limit=args.dbar_limit,
        exclude_years=args.exclude_years,
    )
    if not os.path.exists(run.path('reg_info')):
        raise SystemExit(
            f'No regression inputs for {run.pcm} ({run.path("reg_info")}): run regression prep and the models first'
        )
    with open(run.path('reg_info')) as file:
        target = json.load(file)['target']
    algorithm = f'_{args.algorithm}_' if args.algorithm else '_'
    grid = f'_grid_{args.grid}' if args.grid else ''
    files = {
        'CV': newest(results_folder(run, 'cv_results'), f'*{algorithm}*{grid}_outercvErrors_acc*.csv'),
        'test': newest(results_folder(run, 'test_results'), '*_testPredictions_acc*.csv'),
    }
    files = {name: files[name] for name in names if files[name]}
    if not files:
        raise SystemExit(f'No {" or ".join(names)} results in {run.path("model_results")}')
    data = {}
    for name, path in files.items():
        print(f'{name}: {path}')
        df = pd.read_csv(path, index_col=0)
        data[name] = df.assign(error=df['lincal_pred'] - df[target]).dropna(subset=['error'])
        data[name].attrs['file'] = path
    return run, target, data


def set_labels(data):
    """Legend names of the result sets: CV (<cv_type> <n>-fold), test (<years>)."""
    labels = {}
    if 'CV' in data:
        description = run_description(data['CV'].attrs['file'])
        labels['CV'] = f'CV ({description.get("CV_TYPE", "")} {description.get("N_OUTER", "?")}-fold)'
    if 'test' in data:
        years = sorted(data['test']['year'].unique()) if 'year' in data['test'] else []
        labels['test'] = 'test' + (f' ({", ".join(str(y) for y in years)})' if years else '')
    return labels


def cv_run_name(run, data):
    """Name of the CV run for the figure file name, without the results prefix (the run name is
    already in the folder and title); 'test' without CV results."""
    if 'CV' not in data:
        return 'test'
    name = re.sub(r'_outercvErrors_acc\d{8}\.csv$', '', os.path.basename(data['CV'].attrs['file']))
    for prefix in (run.results_prefix, 'crusoe'):
        name = name.removeprefix(f'{prefix}_')
    return name


def save(fig, run, args, out_name, cv_run):
    out = args.out or os.path.join(run.run_dir, 'plots', f'{out_name}_{run.pcm}_{cv_run}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bin-size', type=float, default=100, help='Pressure bin size (dbar, default 100)')
    add_results_args(parser, 'errors_by_depth')
    args = parser.parse_args()
    run, target, data = load_results(args)
    if not all(is_depth_resolved(df) for df in data.values()):
        raise SystemExit('Not a depth-resolved run: the results have no pressure levels to bin')

    ranges = depth_ranges(max(df['pressure'].max() for df in data.values()), args.bin_size)
    fig, ax = plt.subplots(figsize=(5, 1 + 0.55 * len(ranges)))
    offsets = np.linspace(-0.2, 0.2, len(data)) if len(data) > 1 else [0]
    names = set_labels(data)
    handles = []
    for (name, df), offset in zip(data.items(), offsets):
        binned = get_depth_bias(df, ranges)
        bplot = boxplot_depth_binned(ax, binned, np.arange(1, len(ranges) + 1) + offset, COLORS[name])
        metrics = error_metrics(df['lincal_pred'], df[target])
        label = f'{names[name]}: n = {metrics["n"]:,}, RMSE {metrics["rmse"]:.1f}'
        handles.append((bplot['boxes'][0], label))
    labels = [f'{start:g}-{end:g}' for start, end, _ in ranges]
    ax.set_yticks(range(1, len(labels) + 1), labels)
    ax.set_ylim(len(labels) + 0.6, 0.4)  # depth increasing downwards
    ax.set_ylabel('Pressure [dbar]')
    ax.set_xlabel(f'{target} error [{args.units or UNITS.get(target, "")}]')
    ax.axvline(0, color='k', linestyle='dotted', linewidth=1.5, alpha=0.7, zorder=0)
    ax.grid(axis='x', zorder=1, alpha=0.4)
    ax.legend(*zip(*handles), loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=8, frameon=False)
    ax.set_title(f'{run.name} \ncalibrated prediction - observation', fontsize=9)
    fig.tight_layout()
    save(fig, run, args, 'errors_by_depth', cv_run_name(run, data))


if __name__ == '__main__':
    main()
