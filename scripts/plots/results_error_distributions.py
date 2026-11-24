"""Distribution of the nested CV errors of a run: (a) histogram, (b) box plots by cluster.

Reads the newest nested CV (outercvErrors) file of --config, as analysis/error_summary.py does,
and plots the calibrated errors (lincal_pred - target) of every sample (all levels of a
depth-resolved run):
    (a) histogram line of the errors, with n, RMSE, bias and median absolute error; bins span the
        0.5-99.5th percentiles (--bin-width wide, default: Freedman-Diaconis)
    (b) box plots of the errors by assigned cluster, plus all clusters together; boxes are the
        quartiles, whiskers 1.5 IQR, outliers not shown
Saves <run_dir>/plots/error_distributions_<pcm>_<CV run>.png (or --out).

Usage (from the repo root):
    python scripts/plots/results_error_distributions.py --config configs/example_drake.toml
    python scripts/plots/results_error_distributions.py --config configs/example_drake.toml --algorithm RFR --grid rfr_test
"""

import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.colors as mpcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from results_errors_by_depth import (  # noqa: E402
    COLORS,
    UNITS,
    add_results_args,
    cv_run_name,
    load_results,
    save,
    set_labels,
)
from summarize_cv_errors import error_metrics  # noqa: E402

OUT_NAME = 'error_distributions'


def histogram_line(ax, errors, color, bin_width=None, lw=1.5):
    """Step histogram of the errors over their 0.5-99.5th percentiles; returns the bin edges."""
    low, high = np.percentile(errors, [0.5, 99.5])
    if bin_width:
        edges = np.arange(np.floor(low / bin_width) * bin_width, high + bin_width, bin_width)
    else:
        edges = np.histogram_bin_edges(errors, bins='fd', range=(low, high))
    ax.hist(errors, bins=edges, histtype='step', color=color, linewidth=lw, zorder=2)
    return edges


def boxplot_by_cluster(ax, groups, color, width=0.55, lw=1.5):
    """Vertical box plots of {label: errors}, one per x position, in one color."""
    style = {'color': color, 'linewidth': lw}  # copied for each: matplotlib edits boxprops in place
    bplot = ax.boxplot(
        list(groups.values()),
        widths=width,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bin-width', type=float, default=None, help='Histogram bin width (default: automatic)')
    add_results_args(parser, OUT_NAME)
    args = parser.parse_args()
    run, target, data = load_results(args, names=('CV',))
    cv = data['CV']
    units = args.units or UNITS.get(target, '')
    color = COLORS['CV']

    fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(10, 4.2), width_ratios=[1.2, 1])

    edges = histogram_line(ax_hist, cv['error'], color, args.bin_width)
    metrics = error_metrics(cv['lincal_pred'], cv[target])
    outside = int(((cv['error'] < edges[0]) | (cv['error'] > edges[-1])).sum())
    ax_hist.text(
        0.97,
        0.95,
        f'n = {metrics["n"]:,}\nRMSE {metrics["rmse"]:.1f}\nbias {metrics["bias"]:.1f}\n'
        f'MedAE {metrics["medae"]:.1f}\n{outside:,} outside the axis',
        transform=ax_hist.transAxes,
        ha='right',
        va='top',
        fontsize=8,
    )
    ax_hist.axvline(0, color='k', linestyle='dotted', linewidth=1.5, alpha=0.7, zorder=0)
    ax_hist.set_xlabel(f'{target} error [{units}]')
    ax_hist.set_ylabel('Number of samples')
    ax_hist.grid(alpha=0.4, zorder=1)
    ax_hist.set_title('(a)', loc='left', fontsize=10)

    clusters = sorted(cv['cluster'].dropna().unique())
    groups = {'All': cv['error'].values} | {f'{int(c)}': cv.loc[cv['cluster'] == c, 'error'].values for c in clusters}
    boxplot_by_cluster(ax_box, groups, color)
    ax_box.set_xticks(
        range(1, len(groups) + 1), [f'{label}\n(n = {len(values):,})' for label, values in groups.items()], fontsize=8
    )
    ax_box.set_xlabel('Cluster')
    ax_box.set_ylabel(f'{target} error [{units}]')
    ax_box.axhline(0, color='k', linestyle='dotted', linewidth=1.5, alpha=0.7, zorder=0)
    ax_box.grid(axis='y', alpha=0.4, zorder=1)
    ax_box.set_title('(b)', loc='left', fontsize=10)

    fig.suptitle(f'{run.name} {run.pcm}, {set_labels(data)["CV"]}: calibrated prediction - observation', fontsize=9)
    fig.tight_layout()
    save(fig, run, args, OUT_NAME, cv_run_name(run, data))


if __name__ == '__main__':
    main()
