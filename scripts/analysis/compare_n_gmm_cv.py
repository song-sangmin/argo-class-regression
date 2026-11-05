"""Cross-validation errors of the clustered RFR for different numbers of GMM classes.

A quick check of whether trainval errors improve with the number of classes (2-10, as in the
BIC/AIC/silhouette comparison), without rerunning the P2 and regression steps for each one:
    1. Clustering: refit the GMM only, on PC1..PC8 of the existing pc8_gmm6_dbar500 PCM_finder.
       The PCA (8 PCs, CT/SA at 0-500 dbar, same centering and scaling) does not depend on the
       number of classes, so this is the fit fit_argo_clustering.py --n-gmm k would make
       (diagonal covariance, random_state 0, max_iter 10000; classes renumbered north (1) to south).
    2. Classes for the trainval samples: each sample takes the class probabilities of the same
       profile it got its pc8_gmm6 classes from (class_profid in the P2 trainval file). The 45
       samples gap-filled from a neighbouring profile keep that profile; apply_P2_classes.py might
       gap-fill a few of them differently (it requires the neighbours to agree on the class).
    3. Sea ice clusters: classes with more than 10 float samples with sea_ice > 0.1, as in
       regression-input-prep.py.
    4. Platform CV (10 outer folds by default, drawn within each class as in model-optimization),
       one RandomForestRegressor per class with sklearn's default settings (no tuning) and no
       calibration. Errors are of weighted_pred (class probability weighted), as in
       summarize_cv_errors.py. k = 1 (no clustering) is included as a baseline.
The trainval samples, features and target are those of the pc8_gmm6 regression csv, so every k
is scored on the same 17,461 samples.
Another PCA (e.g. 0-900 dbar): pass its PCM_finder and P2 files (from fit_argo_clustering.py and
apply_P2_classes.py --datasets soccom socat with the same --dbar-limit). Samples without a class in
that fit are left out; --keep-samples <other run's n_gmm_cv_predictions.csv> scores a run on the
samples of another, so two fits are compared on the same samples. Synthetic data: pass the synthetic
PCM_finder, P2_syn_trainval and regression csv, with --drop-features adt; for another dbar limit, the
P2_syn_trainval of the 500 dbar fit keeps the same (subsampled) samples, each with its own profile. The folds differ between k (they are drawn per class), so
it runs 5 fold seeds by default, to see how much of a difference is fold noise. About 5 s per k and seed.

Writes to --out-dir:
    n_gmm_cv_summary.csv      one row per k (and seed): RMSE, MAE, median abs error, bias, R2,
                              fold RMSE spread, RMSE over 35-70S, sea ice clusters, class sizes
    n_gmm_cv_summary_by_k.csv mean over seeds per k, with the RMSE difference from k = 1 paired by seed
    n_gmm_cv_predictions.csv  out-of-fold weighted_pred for every k and seed, per sample
    n_gmm_cv_rmse.png         RMSE vs number of classes

Usage:
    python compare_n_gmm_cv.py
    python compare_n_gmm_cv.py --n-gmm 2 4 6 8 10 --seeds 42
    P=.../P2-clustered/pc8_gmm6_dbar900
    python compare_n_gmm_cv.py --finder $P/PCM_finder_pc8_gmm6_dbar900.parquet \
        --p2-trainval $P/P2_soccom_pc8_gmm6_dbar900.parquet $P/P2_socat_pc8_gmm6_dbar900.parquet --out-dir .../n_gmm_check_dbar900
"""

import argparse
import os
from datetime import datetime

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

from argoclare import crossvalidation  # noqa: E402

DATA = '/Users/sangminsong/Documents/PROJECTS-PYTHON/CRUSOE-SOCarbon/x_localdata'
P2_DIR = f'{DATA}/working-vars/P2-clustered/pc8_gmm6_dbar500'
DEFAULTS = {
    'finder': f'{P2_DIR}/PCM_finder_pc8_gmm6_dbar500_acc20261016.parquet',
    'p2_trainval': f'{P2_DIR}/P2_combined_trainval_socat_soccom_2014-2023_acc20261016.parquet',
    'reg_trainval': f'{DATA}/working-vars/regression-inputs/combined_trainval_pc8_gmm6_dbar500_seaiceclusters56_acc20261016.csv',
    'out_dir': f'{DATA}/argoclare-runs/n_gmm_check',
}
FEATURES = ['CT', 'SA', 'log_mld', 'adt', 'sea_ice', 'wind_speed', 'atmos_pres_atm', 'max_solar_rad']
TARGET = 'delta_pco2'
FLOAT_PLATFORMS = ['bgcArgo', 'synBgcArgo']  # as in regression-input-prep.py
ICE_THRESHOLD, MIN_ICE_PROFILES = 0.1, 10


def fit_classes(pcs, latitude, n_gmm):
    """GMM class probabilities (profid x n_gmm) numbered by mean latitude of assigned profiles, north first."""
    gmm = GaussianMixture(n_components=n_gmm, covariance_type='diag', max_iter=10000, random_state=0).fit(pcs)
    labels, probs = gmm.predict(pcs), gmm.predict_proba(pcs)
    order = pd.Series(latitude).groupby(labels).mean().sort_values(ascending=False).index.values
    return pd.DataFrame(probs[:, order], index=pcs.index, columns=[f'cluster{k}_prob' for k in range(1, n_gmm + 1)])


def sample_profiles(p2_trainval, reg):
    """class_profid of each regression sample, matched on the regression csv's sample_key
    (platform_id + sample time to the minute, as in regression-input-prep.py)."""
    p2 = p2_trainval.dropna(subset=FEATURES + [TARGET])
    key = p2['platform_id'].astype(str) + '_' + pd.to_datetime(p2['datetime']).dt.strftime('%Y%m%d%H%M')
    profile_of = p2.assign(key=key.values).groupby('key')['class_profid'].first()
    base_key = reg.index.to_series().str.replace(r'_\d+$', '', regex=True)  # drop the repeat counter
    base_key = base_key.where(base_key.isin(profile_of.index), reg.index.to_series())
    profids = pd.Series(profile_of.reindex(base_key.values).values, index=reg.index)
    if profids.isna().any():
        print(f'  {int(profids.isna().sum())} regression samples have no class in the P2 file(s): left out')
    return profids.dropna()


def sea_ice_clusters(data, n_gmm):
    floats = data[data['platform'].isin(FLOAT_PLATFORMS)]
    return [
        k
        for k in range(1, n_gmm + 1)
        if (floats.loc[floats.cluster == k, 'sea_ice'] > ICE_THRESHOLD).sum() > MIN_ICE_PROFILES
    ]


def run_cv(data, n_gmm, ice_clusters, n_outer, seed):
    """Out-of-fold weighted_pred (and fold number) for every sample: one default RFR per class."""
    feats = {k: FEATURES if k in ice_clusters else [f for f in FEATURES if f != 'sea_ice'] for k in range(1, n_gmm + 1)}
    containers = {
        k: crossvalidation.setup_cv_container(
            data[data.cluster == k], n_outer=n_outer, spatial_cv='platform', indexer='platform_id', random_state=seed
        )
        for k in range(1, n_gmm + 1)
        if (data.cluster == k).any()
    }
    preds = []
    for f in range(1, n_outer + 1):
        foldtag = f'fold{f}'
        val = pd.concat([c.get_validation_data(foldtag) for c in containers.values()])
        if val.empty:
            continue
        cluster_preds = pd.DataFrame(index=val.index, columns=range(1, n_gmm + 1), dtype=float)
        for k, container in containers.items():
            train = container.get_training_data(foldtag)
            if len(train) == 0:  # class with too few platforms to leave one out
                continue
            model = RandomForestRegressor(random_state=24, n_jobs=-1).fit(train[feats[k]], train[TARGET])
            cluster_preds[k] = model.predict(val[feats[k]])
        probs = val[[f'cluster{k}_prob' for k in range(1, n_gmm + 1)]].to_numpy()
        have = cluster_preds.notna().to_numpy()
        # weighted over the classes with a model in this fold (all of them except for tiny classes)
        weighted = np.nansum(probs * cluster_preds.to_numpy(), axis=1) / (probs * have).sum(axis=1)
        own = cluster_preds.to_numpy()[np.arange(len(val)), val['cluster'].to_numpy() - 1]
        preds.append(pd.DataFrame({'fold': f, 'weighted_pred': weighted, 'own_pred': own}, index=val.index))
    return pd.concat(preds).reindex(data.index)


def metrics(pred, truth):
    err = pred - truth
    return {
        'rmse': np.sqrt(np.mean(err**2)),
        'mae': np.mean(np.abs(err)),
        'medae': np.median(np.abs(err)),
        'bias': np.mean(err),
        'r2': 1 - np.sum(err**2) / np.sum((truth - truth.mean()) ** 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--n-gmm', type=int, nargs='+', default=list(range(2, 11)), help='Numbers of classes (default 2-10)'
    )
    parser.add_argument(
        '--seeds', type=int, nargs='+', default=[42, 0, 1, 2, 3], help='Fold seeds (default 42 0 1 2 3)'
    )
    parser.add_argument('--n-outer', type=int, default=10, help='Outer folds (default 10)')
    parser.add_argument('--no-single', action='store_true', help='Leave out the k = 1 (no clustering) baseline')
    for key, value in DEFAULTS.items():
        parser.add_argument('--' + key.replace('_', '-'), default=value, nargs='+' if key == 'p2_trainval' else None)
    parser.add_argument(
        '--drop-features', nargs='+', default=[], help='Features to leave out (e.g. adt for synthetic data)'
    )
    parser.add_argument(
        '--keep-samples',
        default=None,
        help='csv indexed by sample_key (e.g. n_gmm_cv_predictions.csv): only these samples',
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    FEATURES[:] = [f for f in FEATURES if f not in args.drop_features]

    finder = pd.read_parquet(args.finder).set_index('profid')
    pcm = finder.attrs.get('pcm_params', os.path.basename(args.finder))
    pcs = finder[[c for c in finder.columns if c.startswith('PC') and c[2:].isdigit()]]
    reg = pd.read_csv(args.reg_trainval, index_col=0)
    p2_files = [args.p2_trainval] if isinstance(args.p2_trainval, str) else args.p2_trainval
    profids = sample_profiles(pd.concat([pd.read_parquet(f) for f in p2_files], ignore_index=True), reg)
    if args.keep_samples:
        keep = pd.read_csv(args.keep_samples, index_col=0, usecols=[0]).index
        print(f'  keeping the {len(keep)} samples in {os.path.basename(args.keep_samples)}')
        profids = profids[profids.index.isin(keep)]
    in_fit = profids.isin(finder.index)  # e.g. profiles complete above the dbar limit of another fit
    if not in_fit.all():
        print(f'  {int((~in_fit).sum())} samples whose profile is not in this PCM_finder: left out')
        profids = profids[in_fit]
    reg = reg.loc[profids.index]
    print(f'{pcm}: {len(reg)} trainval samples, {len(finder)} profiles in the PCM_finder ({pcs.shape[1]} PCs)')

    base = reg[['platform', 'platform_id', 'latitude'] + FEATURES + [TARGET]].copy()
    base['platform_id'] = base['platform_id'].astype(str)
    n_list = ([] if args.no_single else [1]) + sorted(set(args.n_gmm))
    rows, all_preds = [], {}
    for n_gmm in n_list:
        if n_gmm == 1:
            probs = pd.DataFrame({'cluster1_prob': 1.0}, index=reg.index)
        else:
            t0 = datetime.now()
            probs = fit_classes(pcs, finder['latitude'].values, n_gmm).reindex(profids.values)
            probs.index = reg.index
            if n_gmm == 6 and str(pcm) in os.path.basename(args.reg_trainval):  # check: refit = classes in use
                same = np.allclose(
                    probs.to_numpy(), reg[[f'cluster{k}_prob' for k in range(1, 7)]].to_numpy(), atol=1e-6
                )
                print(f'  k = 6 refit matches the regression csv probabilities: {same}')
            print(f'k = {n_gmm}: GMM fit in {(datetime.now() - t0).seconds} s')
        data = base.join(probs)
        data['cluster'] = probs.to_numpy().argmax(axis=1) + 1
        ice = sea_ice_clusters(data, n_gmm)
        sizes = data['cluster'].value_counts().reindex(range(1, n_gmm + 1), fill_value=0)
        for seed in args.seeds:
            t0 = datetime.now()
            oof = run_cv(data, n_gmm, ice, args.n_outer, seed)
            ok = oof['weighted_pred'].notna()
            truth = data.loc[ok, TARGET]
            fold_rmse = (
                oof[ok].groupby('fold').apply(lambda d: np.sqrt(np.mean((d.weighted_pred - truth[d.index]) ** 2)))
            )
            band = ok & data['latitude'].between(-70, -35)
            row = {'n_gmm': n_gmm, 'seed': seed, 'n': int(ok.sum())}
            row |= metrics(oof.loc[ok, 'weighted_pred'], truth)
            row |= {
                'fold_rmse_mean': fold_rmse.mean(),
                'fold_rmse_std': fold_rmse.std(),
                'rmse_own_cluster': metrics(oof.loc[ok, 'own_pred'], truth)['rmse'],
                'rmse_70S_35S': metrics(oof.loc[band, 'weighted_pred'], data.loc[band, TARGET])['rmse'],
                'sea_ice_clusters': ' '.join(map(str, ice)),
                'min_class_n': int(sizes.min()),
                'class_n': ' '.join(map(str, sizes.tolist())),
            }
            rows.append(row)
            all_preds[f'k{n_gmm}_seed{seed}'] = oof['weighted_pred']
            print(
                f'k = {n_gmm}, seed {seed}: RMSE {row["rmse"]:.3f} (folds {row["fold_rmse_mean"]:.2f} '
                f'+/- {row["fold_rmse_std"]:.2f}), MAE {row["mae"]:.3f}, sea ice {ice}, '
                f'class sizes {sizes.tolist()} ({(datetime.now() - t0).seconds} s)'
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out_dir, 'n_gmm_cv_summary.csv'), index=False)
    pd.DataFrame(all_preds).join(base[['platform', 'latitude', TARGET]]).to_csv(
        os.path.join(args.out_dir, 'n_gmm_cv_predictions.csv')
    )
    cols = [
        'n_gmm',
        'seed',
        'rmse',
        'mae',
        'medae',
        'bias',
        'r2',
        'fold_rmse_std',
        'rmse_own_cluster',
        'rmse_70S_35S',
        'sea_ice_clusters',
        'min_class_n',
    ]
    print('\nOut-of-fold errors of weighted_pred (prediction - truth), default RFR, no calibration:')
    print(summary[cols].round(3).to_string(index=False))

    # Mean over seeds for each k, and the RMSE difference from k = 1 paired by seed (same seed, same
    # k = 1 folds); seed-to-seed spread shows how much of a difference is fold noise
    by_k = summary.groupby('n_gmm').agg(
        rmse=('rmse', 'mean'),
        rmse_seed_sd=('rmse', 'std'),
        mae=('mae', 'mean'),
        r2=('r2', 'mean'),
        rmse_own_cluster=('rmse_own_cluster', 'mean'),
        rmse_70S_35S=('rmse_70S_35S', 'mean'),
        min_class_n=('min_class_n', 'first'),
        sea_ice_clusters=('sea_ice_clusters', 'first'),
    )
    if 1 in by_k.index:
        rmse_by_seed = summary.pivot(index='seed', columns='n_gmm', values='rmse')
        paired = rmse_by_seed.sub(rmse_by_seed[1], axis=0)
        by_k.insert(2, 'd_rmse_vs_k1', paired.mean())
        by_k.insert(3, 'd_rmse_vs_k1_sd', paired.std())
    by_k.to_csv(os.path.join(args.out_dir, 'n_gmm_cv_summary_by_k.csv'))
    print(f'\nMean over seeds {args.seeds}:')
    print(by_k.round(3).to_string())

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for col, label, color, marker in [
        ('rmse', 'all samples', 'C0', 'o'),
        ('rmse_70S_35S', '70S-35S', 'C1', 's'),
    ]:
        ax.scatter(summary['n_gmm'], summary[col], s=12, color=color, alpha=0.35, linewidths=0)  # each seed
        ax.plot(by_k.index, by_k[col], color=color, marker=marker, lw=2, label=f'{label} (mean of seeds)')
    ax.set_xlabel('Number of GMM classes (1 = no clustering)')
    ax.set_ylabel(f'CV RMSE of {TARGET} (µatm)')
    ax.set_xticks(n_list)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    ax.set_title(f'{pcm}: default RFR, {args.n_outer}-fold platform CV; small dots: seeds {args.seeds}', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'n_gmm_cv_rmse.png'), dpi=150)
    print(f'\nSaved summary, predictions and plot to {args.out_dir}')


if __name__ == '__main__':
    main()
