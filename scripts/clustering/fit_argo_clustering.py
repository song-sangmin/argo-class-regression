"""Fit the profile classification model (PCA + GMM) on the L3 Argo profiles of a run.

Same model as 2.1_PCM_fit_coreArgo_2014-2024.ipynb (argoclare pcm.run_pca_and_gmm), with
three changes (see notes below):
    1. Read CT and SA from the [files] L3_data of each [P2] fit_datasets (CRUSOE: core, bgc)
       at pressure levels < dbar_limit only, and keep profiles with CT and SA at every one of
       those levels (the profiles run_pca_and_gmm() keeps). A profid in several datasets is
       taken from the first.
    2. Center CT and SA at each level by the mean over these profiles, then divide all CT
       values by their overall std and all SA values by theirs (centering only on profiles
       that are fit; the notebook centered on all profiles, including ones dropped later)
    3. PCA (n_pca) and GMM (n_gmm, diagonal covariance, random_state 0, max_iter 10000)
    4. Renumber classes 1..n_gmm by mean profile latitude, north to south, so class numbers
       do not depend on the GMM's arbitrary label order and stay the same across reruns
With exclude_years, profiles from those years are not used for the centering, scaling, PCA or
GMM, but are classified with the fitted model and kept in PCM_finder with in_fit = False.
n_pca, n_gmm, dbar_limit and exclude_years come from [P2], or from --n-pca, --n-gmm,
--dbar-limit, --exclude-years. They make the clustering tag {pcm} (e.g. pc8_gmm6_dbar500), so
each fit gets its own files and several fits can share one run (see README.md).
Writes:
    [files] P2_finder (parquet): one row per clustered profid with profid, class1_prob..classN_prob,
        class_assign, PC1..PCn, wmoid, prof_datetag, datetime, latitude, longitude, source
        (the L3 dataset); the fit metadata is in df.attrs
    [files] P2_fit (json): the same metadata (parameters, input files, explained variance, class
        sizes, mean class latitudes, GMM label -> class mapping)
    [files] P2_model (joblib), with --save-model: level means, CT/SA scaling, the PCA and GMM,
        and the class mapping, to classify other profiles later

Usage:
    python fit_argo_clustering.py --config ../../configs/southernOcean_CRUSOE.toml       # pc8_gmm6_dbar500
    python fit_argo_clustering.py --config ../../configs/southernOcean_CRUSOE.toml --n-gmm 7 --dbar-limit 900
    python fit_argo_clustering.py --config ../../configs/example.toml --exclude-years 2020 --save-model
"""
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM

from argoclare import pipeline


def read_fit_profiles(sources, dbar_limit):
    """CT and SA (profid x level) for profiles complete at every level < dbar_limit, plus their index."""
    ct_parts, sa_parts, index_parts = [], [], []
    seen = set()
    for name, data_file, index_file in sources:
        data = xr.open_dataset(data_file)
        levels = np.flatnonzero(data.pressure.values < dbar_limit)
        ct = data['CT'].transpose('profid', 'pressure').isel(pressure=levels).values
        sa = data['SA'].transpose('profid', 'pressure').isel(pressure=levels).values
        profids = data.profid.values.astype(str)
        keep = ~np.isnan(ct).any(axis=1) & ~np.isnan(sa).any(axis=1) & ~np.isin(profids, list(seen))
        print(f'  {name}: {keep.sum()} of {len(profids)} profiles have CT and SA at all '
              f'{len(levels)} levels < {dbar_limit:g} dbar (not already in an earlier source)')
        pressures = data.pressure.values[levels]
        ct_parts.append(pd.DataFrame(ct[keep], index=profids[keep], columns=pressures))
        sa_parts.append(pd.DataFrame(sa[keep], index=profids[keep], columns=pressures))
        seen |= set(profids[keep])

        idx = xr.open_dataset(index_file)
        idx_df = pd.DataFrame({'wmoid': idx.wmoid.values, 'prof_datetag': idx.prof_datetag.values.astype(str),
                               'datetime': pd.to_datetime(idx.datetime.values),
                               'latitude': idx.latitude.values, 'longitude': idx.longitude.values},
                              index=idx.profid.values.astype(str))
        index_parts.append(idx_df.loc[profids[keep]].assign(source=name))
    ct, sa = pd.concat(ct_parts), pd.concat(sa_parts)
    index = pd.concat(index_parts)
    assert (ct.index == index.index).all() and ct.index.is_unique
    return ct, sa, index


def main():
    parser = pipeline.add_pcm_args(pipeline.parser(__doc__))
    parser.add_argument('--save-model', action='store_true',
                        help='Also save the fitted model ([files] P2_model) to classify other profiles later')
    args = parser.parse_args()
    run = pipeline.start(args, __file__)
    p2, pcm_params = run.p2, run.pcm
    n_pca, n_gmm, dbar_limit = p2['n_pca'], p2['n_gmm'], p2['dbar_limit']
    exclude_years = sorted(p2.get('exclude_years') or [])
    datasets = p2.get('fit_datasets', ['core', 'bgc'])
    sources = [(ds, run.input('L3_data', ds), run.input('L3_index', ds)) for ds in datasets]
    finder_file, fit_file = run.output('P2_finder'), run.output('P2_fit')
    model_file = run.output('P2_model') if args.save_model else None
    print(f'==> Fitting {pcm_params}: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))

    ct, sa, index = read_fit_profiles(sources, dbar_limit)
    in_fit = ~index['datetime'].dt.year.isin(exclude_years).values
    if exclude_years:
        print(f'  {int((~in_fit).sum())} profiles from {exclude_years} left out of the fit (classified after)')

    # Center each level over the fit profiles, then scale CT and SA blocks by their overall std
    ct_mean, sa_mean = ct[in_fit].mean(axis=0), sa[in_fit].mean(axis=0)
    ct_c, sa_c = ct - ct_mean, sa - sa_mean
    ct_std, sa_std = float(np.std(ct_c.values[in_fit], ddof=1)), float(np.std(sa_c.values[in_fit], ddof=1))
    X = np.hstack([ct_c.values / ct_std, sa_c.values / sa_std])
    print(f'  PCA input: {int(in_fit.sum())} profiles x {X.shape[1]} (CT, SA levels); CT std {ct_std:.3f}, SA std {sa_std:.4f}')

    pca = PCA(n_pca).fit(X[in_fit])
    Xtrans = pca.transform(X)
    print(f'  explained variance: {pca.explained_variance_ratio_.sum():.4f}: ',
          datetime.now().strftime('%Y%m%d %H:%M:%S'))
    gmm = GMM(n_components=n_gmm, covariance_type='diag', max_iter=10000, random_state=0).fit(Xtrans[in_fit])
    labels, probs = gmm.predict(Xtrans), gmm.predict_proba(Xtrans)
    print(f'  GMM converged: {gmm.converged_} after {gmm.n_iter_} iterations: ',
          datetime.now().strftime('%Y%m%d %H:%M:%S'))

    # Class numbers 1..n_gmm by mean latitude of the assigned (fit) profiles, north to south
    mean_lat = pd.Series(index['latitude'].values[in_fit]).groupby(labels[in_fit]).mean()
    order = mean_lat.sort_values(ascending=False).index.values # GMM labels, north first
    class_of = {int(lab): k + 1 for k, lab in enumerate(order)}

    finder = index.copy()
    finder.index.name = 'profid'
    for lab, k in class_of.items():
        finder[f'class{k}_prob'] = probs[:, lab]
    finder = finder[[c for c in finder.columns if not c.startswith('class')]
                    + [f'class{k}_prob' for k in range(1, n_gmm + 1)]]
    finder['class_assign'] = [class_of[int(lab)] for lab in labels]
    for i in range(n_pca):
        finder[f'PC{i + 1}'] = Xtrans[:, i]
    if exclude_years:
        finder['in_fit'] = in_fit

    fitted = finder[in_fit]
    counts = fitted.class_assign.value_counts().sort_index()
    lats = fitted.groupby('class_assign').latitude.mean()
    print('  class  profiles  mean_lat')
    for k in counts.index:
        print(f'  {k:5d}  {counts[k]:8d}  {lats[k]:8.2f}')

    datetag = datetime.now().strftime('%Y%m%d')
    meta = {'pcm_params': pcm_params, 'n_pca': n_pca, 'n_gmm': n_gmm, 'dbar_limit': dbar_limit,
            'n_levels': int(ct.shape[1]), 'n_profiles': int(len(fitted)),
            'profiles_by_source': fitted.source.value_counts().to_dict(),
            'inputs': {ds: [os.path.basename(d), os.path.basename(i)] for ds, d, i in sources},
            'ct_std': ct_std, 'sa_std': sa_std,
            'explained_variance_ratio': pca.explained_variance_ratio_.round(6).tolist(),
            'gmm': {'covariance_type': 'diag', 'random_state': 0, 'max_iter': 10000,
                    'converged': bool(gmm.converged_), 'n_iter': int(gmm.n_iter_)},
            'class_numbering': 'by mean latitude of assigned profiles, north (1) to south',
            'gmm_label_to_class': {str(k): v for k, v in class_of.items()},
            'class_counts': {str(k): int(v) for k, v in counts.items()},
            'class_mean_latitude': {str(k): round(float(v), 3) for k, v in lats.items()},
            'script': os.path.basename(__file__), 'date': datetag}
    if exclude_years:
        meta['exclude_years'] = exclude_years
        meta['n_profiles_classified_not_fit'] = int((~in_fit).sum())
    meta.update({k: v for k, v in run.attrs().items() if k != 'script'})
    with open(fit_file, 'w') as f:
        json.dump(meta, f, indent=2)
    finder = finder.reset_index()
    finder.attrs = json.loads(json.dumps(meta)) # also stored in the parquet file (df.attrs on read)
    finder.to_parquet(finder_file, index=False)
    print(f'Saved {len(finder)} profiles to {finder_file}: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))

    if model_file:
        import joblib
        joblib.dump({'pressure_levels': ct.columns.values, 'ct_mean': ct_mean.values, 'sa_mean': sa_mean.values,
                     'ct_std': ct_std, 'sa_std': sa_std, 'pca': pca, 'gmm': gmm, 'gmm_label_to_class': class_of,
                     'meta': meta}, model_file)
        print('Saved model to ' + model_file)
    run.finish(pcm_params=pcm_params, class_counts=meta['class_counts'])


if __name__ == '__main__':
    main()
