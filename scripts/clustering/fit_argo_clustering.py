"""Fit the profile classification model (PCA + GMM) on core and BGC Argo L3 profiles, 2014-2024.

Same model as 2.1_PCM_fit_coreArgo_2014-2024.ipynb (argoclare pcm.run_pca_and_gmm), with
three changes (see notes below):
    1. Read CT and SA from the core and bgc L3 DATA files at pressure levels < --dbar-limit
       only, and keep profiles with CT and SA at every one of those levels (the profiles
       run_pca_and_gmm() keeps). A profid in both files is taken from core (core first).
    2. Center CT and SA at each level by the mean over these profiles, then divide all CT
       values by their overall std and all SA values by theirs (centering only on profiles
       that are fit; the notebook centered on all profiles, including ones dropped later)
    3. PCA (--n-pca) and GMM (--n-gmm, diagonal covariance, random_state 0, max_iter 10000)
    4. Renumber classes 1..n_gmm by mean profile latitude, north to south, so class numbers
       do not depend on the GMM's arbitrary label order and stay the same across reruns
Writes to <outpath>/<pcm_params>/ (pcm_params = pc<n_pca>_gmm<n_gmm>_dbar<dbar_limit>):
    PCM_finder_<pcm_params>_acc<datetag>.parquet: one row per clustered profid with profid,
        class1_prob..classN_prob, class_assign, PC1..PCn, wmoid, prof_datetag, datetime,
        latitude, longitude, source ('core' or 'bgc'); the fit metadata is in df.attrs
    PCM_fit_<pcm_params>_acc<datetag>.json: the same metadata (parameters, input files,
        explained variance, class sizes, mean class latitudes, GMM label -> class mapping)

Usage:
    python fit_argo_clustering.py                                          # pc8_gmm6_dbar500
    python fit_argo_clustering.py --n-gmm 7 --dbar-limit 900 --outpath /tmp/test/
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import importer # noqa: E402

default_outpath = importer.local_working_folder + 'P2-clustered/'


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
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-pca', type=int, default=8, help='Number of PCA components')
    parser.add_argument('--n-gmm', type=int, default=6, help='Number of GMM classes')
    parser.add_argument('--dbar-limit', type=float, default=500, help='Use pressure levels < this (dbar)')
    parser.add_argument('--core', nargs=2, metavar=('DATA', 'INDEX'),
                        default=[importer.path_to('L3', 'coreDATA'), importer.path_to('L3', 'coreINDEX')])
    parser.add_argument('--bgc', nargs=2, metavar=('DATA', 'INDEX'),
                        default=[importer.path_to('L3', 'bgcDATA'), importer.path_to('L3', 'bgcINDEX')])
    parser.add_argument('--outpath', default=default_outpath, help='Folder for the <pcm_params>/ output folder')
    args = parser.parse_args()

    dbar_tag = f'{args.dbar_limit:g}'
    pcm_params = f'pc{args.n_pca}_gmm{args.n_gmm}_dbar{dbar_tag}'
    print(f'==> Fitting {pcm_params}: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))

    ct, sa, index = read_fit_profiles([('core', *args.core), ('bgc', *args.bgc)], args.dbar_limit)

    # Center each level over the fit profiles, then scale CT and SA blocks by their overall std
    ct_c, sa_c = ct - ct.mean(axis=0), sa - sa.mean(axis=0)
    ct_std, sa_std = float(np.std(ct_c.values, ddof=1)), float(np.std(sa_c.values, ddof=1))
    X = np.hstack([ct_c.values / ct_std, sa_c.values / sa_std])
    print(f'  PCA input: {X.shape[0]} profiles x {X.shape[1]} (CT, SA levels); CT std {ct_std:.3f}, SA std {sa_std:.4f}')

    pca = PCA(args.n_pca).fit(X)
    Xtrans = pca.transform(X)
    print(f'  explained variance: {pca.explained_variance_ratio_.sum():.4f}: ',
          datetime.now().strftime('%Y%m%d %H:%M:%S'))
    gmm = GMM(n_components=args.n_gmm, covariance_type='diag', max_iter=10000, random_state=0).fit(Xtrans)
    labels, probs = gmm.predict(Xtrans), gmm.predict_proba(Xtrans)
    print(f'  GMM converged: {gmm.converged_} after {gmm.n_iter_} iterations: ',
          datetime.now().strftime('%Y%m%d %H:%M:%S'))

    # Class numbers 1..n_gmm by mean latitude of the assigned profiles, north to south
    mean_lat = pd.Series(index['latitude'].values).groupby(labels).mean()
    order = mean_lat.sort_values(ascending=False).index.values # GMM labels, north first
    class_of = {int(lab): k + 1 for k, lab in enumerate(order)}

    finder = index.copy()
    finder.index.name = 'profid'
    for lab, k in class_of.items():
        finder[f'class{k}_prob'] = probs[:, lab]
    finder = finder[[c for c in finder.columns if not c.startswith('class')]
                    + [f'class{k}_prob' for k in range(1, args.n_gmm + 1)]]
    finder['class_assign'] = [class_of[int(lab)] for lab in labels]
    for i in range(args.n_pca):
        finder[f'PC{i + 1}'] = Xtrans[:, i]

    counts = finder.class_assign.value_counts().sort_index()
    lats = finder.groupby('class_assign').latitude.mean()
    print('  class  profiles  mean_lat')
    for k in counts.index:
        print(f'  {k:5d}  {counts[k]:8d}  {lats[k]:8.2f}')

    datetag = datetime.now().strftime('%Y%m%d')
    savepath = os.path.join(args.outpath, pcm_params)
    os.makedirs(savepath, exist_ok=True)
    finder_file = os.path.join(savepath, f'PCM_finder_{pcm_params}_acc{datetag}.parquet')
    meta = {'pcm_params': pcm_params, 'n_pca': args.n_pca, 'n_gmm': args.n_gmm, 'dbar_limit': args.dbar_limit,
            'n_levels': int(ct.shape[1]), 'n_profiles': int(len(finder)),
            'profiles_by_source': finder.source.value_counts().to_dict(),
            'inputs': {'core': [os.path.basename(f) for f in args.core], 'bgc': [os.path.basename(f) for f in args.bgc]},
            'ct_std': ct_std, 'sa_std': sa_std,
            'explained_variance_ratio': pca.explained_variance_ratio_.round(6).tolist(),
            'gmm': {'covariance_type': 'diag', 'random_state': 0, 'max_iter': 10000,
                    'converged': bool(gmm.converged_), 'n_iter': int(gmm.n_iter_)},
            'class_numbering': 'by mean latitude of assigned profiles, north (1) to south',
            'gmm_label_to_class': {str(k): v for k, v in class_of.items()},
            'class_counts': {str(k): int(v) for k, v in counts.items()},
            'class_mean_latitude': {str(k): round(float(v), 3) for k, v in lats.items()},
            'script': os.path.basename(__file__), 'date': datetag}
    with open(os.path.join(savepath, f'PCM_fit_{pcm_params}_acc{datetag}.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    finder = finder.reset_index()
    finder.attrs = json.loads(json.dumps(meta)) # also stored in the parquet file (df.attrs on read)
    finder.to_parquet(finder_file, index=False)
    print(f'Saved {len(finder)} profiles to {finder_file}: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))


if __name__ == '__main__':
    main()
