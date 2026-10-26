"""Shared P2 steps: give P1 samples the GMM classes of their Argo profile, gap-fill, and save.

Used by apply_P2_classes.py (any P1 dataset) and the synthetic apply scripts in
synthetic-processing/. Classes come from a PCM_finder (parquet) made by fit_argo_clustering.py
(one row per classified profid, class1_prob..classN_prob, class_assign).
Columns added to each sample:
    class1_prob..classN_prob, class_assign
    class_profid   the profile the class came from
    class_source   'matched' (the sample's own/nearest profile) or 'gapfill'
    class_dt_days  days between the sample and class_profid
"""
import json
import os

import numpy as np
import pandas as pd


def read_finder(path):
    """PCM_finder (indexed by profid), its class probability columns, and the fit metadata."""
    if path.endswith('.parquet'):
        finder = pd.read_parquet(path)
        meta = dict(finder.attrs)
        finder = finder.set_index('profid')
    else: # older csv PCM_finder, with the metadata in the matching PCM_fit json
        finder = pd.read_csv(path, index_col='profid')
        meta_file = path.replace('PCM_finder_', 'PCM_fit_').replace('.csv', '.json')
        meta = json.load(open(meta_file)) if os.path.exists(meta_file) else {}
    finder['datetime'] = pd.to_datetime(finder['datetime'], format='mixed') # nanosecond timestamps
    prob_cols = sorted([c for c in finder.columns if c.startswith('class') and c.endswith('_prob')],
                       key=lambda c: int(c[5:-5]))
    print(f'PCM_finder: {len(finder)} classified profiles, {len(prob_cols)} classes '
          f'({meta.get("pcm_params", os.path.basename(path))})')
    return finder, prob_cols, meta


def assign_from_profid(df, profid_col, finder, prob_cols):
    """Class probabilities of each sample's profile (empty if the profile has no class)."""
    hit = df[profid_col].isin(finder.index)
    found = finder.reindex(df.loc[hit, profid_col])
    for c in prob_cols + ['class_assign']:
        df[c] = np.nan
        df.loc[hit, c] = found[c].values
    df['class_profid'] = df[profid_col].where(hit, '')
    df['class_source'] = np.where(hit, 'matched', '')
    df['class_dt_days'] = np.nan
    df.loc[hit, 'class_dt_days'] = ((df.loc[hit, 'datetime'] - found['datetime'].values).abs()
                                    / pd.Timedelta(days=1)).values
    return df


def fill_from_neighbours(df, group_col, candidates, cand_group_col, max_days, prob_cols, require_agree=True):
    """Fill unclassified samples from the nearest earlier and later classified candidates of the
    same group (float or cruise) within max_days. With require_agree, if both sides exist they
    must share a class. Probabilities are taken from the candidate nearest in time.
    Returns the DataFrame and the number of samples filled."""
    missing = df.index[df['class_assign'].isna()]
    cand = candidates.dropna(subset=['class_assign']).sort_values('datetime')
    by_group = {k: g for k, g in cand.groupby(cand_group_col)}
    n_filled = 0
    for i in missing:
        g = by_group.get(df.at[i, group_col])
        if g is None:
            continue
        t = df.at[i, 'datetime']
        dt = (g['datetime'] - t) / pd.Timedelta(days=1)
        before = dt[(dt <= 0) & (dt > -max_days)]
        after = dt[(dt > 0) & (dt < max_days)]
        sides = [s for s in [before.index[-1:] if len(before) else None,
                             after.index[:1] if len(after) else None] if s is not None]
        if not sides:
            continue
        picks = g.loc[np.concatenate(sides)]
        if require_agree and picks['class_assign'].nunique() > 1:
            continue
        best = picks.loc[(picks['datetime'] - t).abs().idxmin()]
        for c in prob_cols + ['class_assign']:
            df.at[i, c] = best[c]
        # the profile the class came from: a PCM_finder profid, or a filled-from sample's profile
        df.at[i, 'class_profid'] = best['class_profid'] if 'class_profid' in best.index else best.name
        df.at[i, 'class_source'] = 'gapfill'
        df.at[i, 'class_dt_days'] = abs((best['datetime'] - t) / pd.Timedelta(days=1))
        n_filled += 1
    return df, n_filled


def classify_float_samples(df, profid_col, finder, prob_cols, max_days):
    """Classes for float samples: their own profid, else the same float's classified profiles in
    PCM_finder just before/after within max_days (must agree; nearest in time). Needs platform_id
    = wmoid as a string. Returns the DataFrame (unclassified rows kept) and (n_matched, n_filled)."""
    df = assign_from_profid(df, profid_col, finder, prob_cols)
    n_matched = int(df['class_assign'].notna().sum())
    float_cands = finder.assign(platform_id=finder['wmoid'].astype(int).astype(str))
    df, n_filled = fill_from_neighbours(df, 'platform_id', float_cands, 'platform_id', max_days, prob_cols)
    return df, (n_matched, n_filled)



def drop_unclassified(df, label, n_matched, n_filled, how):
    """Drop samples without a class and print the counts."""
    n = len(df)
    df = df.dropna(subset=['class_assign']).copy()
    df['class_assign'] = df['class_assign'].astype(int)
    print(f'{label}: {n} samples; {n_matched} classified from their profile, {n_filled} gap-filled {how}; '
          f'dropped {n - len(df)} without a class')
    return df


def save_parquet(df, path, attrs):
    """Save one row per sample as parquet (keeps column dtypes; compressed). `attrs` (parameters,
    inputs, counts) are stored in the file and come back as df.attrs with pd.read_parquet().
    Text columns are written as strings, with missing values as ''."""
    df = df.reset_index(drop=True).copy()
    text_cols = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c].dtype)]
    for c in text_cols:
        df[c] = df[c].where(df[c].notna(), '').astype(str)
    df.columns = [str(c).replace(' ', '_') for c in df.columns]
    df.attrs = {k: (v if isinstance(v, (str, int, float)) else json.loads(json.dumps(v))) for k, v in attrs.items()}
    df.to_parquet(path, index=False)
    return df
