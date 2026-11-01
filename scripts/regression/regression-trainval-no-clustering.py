"""Create a single-cluster trainval csv, to compare the clustered regression with one model
trained on the same data without clustering.

Reads [files] P2_trainval (written by regression-input-prep.py) and makes the same table as
regression-input-prep.py's reg_trainval (same features, target QC, dropped rows and sample_key
index), then replaces the class info with a single cluster:
    cluster          -> gmm_cluster (kept to compare errors by GMM class afterwards)
    cluster{k}_prob  -> dropped
    cluster = 1, cluster1_prob = 1.0
model-optimization counts clusters from the cluster{k}_prob columns, so it trains one model.

Writes combined_trainval_noclusters_<pcm>_acc<date>.csv to a no_clusters/ folder next to
[files] reg_trainval (or --out-dir). Pass it to the model scripts with --data, and set sea ice
explicitly (--sea-ice-clusters 1 to use it everywhere, or --drop-features sea_ice), since
--config would otherwise take the clustered run's sea ice clusters.

Usage:
    python regression-trainval-no-clustering.py --config ../../configs/southernOcean_CRUSOE.toml
"""

import importlib.util
import os
from datetime import datetime

import pandas as pd

from argoclare import pipeline

# regression-input-prep.py has a hyphen in its name, so import it by path
_spec = importlib.util.spec_from_file_location(
    'regression_input_prep', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'regression-input-prep.py')
)
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)


def main():
    parser = pipeline.add_pcm_args(pipeline.parser(__doc__))
    args = parser.parse_args()
    run = pipeline.start(args, __file__)
    cfg = run.section('regression')
    feat_list = [f for f in cfg['features'] if f not in cfg.get('drop_features', [])]
    target = cfg.get('target', 'delta_pco2')

    out_dir = args.out_dir or os.path.join(os.path.dirname(run.path('reg_trainval', ice_tag='')), 'no_clusters')
    out_file = os.path.join(
        os.path.abspath(out_dir), f'combined_trainval_noclusters_{run.pcm}_acc{datetime.now():%Y%m%d}.csv'
    )
    if os.path.exists(out_file) and not run.overwrite:
        raise SystemExit(f'Output exists, pass --overwrite to replace it: {out_file}')

    p2_file = run.input('P2_trainval')
    p2 = pd.read_parquet(p2_file)
    prep.check_columns(p2, feat_list + [target], os.path.basename(p2_file))
    print(f'{os.path.basename(p2_file)}: {len(p2)} trainval samples')
    print('Features:', feat_list, '| target:', target)
    trainval, prob_cols = prep.regression_table(p2, cfg, feat_list, target, 'trainval')

    # Single cluster, keeping the GMM class for comparison
    trainval = trainval.drop(columns=prob_cols).rename(columns={'cluster': 'gmm_cluster'})
    trainval['cluster'] = 1
    trainval['cluster1_prob'] = 1.0

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    trainval.to_csv(out_file)
    run.outputs.append(out_file)
    print(f'\nSaved {len(trainval)} samples (1 cluster; GMM class in gmm_cluster) to {out_file}')
    run.finish(source_pcm=run.pcm, n_trainval=len(trainval))


if __name__ == '__main__':
    main()
