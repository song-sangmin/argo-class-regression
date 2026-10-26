# Clustering (P2) and regression inputs

Every script takes `--config <run.toml>`, like the data-processing scripts. The clustering
settings are in the config's `[P2]` table, and the regression settings are in `[regression]`.

| Step | Command (from `scripts/`) | Writes (`[files]` key) |
|---|---|---|
| Fit | `python clustering/fit_argo_clustering.py --config $C` | `P2_finder`, `P2_fit` (+ `P2_model` with `--save-model`) |
| Classify | `python clustering/apply_P2_classes.py --config $C --dataset <name>`, once per P1 dataset | `P2_<name>` |
| Regression inputs | `python regression/regression-input-prep.py --config $C` | `P2_trainval`, `P2_test`, `P2_application`, `reg_trainval`, `reg_application`, `reg_info` |

- **Fit**: PCA + GMM on the CT/SA profiles of the L3 datasets in `[P2] fit_datasets`
  (`["core", "bgc"]`, fit together). Classes are numbered north (1) to south.
- **Classify**: attaches class probabilities to one P1 dataset. Float datasets (`core`, `bgc`,
  `soccom`) match on the sample's own `profid`. Ship datasets (`socat`) match on
  `nearest_profid`. Samples without a class are gap-filled from the same float or cruise,
  otherwise dropped. Each output covers all years, one file per dataset.
- **Regression inputs**: combines the `[regression] train` datasets and splits them by year
  (`trainval_years`, `test_years`). It drops application profiles that are also float training
  samples (`exclude_training`), finds the sea ice clusters, and writes the csvs that
  `model-optimization/` reads. The sea ice clusters and the features go to `reg_info`, so
  `run_dask_nested_cv.py --config` picks them up without `--sea-ice-clusters`.

## Example (Drake Passage) and CRUSOE

```bash
cd scripts
# Example: cluster all core + BGC profiles together, then one P2 file per platform
C=../configs/example.toml
python clustering/fit_argo_clustering.py --config $C
python clustering/apply_P2_classes.py --config $C --dataset core
python clustering/apply_P2_classes.py --config $C --dataset bgc
python regression/regression-input-prep.py --config $C       # trains on bgc (target pH), applies to core

# CRUSOE: train on SOCCOM + SOCAT, apply to core
C=../configs/southernOcean_CRUSOE.toml
python clustering/fit_argo_clustering.py --config $C
for ds in soccom socat core; do python clustering/apply_P2_classes.py --config $C --dataset $ds; done
python regression/regression-input-prep.py --config $C
```

For CRUSOE, these steps reproduce the `pc8_gmm6_dbar500` files in use exactly (checked on 2026-10-21):
the PCM_finder, the P2 combined trainval/test and core application parquet files, and both
regression-input csvs (byte for byte). Those files are listed in `[files.pc8_gmm6_dbar500]`, so a
rerun stops before overwriting them unless `--overwrite` is passed.

## Several clusterings in one run

L0 to P1 do not depend on the clustering, so a new fit does not rerun the data processing. The
clustering options make a tag, `{pcm}`, such as `pc8_gmm6_dbar500`. Each fit's files go to their
own folder (`P2-clustered/<pcm>/` for CRUSOE, `<run_dir>/P2/<pcm>/` by default), so fits sit side by
side and never overwrite each other.

You can change a fit in two ways:

- **One-off (recommended for trying options)**: pass the same options to every P2 step. The
  config stays the same, and the options are recorded in each step's `provenance/` json.

  ```bash
  V="--n-gmm 7 --dbar-limit 900"                  # -> pcm pc8_gmm7_dbar900
  python clustering/fit_argo_clustering.py --config $C $V
  for ds in soccom socat core; do python clustering/apply_P2_classes.py --config $C --dataset $ds $V; done
  python regression/regression-input-prep.py --config $C $V
  python model-optimization/run_dask_nested_cv.py RFR rfr_test --config $C $V   # reads that fit's csv and reg_info
  ```

  The options are `--n-pca`, `--n-gmm`, `--dbar-limit` and `--exclude-years`. In zsh, write
  `${=V}` instead of `$V` so the options are passed separately.

- **New default**: edit `[P2]` in the config. The scripts then refuse to run until you pass
  `--update-config` once, because the run folder's `config.toml` no longer matches; the old copy
  is kept as `config_<time>.toml`.

Other options:

- `--exclude-years 2024`: profiles from those years are not used to fit (the centering, scaling,
  PCA and GMM). They are still classified with the fitted model and kept in PCM_finder with
  `in_fit = False`. The tag gets `_excl2024`.
- `--save-model` (fit only): also saves the per-level means, the CT/SA scaling, the PCA, the GMM
  and the class mapping (`P2_model`, joblib) to classify other profiles later. It's off by default.

To see what exists for a clustering, from Python:

```python
from argoclare import pipeline
run = pipeline.load('configs/southernOcean_CRUSOE.toml', n_gmm=7)
run.status()                                # every [files] entry for pc8_gmm7_dbar500, and whether it exists
finder = run.open('P2_finder')
```

## Laptop and NERSC

A config can hold machine profiles: `[machines.<name>]` tables that override any setting on that
machine, such as `data_root`, `run_dir` or `[files]` paths. Pick one with `--machine nersc`, or set
it once per machine:

```bash
export ARGOCLARE_MACHINE=nersc     # e.g. in ~/.bashrc on Perlmutter, or in the sbatch script
```

The config file is the same on both machines, so the run folder's `config.toml` matches either
way. `southernOcean_CRUSOE.toml` and `synthetic_CRUSOE.toml` have a `nersc` profile that points at
the regression inputs copied to the repo checkout and at the results folder on pscratch.
