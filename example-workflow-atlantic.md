# Example workflow: Atlantic sector of the Southern Ocean

A larger version of [example-workflow.md](example-workflow.md). The steps and commands are
the same; only the config changes: `configs/example_atlantic.toml`.

| | `example.toml` (Drake Passage) | `example_atlantic.toml` |
|---|---|---|
| Region | 70°W–50°W, 65°S–55°S | 70°W–20°E, south of 35°S |
| Years | 2020 | 2018–2022 |
| Core / BGC float profiles (L3) | ~280 / ~50 | ~38,000 / ~1,800 |
| BGC floats with good surface oxygen | 6 | 12–20 each year |
| Clusters (`[P2] n_gmm`) | 3 | 2 |
| Training / test years | 2020 / – | 2018–2021 / 2022 |
| `[regression] cv_type` | `profile` | `profile` |

Same as the Drake example, the regression predicts surface **oxygen** on the BGC floats from
CT, SA, ADT, max_solar_rad and atmos_pres_atm, and applies the models to the core floats.

## Run it

Follow [example-workflow.md](example-workflow.md) steps 1–4 with:

```bash
cp configs/example_atlantic.toml configs/my_atlantic.toml    # set [files] to your external inputs
C=configs/my_atlantic.toml
```

The external inputs (ERA5, ADT, sea ice, bathymetry, MBL) must cover 70°W–20°E south of 35°S
for 2018–2022.

To run on NERSC or UW Hyak, use the machine profiles at the bottom of the config; see
[Running on an HPC system](example-workflow.md#6-running-on-an-hpc-system), which also shows how
to set the Slurm account, partition and resources with `sbatch` flags instead of editing the job scripts. The data steps take
about 2 hours on a laptop (below), so a common split is to process the data locally and run the
nested CV with the full grids on the cluster.

## How long it takes

Measured on 2026-10-23 for one year (2020) of this region, on a laptop (Apple silicon, 14 cores)
with a home internet connection. Five years take roughly five times as long, except for the
fit and the models.

| Step | Core | BGC | Notes |
|---|---|---|---|
| L0 download (ERDDAP) | 12.4 min (7,843 profiles, 760 MB) | 9.9 min (1,386 profiles, 330 MB) | run core and bgc at the same time; depends on the server |
| L1 QC | 1.5 min | 16 s | |
| L2 open-ocean mask | 4.8 min | 3 s | the GEBCO lookup |
| L3 regrid + MLD | 30 s | 4 s | 6,542 core and 309 BGC profiles kept |
| P1 features | 63 s | 12 s | |
| P2 fit, classify, regression prep | | | under 10 s in all |
| Nested CV, `rfr_test` grid, 5 × 3 folds | | | about 9 s; the full grids take much longer |

**Per year:** about 12 min of downloading (core and BGC side by side), then about 8–9 min of
processing. **For 2018–2022:** about 1 hour of downloads and 45 minutes of processing, so allow
about 2 hours before the first model run. For a quicker first try, set `years = [2020]` (and
`trainval_years = [2020]`, `test_years = []`). That's about 20 minutes, and the 2-cluster
`profile` cross-validation still runs (see below).

## Choosing the cross-validation folds (`cv_type`)

Nested cross-validation holds out part of the training data to test each model. `[regression]
cv_type` sets what a held-out piece is, for both `run_dask_nested_cv.py` and
`run_model_application.py` (`--cv-type` on the command line overrides it):

| `cv_type` | Held out together | Use when |
|---|---|---|
| `platform` | whole floats (and whole SOCAT cruises) | the default for CRUSOE; needs many floats in every cluster |
| `profile` | whole profiles (with depth-resolved data, all depths of a profile stay together) | small examples with few floats |
| `kmeans` | whole regions (k-means on position) | testing how models carry over to unsampled areas |

With `profile`, `regression-input-prep.py` adds a `profid` column to the training csv, and the
folds use the existing whole-platform splitting in `argoclare.crossvalidation` with `profid` as the
group (`spatial_cv='platform', indexer='profid'`). Rerun regression prep after changing `cv_type`.

`profile` folds let a float's other profiles stay in training, so the scores are more
optimistic than with `platform` folds. Profiles from the same float a few days apart are
alike, so the model gets credit partly for recognizing the float. Tested on this example's data
(2026-10-23, RFR `rfr_test` grid, 5 outer × 3 inner folds):

| Data | `cv_type` | Training samples | Oxygen RMSE (µmol/kg) | MAE | Oxygen std |
|---|---|---|---|---|---|
| 2018–2021 | `profile` | 1,298 | 6.1 | 4.2 | 40.8 |
| 2018–2021 | `platform` | 1,298 | 9.3 | 6.1 | 40.8 |
| 2020 only | `profile` | 308 | 7.5 | 5.2 | 39.3 |
| 2020 only | `platform` | 308 | 11.7 | 8.5 | 39.3 |

Both run here, even with one year, but report `platform` errors when the question is how well
the method works for a float it has never seen.
