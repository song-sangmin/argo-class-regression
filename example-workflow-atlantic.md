# Example workflow: Atlantic sector of the Southern Ocean

A larger version of [example-workflow-drake.md](example-workflow-drake.md). The steps and commands are
the same; only the config changes: `configs/example_atlantic.toml`.

| | `example.toml` (Drake Passage) | `example_atlantic.toml` |
|---|---|---|
| Region | 70°W–50°W, 65°S–55°S | 70°W–20°E, south of 35°S |
| Years | 2018–2022 | 2018–2022 |
| Core / BGC float profiles (L3) | ~1,600 / ~140 | ~38,000 / ~1,800 |
| BGC floats with good surface oxygen | 11 in 2018–2021 | 12–20 each year |
| Clusters (`[P2] n_gmm`) | 2 | 2 |
| Training / test years | 2018–2021 / 2022 | 2018–2021 / 2022 |
| `[regression] cv_type` | `profile` | `profile` |
| Run time on a laptop | about 5 min | about 2 hours (below) |

As in the Drake example, the regression predicts surface **oxygen** on the BGC floats and applies
the models to the core floats, here from CT, SA, ADT, max_solar_rad and atmos_pres_atm.

## Run it

Follow [example-workflow-drake.md](example-workflow-drake.md) steps 1–4 with:

```bash
cp configs/example_atlantic.toml configs/my_atlantic.toml    # set [files] to your external inputs
C=configs/my_atlantic.toml
```

Its features (`CT`, `SA`, `adt`, `max_solar_rad`, `atmos_pres_atm`, in both `[P1]` and `[regression]`)
need three external inputs: the bathymetry, ERA5 sea level pressure (`era5_slp`) and the CMEMS ADT
(`adt`). They must cover 70°W–20°E south of 35°S for 2018–2022; the wind, sea ice and MBL files
aren't read.

To run on NERSC or UW Hyak, use the machine profiles at the bottom of the config; see
[Running on an HPC system](example-workflow-drake.md#6-running-on-an-hpc-system), which also shows how
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

## Depth-resolved oxygen (0–1000 dbar)

`configs/example_atlantic_depth.toml` predicts oxygen on every 20 dbar level from 0 to 1000 dbar
instead of at the surface, for 2020–2021 (training 2020, test 2021). It needs the same external
inputs as the surface example (bathymetry, ERA5 sea level pressure, CMEMS ADT). With pixi:

```bash
pixi run example-atlantic-depth    # L0-L3, P1, clustering, regression prep, models, error summary
```

What changes from the surface example:

- **L1/L3:** oxygen is regridded with T/S in L3 (`[L3] bgc_vars = ["oxygen"]`). Oxygen with a
  QC flag outside `[L1] use_flags` is set to NaN before regridding (`[L3] qc_masked_vars`), and
  the L1 pH filter is off (`bgc_qc_vars = []`), so floats without a pH sensor are kept.
- **Regression prep** (`[regression] depth_resolved = true`, `pressure_levels = [0, 1000, 20]`):
  each profile becomes one row per pressure level. `pressure`, `CT`, `SA`, `sigma0` and `oxygen`
  are the values on that level (from L3); `mlp`, `adt`, `max_solar_rad`, `atmos_pres_atm` and the
  cluster probabilities are the profile's and repeat on every level. Row keys end in the level
  (`_p0100`), and the application set is written as parquet (`[files] reg_application`).
- **Folds:** `cv_type = "platform"`. Depths of a profile, and profiles of a float, are strongly
  correlated, so whole floats are held out.
- **Error summary:** a second table gives the errors by pressure layer (0–100, 100–300, 300–600,
  600–1000 dbar; change with `--pressure-bins`).

The linear calibration is still fit per cluster on all levels pooled, so it mostly reflects the
deep levels, which outnumber the upper ocean. Check the calibrated and uncalibrated errors by
pressure layer before relying on it. In a test on the Drake Passage floats (2018–2021 training,
2022 test, RFR `rfr_test`), the 0–100 dbar test bias was about −10 µmol/kg both with and without
calibration, while below 300 dbar it was under 2 µmol/kg.
