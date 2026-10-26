# Example workflow: from Argo floats to a clustered regression model

This guide runs the whole pipeline on a small example: one year (2020) of core and BGC Argo floats
in the Drake Passage (70°W–50°W, 65°S–55°S, 0–2000 dbar). The example clusters all float profiles
by their temperature and salinity structure, then trains one regression model per cluster. Each
model predicts surface **oxygen** on BGC floats from **CT, SA, ADT, max_solar_rad and
atmos_pres_atm**, and the trained models are applied to the core floats, which have no oxygen
sensor.

Every setting lives in one file, `configs/example.toml`. Every script takes `--config <file>`,
writes to a run folder, and never overwrites existing outputs unless you pass `--overwrite`.

## 1. Install

```bash
git clone https://github.com/song-sangmin/argo-class-regression.git
cd argo-class-regression
```

Then either:

```bash
pixi shell                                  # pixi: environment and argoclare installed automatically
```

or:

```bash
mamba env create -f environment.yml && conda activate argocv-dev
python -m pip install -e .                  # installs the argoclare package (editable)
```

Python 3.11 or newer is needed, because the configs are read with `tomllib`.

## 2. Make your run config

```bash
cp configs/example.toml configs/example_drake.toml
```

In `configs/example_drake.toml`:

- `[run] name`: a new name, e.g. `"example_drake2020"`. Outputs go to `<data_root>/<name>/`. The default
  `data_root = "../x_data"` is relative to the config file, so that's `x_data/` in the repo (ignored by git).
- `[region]`, `[run] years`: the area, depth range and years to download.
- `[regression]`: what to predict and from what. The example uses
  `target = "oxygen"` and `features = ["CT", "SA", "adt", "max_solar_rad", "atmos_pres_atm"]`.
  Only BGC samples whose `oxygen_qc` is in `target_qc_flags` (1, 2, 5, 8) are used for training.
- `[files]`: where your copies of the external inputs are (step 3).

## 3. Download the external inputs

The Argo floats are downloaded by the pipeline itself (step 4). The P1 step adds environmental
features from gridded products, and it computes all of them, so every file below is needed for
each run year, even for features your regression doesn't use. Put them where `[files]` says, by
default `x_data/external/`, or change the paths.

| `[files]` key | Product | Notes |
|---|---|---|
| `bathy` | GEBCO 2024 grid, sub-ice topography (netcdf) — https://www.gebco.net | variable `elevation`; masks out profiles shallower than 1000 m |
| `era5_slp` | ERA5 hourly single levels, mean sea level pressure (`msl`) — Copernicus Climate Data Store | one file per year (`{yr}` in the name), dims `valid_time, latitude, longitude` |
| `era5_wind` | ERA5 hourly single levels, 10 m u and v wind (`u10`, `v10`) | one file per year |
| `adt` | CMEMS DUACS L4 sea surface height, daily 0.125° (`cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`), variable `adt` — Copernicus Marine | one file per year |
| `seaice` | NSIDC sea ice concentration CDR v6, daily, southern hemisphere 25 km (`cdr_seaice_conc`) | one file per year |
| `mbl_txt` | NOAA GML marine boundary layer CO2, surface, sine-latitude steps (text) — https://gml.noaa.gov/ccgg/GHGreference/ | must cover your years and sin(lat) −1.00 to −0.50 |

For a region outside the Southern Ocean, the bathymetry, ERA5, ADT and MBL files must cover it.
The sea ice file is for the southern hemisphere only.

## 4. Run the pipeline

Run the commands from the repo root. `C` is your config.

```bash
C=configs/example_drake.toml
```

**Floats: download, QC, open-ocean mask, regrid (L0–L3).** L0 downloads from the Ifremer ERDDAP
server.

```bash
python scripts/data-processing/create_L0_raw_argo.py --config $C --dataset core
python scripts/data-processing/create_L0_raw_argo.py --config $C --dataset bgc
python scripts/data-processing/create_L1_QC_argo.py --config $C --dataset core
python scripts/data-processing/create_L1_QC_argo.py --config $C --dataset bgc
python scripts/data-processing/create_L2_masked_argo.py --config $C --dataset core
python scripts/data-processing/create_L2_masked_argo.py --config $C --dataset bgc
python scripts/data-processing/create_L3_gridded_coreArgo.py --config $C
python scripts/data-processing/create_L3_gridded_bgcArgo.py --config $C
```

**Features at each profile (P1).** The MBL step combines the NOAA text file(s) into one netcdf first.

```bash
python scripts/data-processing/create_mbl_co2_dataset.py --config $C
python scripts/data-processing/create_P1_float_features.py --config $C --dataset core
python scripts/data-processing/create_P1_float_features.py --config $C --dataset bgc
```

**Clustering (P2).** The fit uses core and BGC profiles together (`[P2] fit_datasets`). Classify
then writes one file per platform with class probabilities and assignments.

```bash
python scripts/clustering/fit_argo_clustering.py --config $C
python scripts/clustering/apply_P2_classes.py --config $C --dataset core
python scripts/clustering/apply_P2_classes.py --config $C --dataset bgc
```

**Regression inputs.** This step:
- uses the BGC floats as training data (`[regression] train`)
- splits them into training and test years
- removes from the core set any profiles that are also training samples
- writes the csvs the models read

```bash
python scripts/regression/regression-input-prep.py --config $C
```

**Models.** Nested cross-validation picks hyperparameters and estimates the error per cluster.
The application run trains the final models and predicts oxygen for the core floats. Run these
from `scripts/model-optimization/`, so the parallel workers can import the helper modules.
`rfr_test` is a small grid for trying things out; `rfr_full` is the full one (see `tuning_grids.py`).

```bash
cd scripts/model-optimization
python run_dask_nested_cv.py RFR rfr_test --config ../../$C --n-outer 2 --n-inner 2
python run_model_application.py --config ../../$C
```

`--config` gives these scripts the data files, the results folder (`<run_dir>/model-optimization/<pcm>/`),
the target and the features.

> **For a ready-made larger run**, see [example-workflow-atlantic.md](example-workflow-atlantic.md):
> the Atlantic sector, 2018–2022, 2 clusters, with measured run times.
>
> **The model step needs enough training data.** The Drake Passage in 2020 has only about 6 BGC
> floats (about 50 profiles) with good oxygen. That's enough to run and inspect the L0–P2 steps,
> but thin for training and validating per-cluster models. This example uses
> `[regression] cv_type = "profile"`, which holds out whole profiles instead of whole floats.
> That needs fewer floats, but its error estimates are optimistic (see
> [example-workflow-atlantic.md](example-workflow-atlantic.md#choosing-the-cross-validation-folds-cv_type)).
> With `cv_type = "platform"`, every cluster needs several floats. For example, the whole
> Southern Ocean (south of 35°S) for 2014–2023 has 187 BGC floats with good surface oxygen
> (at least 34 per cluster with 6 clusters); that setup was tested on 2026-10-23, with a
> cross-validated RMSE of 8.0 µmol/kg against an oxygen standard deviation of 34.9 µmol/kg.

## 5. What you get

```
x_data/example_drake2020/
    config.toml          copy of the config the run was made with
    provenance/          one json per script run: command, git commit, package versions, inputs, outputs
    L0/ L1-qc/ L2-mask/ L3-interp/  floats at each processing level
    ancillary/           combined MBL CO2
    P1/                  P1_core_features.nc, P1_bgc_features.nc: one row per profile, with features
    P2/pc4_gmm3_dbar500/ PCM_finder (classes of every profile), P2_core / P2_bgc (P1 + class probabilities),
                         and the regression trainval and application sets
    regression/pc4_gmm3_dbar500/   csvs for model-optimization, and regression_info (features, target)
    model-optimization/pc4_gmm3_dbar500/   CV errors, best parameters, predictions for the core floats
```

`pc4_gmm3_dbar500` is the clustering tag: 4 PCA components, 3 GMM classes, profile levels above
500 dbar. In Python:

```python
from argoclare import pipeline
run = pipeline.load('configs/example_drake.toml')
run.status()                          # every file the config names, and whether it exists
p2_bgc = run.open('P2', 'bgc')        # BGC profiles with class probabilities
```

## 6. Running on an HPC system

Everything above runs on a laptop. The data steps (L0–P2) are small enough for one, but the
model step with the full tuning grids (`rfr_full`, `xgb_full`) is much faster on a cluster node.
You can run the whole pipeline on the cluster, or process the data on your laptop and run only
the models on the cluster.

**Machine profiles.** The same config file works on several computers. A `[machines.<name>]`
table at the bottom of the config replaces settings on that machine only:

```toml
[machines.nersc.run]
data_root = "$PSCRATCH/argoclare-data"
```

You choose the profile on each machine, in `~/.bashrc` or in the job script:

```bash
export ARGOCLARE_MACHINE=nersc          # or pass --machine nersc to any script
```

Without it, the settings above the profiles are used as written, so nothing changes on your
laptop. Every path built from `{data_root}` moves with it: the run folder
(`<data_root>/<name>/`, with all outputs) and the external inputs
(`<data_root>/external/`). Paths can use environment variables (`$PSCRATCH`, `$HOME`, `~`). A
variable that isn't set stops the script with a message, so it never writes somewhere
unexpected. The run folder's copy of the config still matches on both machines, because it's
the same file.

**Profiles in the example configs:**
- **NERSC (Perlmutter), active:** `data_root = "$PSCRATCH/argoclare-data"`, which works for any
  NERSC user. Scratch is purged periodically, so for data you want to keep, change it to your
  project's CFS folder (`/global/cfs/cdirs/<project>/<user>/argoclare-data`).
- **UW Hyak (klone), commented out:** uncomment it and fill in `<group>` (your lab's Hyak account)
  and `<netid>`. Data and the run folder go to your lab's `/gscratch/<group>/` storage. Model
  results go to `/gscratch/scrubbed/`, the shared scratch space that is cleaned automatically, so
  only put results there that you can regenerate.

**For another cluster:** copy one of those tables, rename it (`[machines.mycluster.run]`), set
`data_root` to storage that keeps your files, and set `ARGOCLARE_MACHINE=mycluster` on that
cluster. Any other setting can go in the profile too. For example, `[machines.mycluster.files]
model_results = "/scratch/<user>/argoclare_runs/{pcm}"` sends the model results to fast scratch,
and `[machines.mycluster.P1] n_jobs = 16` uses more cores.

**Running only the models on the cluster.** Process the data on your laptop, then copy the run
folder (or just its `regression/` folder) to the same place under the cluster's `data_root`, and run
the model scripts there with the same config:

```bash
rsync -av x_data/example_drake2020/ <cluster>:<data_root>/example_drake2020/
```

**Job scripts.** `scripts/model-optimization/submit_nested_cv.sbatch` and
`submit_model_application.sbatch` run the model scripts as Slurm jobs. They work on any cluster
without editing. Their defaults are for NERSC Perlmutter, and you replace them on other machines when
you submit:

| What | NERSC default | On another cluster |
|---|---|---|
| Slurm account, QOS, node type, time | `#SBATCH --account=m4839`, `--qos=regular`, `--constraint=cpu`, `--time=01:00:00` | `sbatch` flags or `SBATCH_*` variables (below) |
| Log file | `logs/%x_%j.out` in `scripts/model-optimization/` | the same, or `sbatch -o <folder>/%x_%j.out` |
| Environment setup | `module load python && conda activate dask-cv` | `export ARGOCLARE_ENV_SETUP='<your commands>'`, or `""` to use the environment you submit from |
| Where the job runs | the folder you submit from (`scripts/model-optimization/`) | the same, or `export ARGOCLARE_SRC_DIR=<that folder>` |
| Config profile | `ARGOCLARE_MACHINE=nersc` | `export ARGOCLARE_MACHINE=<name>` |

Always submit from `scripts/model-optimization/`. A job submitted from elsewhere stops right away
with a message, unless `ARGOCLARE_SRC_DIR` is set.

**On NERSC**, with nothing set:

```bash
cd scripts/model-optimization
sbatch submit_nested_cv.sbatch RFR rfr_full --config ../../configs/example_drake.toml
```

**On another cluster**, options given to `sbatch` on the command line override the `#SBATCH`
lines in the script. For example, on UW Hyak:

```bash
cd scripts/model-optimization
export ARGOCLARE_MACHINE=hyak                    # the config's [machines.hyak] profile
export ARGOCLARE_ENV_SETUP='eval "$(conda shell.bash hook)" && conda activate argocv-dev'
sbatch -A <group> -p <partition> -q <qos> -C <feature> -t 02:00:00 -c <cores> --mem=<memory> \
       submit_nested_cv.sbatch RFR rfr_full --config ../../configs/example_drake.toml
```

| Flag | Replaces | Notes |
|---|---|---|
| `-A <group>` | `--account=m4839` | your project or lab account (on Hyak, your lab's account) |
| `-p <partition>` | (none) | e.g. `compute` or `ckpt` on Hyak; `hyakalloc` lists what your group can use |
| `-q <qos>` | `--qos=regular` | use a QOS your cluster has |
| `-C <feature>` | `--constraint=cpu` | use a node feature your cluster has |
| `-t`, `-c`, `--mem` | `--time=01:00:00` | job length and resources |
| `-o <folder>/%x_%j.out` | `--output=logs/%x_%j.out` | optional; the folder must exist |

`ARGOCLARE_ENV_SETUP` holds whatever commands make `python` and `argoclare` available in a job.
The example above is one way to do it with conda; use your cluster's module or container commands
if that's how it provides Python. If you activate the environment before submitting, set
`ARGOCLARE_ENV_SETUP=""`, because `sbatch` passes your current environment on to the job.

Settings you use every time can go in `~/.bashrc` on that cluster. Slurm reads the `SBATCH_*`
variables on every `sbatch` call, and command-line flags still override them:

```bash
export SBATCH_ACCOUNT=<group> SBATCH_PARTITION=<partition>
export ARGOCLARE_MACHINE=hyak
export ARGOCLARE_ENV_SETUP='eval "$(conda shell.bash hook)" && conda activate argocv-dev'
```

`submit_algorithm_comparison.sh` (one nested CV job per algorithm) passes all of these on to its
jobs. If you'd rather not set them each time, you can still copy a job script and change its
`#SBATCH` lines and the `ARGOCLARE_ENV_SETUP` default. Check the cluster's own documentation for
its current partitions, storage and scratch-cleaning rules.

## Tips

- **Test on a subset first.** Most scripts take `--years`, a subset flag (`--floats` for L1,
  `--max-profiles` for L3 and P1), and `--out-dir /tmp/test/` to write somewhere else.
- **Changing settings after a run started:** the scripts refuse a config that differs from the
  run folder's copy. Use a new `[run] name`, or pass `--update-config`; the old copy is kept.
- **Trying other clusterings** (e.g. `--n-gmm 4`) doesn't need L0–P1 again. Pass the same options
  to every P2, regression and model step; each clustering gets its own folder
  (see `scripts/clustering/README.md`).
- **Another computer or a cluster:** see section 6.
- **The full Southern Ocean (CRUSOE) setup** (SOCCOM and SOCAT training data, pCO2 target) is in
  `configs/southernOcean_CRUSOE.toml` and `full-crusoe-worfklow.md`.
