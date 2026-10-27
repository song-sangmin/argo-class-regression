# Data processing (L0 to P1)

Each script runs one stage and takes `--config <run.toml>`. 

The config holds the run's settings
(region, years, QC flags, bathymetry limit, resampling, ...) and, in `[files]`, the external
inputs and where each stage reads and writes. The shared code is `argoclare.pipeline`
(`src/argoclare/pipeline.py`). The next steps, clustering and regression inputs, are described
in `../clustering/README.md`.

| Configs | |
|---|---|
| `configs/example.toml` | Small example: Drake Passage, 2018–2022, ERDDAP, core + BGC floats; bathymetry from `pixi run get-example-data`. Outputs go to `x_data/example_drake/`. |
| `configs/southernOcean_CRUSOE.toml` | The CRUSOE dataset: points at the files currently in use (SOCCOM + SOCAT training data). |
| `configs/synthetic_CRUSOE.toml` | The CRUSOE synthetic (E3SM) run (`../synthetic-processing/`). |

To start a new run, copy `example.toml`, give it a new `[run] name`, and point the external inputs
in `[files]` (bathymetry, ERA5, CMEMS ADT, NSIDC sea ice, NOAA MBL; optionally SOCAT and SOCCOM) at
your own copies. Only the bathymetry is always needed. The P1 scripts add the features in
`[P1] features` plus the `[regression] features` and `target`, and read only the gridded files
those need (`FILE_COLUMNS` in `p1_features.py`; see "Download the external inputs" in
`example-workflow-drake.md`). SOCCOM and SOCAT P1 always need ERA5 sea level pressure and the MBL CO2,
for `delta_pco2`. `create_mbl_co2_dataset.py` is only needed when the MBL CO2 is.

| Stage | Script | `[files]` key |
|---|---|---|
| L0 raw Argo | `create_L0_raw_argo.py --dataset core\|bgc` | `L0_profiles`, `L0_index` |
| L1 QC | `create_L1_QC_argo.py --dataset core\|bgc` | `L1` |
| L2 open ocean | `create_L2_masked_argo.py --dataset core\|bgc` | `L2` |
| L3 regridded | `create_L3_gridded_coreArgo.py`, `create_L3_gridded_bgcArgo.py` | `L3_data`, `L3_index` |
| MBL CO2 (if needed) | `create_mbl_co2_dataset.py` | `mbl` |
| P1 float features | `create_P1_float_features.py --dataset core\|bgc` | `P1_core`, `P1_bgc` |
| P1 SOCCOM (optional) | `create_P1_soccom_features.py` | `P1_soccom` |
| L2 SOCAT (optional) | `create_L2_resampled_masked_socat.py` | `socat_L2` |
| P1 SOCAT (optional) | `create_P1_socat_features.py` | `P1_socat` |

The CRUSOE run uses the SOCCOM surface averages (`P1_soccom`) for its BGC training data, so it
does not run `create_P1_float_features.py --dataset bgc`.

Each run has a folder, `<data_root>/<name>/` (or `[run] run_dir`), holding:
- `config.toml`: a copy of the config made by the first script run. A later run with different
  settings is refused unless you pass `--update-config`, so the folder always matches one config.
- `provenance/`: one json per script run, with the command, git commit, package versions,
  machine, and the inputs and outputs.
- the stage outputs (`L0/`, `L1-qc/`, `L2-mask/`, `L3-interp/`, `socat/`, `ancillary/`, `P1/`, `P2/<pcm>/`,
  `regression/<pcm>/`), unless `[files]` points them elsewhere.

Existing outputs are never replaced without `--overwrite`. For quick tests, use `--years`,
the subset flags (`--floats`, `--max-profiles`, `--max-samples`, `--max-cruises`) and
`--out-dir /tmp/test/`.

Example: `python create_L2_resampled_masked_socat.py --config ../../configs/southernOcean_CRUSOE.toml --max-cruises 20 --out-dir /tmp/test/` 

`python create_L3_gridded_bgcArgo.py --config ../../configs/example.toml --years 2020 --max-profiles 200 --out-dir /tmp/test/` 


On another machine (e.g. NERSC), add a `[machines.<name>]` table to the config with that machine's
paths, and set `ARGOCLARE_MACHINE=<name>` there (or pass `--machine <name>`); see
`../clustering/README.md`.

In a notebook, `pipeline.load()` gives the same paths without writing anything:

```python
from argoclare import pipeline
run = pipeline.load('configs/southernOcean_CRUSOE.toml')
run.status()                         # every [files] entry and whether it exists
coreINDEX = run.open('L3_index', 'core')
bgcL1 = run.open('L1', 'bgc', yr=2018)
figpath = run.data_root + '/figures/'
```
