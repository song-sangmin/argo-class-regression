# Data processing (L0 to P1)

Each script runs one stage and takes `--config <run.toml>`. The config holds the run's settings
(region, years, QC flags, bathymetry limit, resampling, ...) and, in `[files]`, the external
inputs and where each stage reads and writes. See `pipeline.py` for the details.

| Configs | |
|---|---|
| `configs/example.toml` | Small example: Drake Passage, 2020, ERDDAP. Outputs go to `x_data/example_drake2020/`. |
| `configs/southernOcean_CRUSOE.toml` | The CRUSOE dataset: points at the files currently in use. |

To start a new run, copy `example.toml`, give it a new `[run] name`, and point the external inputs
in `[files]` (bathymetry, ERA5, CMEMS ADT, NSIDC sea ice, SOCCOM, SOCAT, NOAA MBL) at your own copies.

| Stage | Script | `[files]` key |
|---|---|---|
| L0 raw Argo | `create_L0_raw_argo.py --dataset core\|bgc` | `L0_profiles`, `L0_index` |
| L1 QC | `create_L1_QC_argo.py --dataset core\|bgc` | `L1` |
| L2 open ocean | `create_L2_masked_argo.py --dataset core\|bgc` | `L2` |
| L3 regridded | `create_L3_gridded_coreArgo.py`, `create_L3_gridded_bgcArgo.py` | `L3_data`, `L3_index` |
| MBL CO2 | `create_mbl_co2_dataset.py` | `mbl` |
| L2 SOCAT | `create_L2_resampled_masked_socat.py` | `socat_L2` |
| P1 features | `create_P1_core_features.py`, `create_P1_bgc_features.py`, `create_P1_socat_features.py` | `P1_core`, `P1_bgc`, `P1_socat` |

Each run has a folder, `<data_root>/<name>/` (or `[run] run_dir`), holding:
- `config.toml`: a copy of the config made by the first script run. A later run with different
  settings is refused unless you pass `--update-config`, so the folder always matches one config.
- `provenance/`: one json per script run, with the command, git commit, package versions,
  and the inputs and outputs.
- the stage outputs (`L0/`, `L1-qc/`, `L2-mask/`, `L3-interp/`, `socat/`, `ancillary/`, `P1/`), unless
  `[files]` points them elsewhere.

Existing outputs are never replaced without `--overwrite`. For quick tests, use `--years`,
the subset flags (`--floats`, `--max-profiles`, `--max-samples`, `--max-cruises`) and
`--out-dir /tmp/test/`.
