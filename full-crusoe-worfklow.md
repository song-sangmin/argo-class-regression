# Full CRUSOE workflow

Run from `scripts/`. Every step reads its settings and files from
`configs/southernOcean_CRUSOE.toml`, which points at the files currently in use, so each command
stops at the first output that already exists. Pass `--overwrite` to regenerate a stage in place,
or copy the config with a new `[run] name` to build a separate run.

```bash
C=../configs/southernOcean_CRUSOE.toml

# L0-L3 floats
python data-processing/create_L0_raw_argo.py  --config $C --dataset core          # GDAC snapshot (/Volumes/ocean-repo)
python data-processing/create_L0_raw_argo.py  --config $C --dataset bgc           # ERDDAP (changes daily)
python data-processing/create_L1_QC_argo.py   --config $C --dataset core
python data-processing/create_L1_QC_argo.py   --config $C --dataset bgc
python data-processing/create_L2_masked_argo.py --config $C --dataset core
python data-processing/create_L2_masked_argo.py --config $C --dataset bgc
python data-processing/create_L3_gridded_coreArgo.py --config $C
python data-processing/create_L3_gridded_bgcArgo.py  --config $C

# P1 features (SOCCOM instead of BGC float profiles)
python data-processing/create_mbl_co2_dataset.py --config $C
python data-processing/create_L2_resampled_masked_socat.py --config $C
python data-processing/create_P1_float_features.py --config $C --dataset core
python data-processing/create_P1_soccom_features.py --config $C
python data-processing/create_P1_socat_features.py  --config $C

# P2 clustering and regression inputs (clustering/README.md)
python clustering/fit_argo_clustering.py --config $C
for ds in soccom socat core; do python clustering/apply_P2_classes.py --config $C --dataset $ds; done
python regression/regression-input-prep.py --config $C

# Model optimization on NERSC (model-optimization/README.md); the sbatch scripts set ARGOCLARE_MACHINE=nersc
cd model-optimization
sbatch -J RFR_full submit_nested_cv.sbatch RFR rfr_full --config ../../configs/southernOcean_CRUSOE.toml
```

Another clustering (e.g. 7 classes) reuses L0-P1. Pass the same options to every P2 step:
`--n-gmm 7` for fit, apply, regression prep and model-optimization (see `clustering/README.md`).

Subsets for testing: `--years 2020`, `--floats <wmo ...>` (L1), `--max-profiles N` (L3, P1 floats),
`--max-samples N` (P1 SOCCOM/SOCAT), `--max-cruises N` (SOCAT L2), with `--out-dir /tmp/test/` so
the outputs do not go to the configured paths.

The synthetic (E3SM) run uses `configs/synthetic_CRUSOE.toml`; see `synthetic-processing/README.md`.
