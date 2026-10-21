# Full CRUSOE workflow

Run from `scripts/data-processing/`. Every step reads its settings and files from
`configs/southernOcean_CRUSOE.toml`, which points at the files currently in use, so each command
stops at the first output that already exists. Pass `--overwrite` to regenerate a stage in place,
or copy the config with a new `[run] name` to build a separate run.

```bash
C=../../configs/southernOcean_CRUSOE.toml

python create_L0_raw_argo.py  --config $C --dataset core          # GDAC snapshot (/Volumes/ocean-repo)
python create_L0_raw_argo.py  --config $C --dataset bgc           # ERDDAP (changes daily)
python create_L1_QC_argo.py   --config $C --dataset core
python create_L1_QC_argo.py   --config $C --dataset bgc
python create_L2_masked_argo.py --config $C --dataset core
python create_L2_masked_argo.py --config $C --dataset bgc
python create_L3_gridded_coreArgo.py --config $C
python create_L3_gridded_bgcArgo.py  --config $C

python create_mbl_co2_dataset.py --config $C
python create_L2_resampled_masked_socat.py --config $C
python create_P1_core_features.py  --config $C
python create_P1_bgc_features.py   --config $C
python create_P1_socat_features.py --config $C
```

Subsets for testing: `--years 2020`, `--floats <wmo ...>` (L1), `--max-profiles N` (L3, P1 core),
`--max-samples N` (P1 bgc/socat), `--max-cruises N` (SOCAT L2), with `--out-dir /tmp/test/` so the
outputs do not go to the configured paths.
