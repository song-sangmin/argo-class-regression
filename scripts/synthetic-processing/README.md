# Synthetic Argo and SOCAT processing (E3SM synthetic data)

Synthetic counterpart of the real-data pipeline in `../data-processing/` and `../clustering/`.
Each output has the same layout as its real-data equivalent, so the shared code
(`clustering/fit_argo_clustering.py`, `clustering/p2_classes.py`, `data-processing/p1_features.py`,
and the SOCAT colocation in `data-processing/create_P1_socat_features.py`) runs on it unchanged.
Replaces notebooks 4.2, 4.4 and the synthetic float part of 4.3.

Every script takes `--config configs/synthetic_CRUSOE.toml`, which lists the synthetic inputs and
outputs (and the real files the synthetic floats are matched to). Run in order, from `PKG-argo-clare/scripts/`:

```bash
C=../configs/synthetic_CRUSOE.toml
python synthetic-processing/create_L3_synthetic_argo.py --config $C        # L3_data, L3_index (core, bgc)
python synthetic-processing/create_P1_synthetic_features.py --config $C    # P1_bgc, P1_core
python synthetic-processing/create_P1_synthetic_socat.py --config $C       # P1_socat
python clustering/fit_argo_clustering.py --config $C                       # P2_finder: GMM refit on the synthetic profiles
python synthetic-processing/apply_P2_synthetic_clustering.py --config $C   # P2_syn_trainval (synBgcArgo + synSOCAT)
python synthetic-processing/apply_P2_synthetic_synCore_clustering.py --config $C   # P2_core (application set)
python regression/regression-input-prep.py --config $C                     # regression csvs (no adt; sea ice clusters in reg_info)
```

The files in use (pc8_gmm6_dbar500) are listed in the config, so each step stops before overwriting
them unless `--overwrite` is passed. On 2026-10-21, the last three steps reproduced the files in use
exactly (the P2 parquet files, and both regression csvs byte for byte).

The synthetic SOCAT input (`[files] syn_socat`) comes from `PKG-synthetic-extract/extract_SOCAT.py`.

Notes
- Core vs bgc: a synthetic float follows a real WMO (`fnum`); bgc = SOCCOM WMOs in the real P1 bgc
  file, core = WMOs in the real P1 core file that are not bgc. Other fnums are dropped.
- Features are from the model (sea_ice, mld, wind speed from wind stress, atmospheric pressure,
  pCO2 in ppmv), plus time variables and clear-sky solar radiation; there is no ADT (no sea
  surface height in the model output). Surface CT/SA are at 5 dbar for floats and SOCAT.
- Synthetic SOCAT mimics the real workflow: it is colocated to the nearest clusterable synthetic
  float profile (CT/SA above dbar_limit, within 7 days) and takes that profile's mld/mlp and class.
  The model MLD at the SOCAT sample is kept as `mld_model` for reference only.
- Synthetic training data is subsampled by year in `apply_P2_synthetic_clustering.py`: a random
  (seeded) subset of synthetic bgc profiles per year, sized to the real SOCCOM samples that year in the
  real P2 trainval file (all kept where there are fewer); synthetic SOCAT is already extracted at the
  real SOCAT samples (`PKG-synthetic-extract/build_socatINDEX.py`). Use `--no-subsample` to keep everything.
- The GMM is refit on the synthetic profiles with the same parameters as the real data, so
  synthetic class numbers (north to south by mean latitude) are not the real classes.
- Full L3 run: ~330k profiles, roughly 45 min with 8 workers.
