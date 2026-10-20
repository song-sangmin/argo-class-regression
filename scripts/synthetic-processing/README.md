# Synthetic Argo and SOCAT processing (E3SM synthetic data)

Synthetic counterpart of the real-data pipeline in `../data-processing/` and `../clustering/`.
Each output has the same layout as its real-data equivalent, so the shared code
(`clustering/fit_argo_clustering.py`, `clustering/p2_classes.py`, `data-processing/p1_features.py`,
and the SOCAT colocation in `data-processing/create_P1_socat_features.py`) runs on it unchanged.
Replaces notebooks 4.2, 4.4 and the synthetic float part of 4.3.

Run in order (paths relative to `PKG-argo-clare/scripts/`; outputs in `x_localdata/working-vars/synthetic/`):

| Step | Command | Output |
|---|---|---|
| L3 | `python synthetic-processing/create_L3_synthetic_argo.py` | `L3-interp/synthetic_{core,bgc}{DATA,INDEX}_byProfid_<years>_acc<date>.nc` |
| P1 floats | `python synthetic-processing/create_P1_synthetic_features.py --bgc-index <bgc INDEX> --core-index <core INDEX>` | `P1-processed/P1_synArgo_{bgc,core_application}_features_yr<years>_acc<date>.nc` |
| P1 SOCAT | `python synthetic-processing/create_P1_synthetic_socat.py --indexes <core INDEX> <bgc INDEX> --datas <core DATA> <bgc DATA>` | `P1-processed/P1_synSOCAT_features_yr<years>_acc<date>.nc` |
| fit | `python clustering/fit_argo_clustering.py --core <core DATA> <core INDEX> --bgc <bgc DATA> <bgc INDEX> --outpath <working-vars>/synthetic/P2-clustered/` | `P2-clustered/pc8_gmm6_dbar500/PCM_finder_..._acc<date>.parquet` (+ `PCM_fit_...json`) |
| P2 trainval | `python synthetic-processing/apply_P2_synthetic_clustering.py --pcm-finder <PCM_finder> --bgc-p1 <P1 bgc> --socat-p1 <P1 SOCAT>` | `P2-clustered/pc8_gmm6_dbar500/P2_synthetic_combined_trainval_socat_bgc_<years>_acc<date>.parquet` |
| P2 application | `python synthetic-processing/apply_P2_synthetic_synCore_clustering.py --pcm-finder <PCM_finder> --core-p1 <P1 core>` | `P2-clustered/pc8_gmm6_dbar500/P2_synArgo_core_application_<years>_acc<date>.parquet` |

The synthetic SOCAT input (`synSOCAT-daily_matches.nc`) comes from `PKG-synthetic-extract/extract_SOCAT.py`.

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
