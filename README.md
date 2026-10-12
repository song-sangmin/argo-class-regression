# Class-dependent Regression Modeling using Argo data 

Utilities for decision tree regression modeling (clustering, feaure processing, cross-validation) using Argo data


Contact: Song Sangmin <sangsong@uw.edu> \
University of Washington, School of Oceanography\

Contributors: Scott Henderson, Alison Gray\


## Installation

To run locally, clone (or fork) repo. 
Things to run once in terminal, from root folder:

```
mamba env create -f environment.yml  	# argocv-dev
conda activate argocv-dev

python -m pip install -e . 		        # installs argoclare as editable (-e)
```

Common functions for ML development are installed within the conda environment `argocv-dev` as the package [`argocv`](./src/argocv/).
From any notebook with the environment `argocv-dev` activated, can run `import argocv`, see naming conventions in folder.

May have to run `git config filter.nbstripout.clean "nbstripout --keep-output"`.


## Data Sources

- Argo data were accessed with [Argopy](https://argopy.readthedocs.io/en/latest/) from the Jan 2025 GDAC
- Satellite data from [CMEMS](https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047/download) (cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_adt-sla)
- Wind speed from [ERA5 Reanalysis](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview)
- Bathymetry from [ETOPO2](https://cmr.earthdata.nasa.gov/search/concepts/C1214055363-SCIOPS)
- Solar radiation computed from [Solarpy](https://pypi.org/project/solarpy/)
- Atmospheric xCO2 from [NOAA Marine Boundary Layer](https://gml.noaa.gov/ccgg/mbl/data.php) product
- Sea ice concentration from [NSIDC](https://nsidc.org/data/g02202/versions/6), 25km resolution daily
- Sea level pressure from [NCEP/DOE Reanalysis II](https://psl.noaa.gov/data/gridded/data.ncep.reanalysis2.html), 4x daily

## Funding & References

This work is funded by [SOCCOM3](https://soccom.org/).

Argo (2025). Argo float data and metadata from Global Data Assembly Centre (Argo GDAC) – Snapshot of Argo GDAC of January 2025. SEANOE. https://doi.org/10.17882/42182#116315

