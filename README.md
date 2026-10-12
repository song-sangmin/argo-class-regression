# CRUSOE-SOCarbon: ML regression modeling of surface ocean carbon using sparse observations


Class-conditional Regression Using in Situ observations for Ocean Estimation (CRUSOE) \
Contact: Song Sangmin <sangsong@uw.edu> \
University of Washington, School of Oceanography\

Contributors: Alison Gray, Ola Formo-Kihle, Ken Johnson\
Last updated: Jun 2026


## Installation

To run locally, clone (or fork) repo. 
Things to run once in terminal, from root folder:

```
mamba env create -f environment.yml  	# crusoe-dev now 3.11
conda activate crusoe-dev

pre-commit install   					# for nbstripout pre-commit hook
nbstripout install
python -m pip install -e . 		        # installs crusoe and crplot as editable (-e)

```

Common functions for ML development are installed within the conda environment `crusoe-dev` as the package [`crusoe`](./src/crusoe/), with functions for plotting as the package [`crplot`](./src/crplot/).
From any notebook with the environment `crusoe-dev` activated, can run `import crusoe`, see naming conventions in folder.

May have to run `git config filter.nbstripout.clean "nbstripout --keep-output"`.


## Notebooks

Code directory in [Notebooks](./notebooks/) describes main workflow steps.
Main file for model development is:

- [`3.3_MAIN_regression_nested_crossval_tuning.ipynb`](./notebooks/3.3_MAIN_regression_nested_crossval_tuning.ipynb) cross-validation with nested parameter tuning and comparison of RFR, XGB, ERT

Datasets can be accessed from notebooks using functions from the central `data_loader.py` (update these datetags as selected params change). Convention is currently `import data_loader as loader`, note if you clone this you should update the `root_data_folder` in the loader file. 


<!-- ### Folder Directory

- `scripts/` : code for analysis
- `data/` : float, ship, and glider data as downloaded
- `working-vars/` : calculated output variables from analysis
- `images/` : final output figures
 -->

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

Copernicus Climate Change Service, Climate Data Store, (2023): ERA5 hourly data on single levels from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). DOI: 10.24381/cds.adbb2d47 (Accessed on 25-03-2026)

Lan, X., Tans, P., Thoning, K., & NOAA Global Monitoring Laboratory. (2024). NOAA Greenhouse Gas Marine Boundary Layer Reference - CO2. [Data set]. NOAA GML. https://doi.org/10.15138/DVNP-F961

Meier, W. N., Fetterer, F., Windnagel, A. K., Stewart, J. S. & Stafford, T. (2026). NOAA/NSIDC Climate Data Record of Passive Microwave Sea Ice Concentration. (G02202, Version 6). [Data Set]. Boulder, Colorado USA. National Snow and Ice Data Center. https://doi.org/10.7265/b18j-z797. [describe subset used if applicable]. Date Accessed 06-15-2026.

