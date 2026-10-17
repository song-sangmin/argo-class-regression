import pandas as pd
import xarray as xr
import numpy as np
import PyCO2SYS as pyco2
from solarpy import irradiance_on_plane
import pyproj
import os
from datetime import datetime, timedelta

from argoclare import argotools as mod_argo
from argoclare import oceantools as mod_ocean
import gsw
from tqdm import tqdm

root_data_folder = '/Volumes/crusoe-repo/data/' 
local_working_folder = '/Users/sangminsong/Documents/PROJECTS-PYTHON/CRUSOE-SOCarbon/x_localdata/'


# def print_dict_counts(plat_dict):
# %% miscellaneous useful

def print_bounds(platDF, show_spatial=False) -> None:
    print('Sample count: ' + '\t' + str(len(platDF)))
    print('\tDates: \t' + str(platDF.datetime.min()) + ' to ' + str(platDF.datetime.max()))
    if show_spatial:
        print('Latitude:\t' + str(platDF.latitude.min()) + ' to ' + str(platDF.latitude.max()))
        print('Longitude:\t' + str(platDF.longitude.min()) + ' to ' + str(platDF.longitude.max()))

def match_profid_from_datetag(soccomDF, floatINDEX):
    """ 
    Match SOCCOM 20m averages with 'profid' (WMO + cycle number) from bgcINDEX or coreINDEX 
    Use 'prof_datetag' (WMO + YYYYMMDD) as indexer
    @param      soccomDF with a prof_datetag column
                floatINDEX has coordinated 'profid'
    
    """
     # Add prof_datetag to the dataset for matching
    finder_index = index_by_datetag(floatINDEX, type='dataset')
    finder_index['profid'] = finder_index['profid'].apply(lambda x: str(x).replace('test','') if 'test' in str(x) else str(x))
    soccomDF['profid'] = soccomDF['prof_datetag'].apply(lambda x: finder_index.loc[str(x)]['profid'] if x in finder_index.index.tolist() else np.nan)

    missing = []
    for x in soccomDF['prof_datetag'].values:
        if str(x) not in finder_index.index.tolist():
            missing.append(x)
        
    soccomDF = soccomDF.dropna(subset=['profid']).copy().set_index('profid')

    return soccomDF, missing

def match_profid_nearest_time(soccomDF, floatINDEX, tolerance='1h'):
    """
    Same as match_profid_from_datetag(), but matches on the nearest profile time
    of the same float within a tolerance, instead of an exact 'prof_datetag'
    (WMO + YYYYMMDDHH). Catches profiles whose times sit a few minutes apart
    across an hour boundary (e.g. 08:59 vs 09:01), which give different datetags.
    @param      soccomDF with 'wmoid', 'datetime' and 'prof_datetag' columns
                floatINDEX has coordinate 'profid' and variables 'wmoid', 'datetime'
                tolerance: max time separation allowed, e.g. '1h', '30min'
    @return     soccomDF indexed by matched 'profid', with column 'time_sep_min'
                (minutes between SOCCOM and floatINDEX profile times)
                missing: list of prof_datetags with no profile within tolerance
    """
    finder = pd.DataFrame({
        'profid': [str(x).replace('test', '') for x in floatINDEX['profid'].values],
        'wmoid': floatINDEX['wmoid'].values.astype(float).astype(int).astype(str),
        'prof_time': pd.to_datetime(floatINDEX['datetime'].values.astype(str), format='mixed').astype('datetime64[ns]'),
    }).sort_values('prof_time')

    socDF = soccomDF.drop(columns=['profid'], errors='ignore').copy() # match_profid_from_datetag adds 'profid' in place
    socDF['wmoid'] = socDF['wmoid'].astype(str)
    socDF['datetime'] = pd.to_datetime(socDF['datetime']).astype('datetime64[ns]')
    socDF['_order'] = np.arange(len(socDF)) # keep original row order

    # For each SOCCOM row, nearest profile in time from the same float
    matched = pd.merge_asof(socDF.sort_values('datetime'), finder,
                            left_on='datetime', right_on='prof_time', by='wmoid',
                            direction='nearest', tolerance=pd.Timedelta(tolerance))
    matched['time_sep_min'] = (matched['datetime'] - matched['prof_time']).abs().dt.total_seconds() / 60
    # Matches that the exact datetag would have missed (times fall in different hours)
    found = matched['profid'].notna()
    n_new = (matched.loc[found, 'prof_datetag'] !=
             matched.loc[found, 'wmoid'] + '_' + matched.loc[found, 'prof_time'].dt.strftime('%Y%m%d%H')).sum()
    matched = matched.sort_values('_order').drop(columns=['_order', 'prof_time'])

    missing = matched.loc[matched['profid'].isna(), 'prof_datetag'].tolist()
    n_dup = matched['profid'].dropna().duplicated().sum()
    if n_dup > 0:
        print(f'Warning: {n_dup} SOCCOM rows matched a profid already used by another row')
    print(f'Matched {found.sum()} of {len(matched)} rows within {tolerance} '
          f'({n_new} not matched by exact datetag); {len(missing)} missing')

    soccomDF = matched.dropna(subset=['profid']).set_index('profid')

    return soccomDF, missing

def add_datetag_from_profid(platDF, floatINDEX):
    """ Reverse of "match_profid_from_datetag"""
    datetag_finder = index_by_datetag(floatINDEX.to_dataframe()).reset_index().set_index('profid')
    platDF['prof_datetag'] = platDF['profid'].apply(lambda x: datetag_finder.loc[str(x)]['prof_datetag'] if str(x) in datetag_finder.index.tolist() else np.nan)


def index_by_datetag(platDF, type='dataset'):
    """ 
    bgcINDEX is an xr dataset
    make pd Dataframe indexed by prof_datetag so you can assign consistent profids to soccom 20m aves"""

    # finder = platDF.copy()
    if type == 'dataset':
        finder = platDF.to_dataframe()
    else: finder = platDF.copy()
    finder['datetime'] = finder['datetime'].astype('datetime64[ns]')
    # finder['prof_datetag'] = finder['profid'].apply(lambda x: str(x).split('_')[0])

    finder['prof_datetag'] = finder.apply(lambda row: str(int(row.wmoid)) + '_' + row.datetime.strftime('%Y%m%d%H'), axis=1)
    finder = finder.reset_index().set_index('prof_datetag')
    # finder['prof_datetag'] = finder.apply(lambda row: row.prof_datetag + '_' + row.datetime[:10].replace('-',''), axis=1)
    # finder_index = finder.reset_index().groupby('prof_datetag').first()

    return finder

# moved to mod_argo
# def datenum_to_datetime(matlab_datenum):
#   days = matlab_datenum % 1
#   dt = datetime.fromordinal(int(matlab_datenum)) + timedelta(days=days) - timedelta(days=366)
#   return dt

# moved to mod_ocean
# def add_mixedlayer_pressure(platINDEX, platDS):
#     """ Originally from 2.0_process_pco2_data
#     """
#     # Calculate mixed layer pressure 
#     plat_mlps, no_data = mod_argo.calc_mlp(platDS, threshold=0.03)
#     plat_mlps['mld'] = gsw.z_from_p(plat_mlps.mlp.values, platINDEX.sel(profid=plat_mlps.index).latitude.values)

#     # Select out only valid
#     platINDEX = platINDEX.sel(profid=plat_mlps.index)
#     platDS = platDS.sel(profid=plat_mlps.index)

#     platINDEX['mld']= -xr.DataArray(plat_mlps.mld.values, dims='profid') # Notice negative
#     platDS['mld']= -xr.DataArray(plat_mlps.mld.values, dims='profid') # Notice negative
#     platINDEX['mlp'] = xr.DataArray(plat_mlps.mlp.values, dims='profid')
#     platDS['mlp'] = xr.DataArray(plat_mlps.mlp.values, dims='profid')

#     return platINDEX, platDS

# %% bgcArgo P1 processing: SOCCOM 20m averages
# 1.2_P1_bgcArgo_feature_prep.ipynb

def process_soccom_20m_averages(soccom_df, start_date = '2013-12-31', end_date = '2023-12-31', use_var='pCO2_pHbias5_pK1'): #, use_cols=None):
    """ 
    Select which variable (carbon correction) to use from the 20m averages
    Fills in missing clusters
    """
    # soccom_df = loader.import_soccom_20m_averages(processed=False, start_date=start_date, end_date=end_date)
    soccom_df = soccom_df.rename(columns={'Lat':'latitude', 'Lon':'longitude',
                        'MLD':'mld', 'AbsSalt':'SA', 'ConservTemp':'CT'})
    
    # ERA5 sea level pressure uses longitude range -180 to 180
    soccom_df = mod_ocean.set_longitude_range(soccom_df, end_type = '180')
    # soccom_df['atmos_pres_Pa'] = soccom_df.apply(lambda row: get_row_sealevelpressure(row), axis=1)
    # soccom_df['atmos_pres_atm'] = soccom_df['atmos_pres_Pa'] / 1013.25 / 100

    print('Imported SOCCOM pCO2 data using variable ' + use_var)
    soccom_df['pco2_ocean'] = soccom_df[use_var]

    return soccom_df


# %% SOCAT P1 processing: 
# 1.3_P1_socat_features_with_colocation.ipynb

def convert_socat_fco2(socatDF):
    """ 
    Convert socat fCO2 to pCO2 using atmospheric pressure 
    # socatDS: xarray Dataset of SOCAT with 'expoID'
    socatDF: in 2.0 preprocessing use colocation DF from 0.5_
    """
    print('Calculating pCO2 from fCO2 using atmospheric pressure and pyCO2SYS...')
    # socatDF = socatDS.to_dataframe()
    socatDF['datetime'] = socatDF.datetime.astype('datetime64[ns]')

    # Calculate SA, CT for vapor pressure
    socatDF['SA'] = gsw.SA_from_SP(socatDF['sal'],  (np.tile(0, len(socatDF))), socatDF['longitude'], socatDF['latitude'])
    socatDF['CT'] = gsw.CT_from_t(socatDF['SA'], socatDF['sst'], (np.tile(0, len(socatDF))))

    socatDF['cruiseID'] = socatDF['socatid'].apply(lambda x: x.split('_')[0])

    if 'atmos_pres_atm' not in socatDF.columns:
        socatDF = add_sea_level_pressure(socatDF)

    # Convert fco2 to pco2
    socatDF['pco2_ocean'] = socatDF.apply(lambda row: pyco2.sys(
                                par1 = row['fco2rec'], 
                                par1_type = 5,
                                temperature = row.sst,  
                                pressure=0,
                                temperature_out = row.sst, # need to specify either temp or pres_out
                                pressure_atmosphere_out= row.atmos_pres_atm # from ncep
                                )['pCO2'], axis=1)
    socatDF = socatDF.rename(columns={'yearday': 'linear_time'})

    return socatDF


# %% Adding regression features to P1 datasets

def add_regression_time_vars(platDF):
    # Add new time variables
    platDF['linear_time'] = platDF.datetime.apply(lambda x: mod_ocean.datetime2ytd(np.datetime64(x), ref_time='2014-01-01'))
    platDF['ydcos']= mod_ocean.get_ydsines(platDF.linear_time.values)[0]
    platDF['ydsin']= mod_ocean.get_ydsines(platDF.linear_time.values)[1]
    platDF = mod_ocean.add_decimalyr(platDF)

    return platDF

def add_sea_level_pressure(platDF):
    """ 
    Nearest ERA5 0.25 deg grid point and hour. Fixed Oct 2026: previously looked up
    round((lon + 180)/2.5)*2.5, meant for a 0-360 grid, which read western hemisphere samples
    180 deg away and clamped eastern hemisphere samples to 179.75E on the -180-180 ERA5 grid.
    """
    print('Adding atmospheric pressure from ERA5 sea level pressure data...')
    platDF['datetime'] = pd.to_datetime(platDF['datetime'])
    platDF = mod_ocean.set_longitude_range(platDF, end_type = '180') # ERA5 grid is -180 to 180
    years = platDF['datetime'].dt.year.values
    slp = np.full(len(platDF), np.nan)
    for year in np.unique(years): # open each year's file once
        pos = np.flatnonzero(years == year)
        era_slp = xr.open_dataset(local_working_folder + 'era5/era5_mean_slp_yr' + str(year) + '.nc')
        slp[pos] = [get_row_sealevelpressure(row, era_slp) for _, row in platDF.iloc[pos].iterrows()]
    platDF['atmos_pres_Pa'] = slp
    platDF['atmos_pres_atm'] = platDF['atmos_pres_Pa'] / 1013.25 / 100
    print("Added ERA5 sea level pressure.")
    return platDF


def add_atmospheric_carbon_vars(platDF, convert_pco2=True, add_slp=False): # , make_delta = True):
    """ 
    Used to be add_regression_carbon_vars
    
    Add marine boundary layer xCO2 (ppm) to dataset

    """
    if add_slp:
        platDF = add_sea_level_pressure(platDF)
        
    if 'decimalyr' not in platDF.columns:
        platDF = mod_ocean.add_decimalyr(platDF)
    # platDF.to_csv(local_working_folder + 'temp_platDF' + datetime.now().strftime('%Y%m%d') + '.csv')

    # Get atmospheric fco2 from MBL marine boundary layer product (by latitude only)
    print('Adding atmospheric CO2 from MBL marine boundary layer product...')
    platDF['sin_lat'] = np.sin(platDF['latitude']*np.pi/180)
    platDF['atmos_co2_ppm'] = platDF.apply(get_row_atmos_co2_ppm, axis=1)
    # platDF = convert_co2_ppm_to_pco2(platDF, ppm_col='atmos_co2_ppm')
    # if make_delta:
    #     platDF['delta_pco2'] = platDF['pco2'] - platDF['pco2_atm']
    
    if convert_pco2:
        print('Converting atmospheric CO2 (ppm) to pCO2 (uatm) using vapor pressure...')
        platDF = mod_ocean.set_longitude_range(platDF, end_type = '360')
        platDF = convert_co2_ppm_to_pco2(platDF) # internal vapor_pres calculation
        print('Converted to pCO2.')

    return platDF

# %% ATMOSPHERIC AND OCEANIC CARBON VARS

def get_row_sealevelpressure(row, era_slp=None):
    """ 
    Using ERA5 (Jul 2026)
    Requires longitude range -180 to 180
    era_slp: that year's ERA5 dataset, opened from file if None"""
    # Add atmospheric pressure from NCEPII sea level pressure data. Match lon range 0-360
    # Needed for conversion from fco2 to pco2
    # Returns values in Pascals (divide by 101325 to get to atm)
    year = row.datetime.year
    
    # if product == 'ncep2':
    #     filepath = root_data_folder + 'ncep2-slp/'
    #     ncep_sea_level_pressure = xr.open_dataset(filepath + 'mslp.' + str(year) + '.nc')

    #     # Uses rounded longitude and latitude, faster
    #     # socatDF = mod_ocean.set_longitude_range(socatDF, end_type = '360') # called in convert_ method
    #     atmos_pres_Pa = ncep_sea_level_pressure.sel(lon=row.lon_round, lat=row.lat_round, time = row.datetime,
    #                         method='nearest')['mslp'].values.tolist()

    if era_slp is None:
        filepath = local_working_folder + 'era5/'
        era_slp = xr.open_dataset(filepath + 'era5_mean_slp_yr' + str(year) + '.nc')
    # print('Using ERA5 sea level pressure for year ' + str(year) + ' to get atmospheric pressure.')
    # print('Warning: Longitude range assumes -180 to 180')
    atmos_pres_Pa = era_slp.sel(longitude=row.longitude, latitude=row.latitude, valid_time = row.datetime,
                            method='nearest')['msl'].values.tolist()
    
    return atmos_pres_Pa

    
def get_row_atmos_co2_ppm(row):
    """ 
    Using MBL marine boundary layer product (Jul 2026)
    Assumes longitude """
    mbl_co2 = xr.open_dataset(root_data_folder + 'marineboundarylayer/co2_mbl_2014-2024_dataset_combined.nc')
    mbl_co2.lat_deg.values

    # originally in ppm
    result_ppm = mbl_co2.sel(decimalyr=row.decimalyr, method='nearest').interp(sin_lat=row.sin_lat)['mean'].values
    return result_ppm

    return atmos_pres_Pa

def convert_co2_ppm_to_pco2(platDF, ppm_col='atmos_co2_ppm'):
    """ convert ppm to pco2 using Zeebe and Wolf-Gladrow 2001 formulation
    Assumes data has longitude range 0-360
    """
    platDF['sst'] = gsw.t_from_CT(platDF['SA'], platDF['CT'], (np.tile(0, len(platDF))))
    platDF['sss'] = gsw.SP_from_SA(platDF['SA'], (np.tile(0, len(platDF))), platDF['longitude'], platDF['latitude'])

    #10/06
    platDF = mod_ocean.set_longitude_range(platDF, end_type = '360')
    platDF['vapor_pres_atm'] = platDF.apply(lambda row: get_row_atmos_vapor_pres(row), axis=1)
    platDF['pco2_atmos'] = platDF.apply(lambda row: row[ppm_col] * (row.atmos_pres_atm - row.vapor_pres_atm), 
                                        axis=1) # in uatm 

    return platDF

def get_row_atmos_vapor_pres(row):
    """ Used in marine boundary layer xCO2 (ppm) to pco2 (uatm) conversion 
    Requires data to have sst and sss 
    Zeebe and Wolf-Gladrow 2001
    """
    sst_kelvin = row.sst + 273.15
    ln_vapor_pres = (24.4542 
               - 67.4509 * (100 / sst_kelvin) 
               - 4.8489 * np.log(sst_kelvin / 100) 
               - 0.000544 * row.sss)
    vapor_pres = np.exp(ln_vapor_pres)
    return vapor_pres # in atmospheres



# %% SATELLITE ADT

def add_satellite_adt(platDF): # , year_range = range(2014,2024)):
    """ Uses lon range 180!! edited jul 13 """
    year_range = [x for x in np.sort(platDF.year.unique())]
    adt_dict = {k:None for k in year_range}
    for open_yr in adt_dict.keys():
        open_yr = int(open_yr)
        folder = root_data_folder + 'copernicusmarine/' # used to be in OneDrive/Code/CRUSOE
        filepath = ('cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_adt-sla_179.94W-179.94E_88.94S-35.06S_' + str(open_yr) + '-01-01-' + str(open_yr) + '-12-31.nc')
        adt_dict[open_yr] = xr.open_dataset(folder + filepath)

    platDF =  mod_ocean.expand_datetime(platDF, type='dataframe')

    # For satellite ADT set longitude range to 0-180
    platDF = mod_ocean.set_longitude_range(platDF, end_type = '180')
    platDF_annual = {k:df for k, df in platDF.groupby('year')}
    for yr in year_range:
        adt_data = adt_dict[yr]
        temp = platDF_annual[yr].copy()
        temp['adt'] =  temp.apply(lambda row: get_row_adt(row, adt_data), axis=1)
        platDF_annual[yr] = temp
    platDF_added = pd.concat(platDF_annual.values(), axis=0)

    print('Added columns "adt".')

    return platDF_added

def get_row_adt(row, adt_year_data):
    return adt_year_data.sel(time=row.datetime, latitude=row.latitude, longitude=row.longitude, 
                        method='nearest').adt.values.tolist()


# %% ERA5 wind speed
def get_row_windspeed(row, wind_year_data):
    # Use rounded
    row_loc = [np.round(x*4,0)/4 for x in [row.latitude, row.longitude]]
    nearest = wind_year_data.sel(latitude=row_loc[0], 
                                 longitude=row_loc[1], 
                                 valid_time = row.datetime,
                        method='nearest')
    return (nearest['u10'].values, nearest['v10'].values)

def get_row_avg_windspeed(row, wind_year_data, hours_before=72):
    row_loc = [np.round(x*4,0)/4 for x in [row.latitude, row.longitude]]
    period_search = [row.datetime - timedelta(hours=x) for x in range(1,hours_before)]
    nearest = wind_year_data.sel(latitude = row_loc[0], longitude = row_loc[1],
                    valid_time = period_search, method='nearest')
    
    return nearest.speed.values.mean()


def add_wind_speed(platDF, hours_before=None): 
    """ 
    :param hours_before: list of hrs before observation time to average over
    :return: platDF with added wind speed columns
    """
    year_range = [x for x in np.sort(platDF.year.unique())]
    wind_dict = {k:None for k in [i for i in year_range]}
    for yr in year_range:
        # yearly_era5 = xr.open_dataset('../climatedatastore/era5_winds_' + str(yr) + '.nc')
        yearly_era5 = xr.open_dataset(local_working_folder + 'era5/era5_wind_speed_' + str(yr) + '.nc')
        wind_dict[yr] = yearly_era5
    
    platDF =  mod_ocean.expand_datetime(platDF, type='dataframe') # shipDF
    platDF_annual = {k:df for k, df in platDF.groupby('year')}
    for yr in tqdm(year_range):
        temp = platDF_annual[yr].copy()

        if hours_before == None:
            temp['wind_components'] =  temp.apply(lambda row: get_row_windspeed(row, wind_dict[yr]), axis=1)
            temp['wind_speed'] = temp['wind_components'].apply(lambda x: np.sqrt(x[0]**2 + x[1]**2))
        else: # if hours_before is not None:
            for tperiod in hours_before:
                new_var_name = 'wind_speed_avg_' + str(tperiod) + 'h'
                temp[new_var_name] = temp.apply(lambda row: get_row_avg_windspeed(row, wind_dict[yr], 
                                                                                    hours_before=tperiod), axis=1)
        platDF_annual[yr] = temp

    platDF_added = pd.concat(platDF_annual.values(), axis=0)

    if hours_before == None:
        print('Added columns "wind_components", "wind_speed"')
    if hours_before is not None:
        print('Wind speed average given over past ' + str(hours_before) + ' hours')
        print('Added column ' + new_var_name)
    

    return platDF_added

# %% SOLAR RADIATION
def get_solar_radiation(row):
    vnorm = np.array([0, 0, -1])  # plane pointing zenith
    h = 0  # sea-level
    return irradiance_on_plane(vnorm, h, row['datetime'], row['latitude'])

def get_max_solar_radiation(row):
    vnorm = np.array([0, 0, -1])  # plane pointing zenith
    h = 0  # sea-level

    list = []
    for buffer in range(-12,13):
        time = row['datetime'] + timedelta(hours=buffer)
        list.append(irradiance_on_plane(vnorm, h, time, row['latitude']))

    return max(list)

def add_solar_radiation(platDF, daily_max=True):
    # platDF['datetime'] = pd.to_datetime(platDF['datetime'], format='%Y-%m-%d %H:%M:%S')
    platDF['datetime'] = platDF['datetime'].astype('datetime64[ns]')
    if daily_max:
        platDF['max_solar_rad'] = platDF.apply(get_max_solar_radiation, axis=1)
        print('Added column "max_solar_rad".')
    else:
        platDF['solar_rad'] = platDF.apply(get_solar_radiation, axis=1)
        print('Added column "solar_rad". ')

    return platDF

# %% CDR NSIDC SEA ICE

def get_row_seaice(row, seaice_year_data):
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3412", always_xy=True)
    x_meters, y_meters = transformer.transform(row.longitude, row.latitude)
    nearest_value = seaice_year_data.sel(x=x_meters, y=y_meters, time = row.datetime, method='nearest')['cdr_seaice_conc'].values
    return nearest_value


def get_row_avg_seaice(row, seaice_year_data, days_before=7):
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3412", always_xy=True)
    x_meters, y_meters = transformer.transform(row.longitude, row.latitude)

    period_search = [row.datetime - timedelta(days=x) for x in range(1,days_before)]
    nearest =  seaice_year_data.sel(x=x_meters, y=y_meters, time = period_search, method='nearest')
    
    return nearest['cdr_seaice_conc'].values.mean()


def add_sea_ice(platDF, days_before=None): 
    """ NSIDC CDR sea ice, 25km resolution, daily """
    year_range = [x for x in np.sort(platDF.year.unique())]
    ice_dict = {k:None for k in [i for i in year_range]}
    for yr in year_range:
        yearly_ice = xr.open_dataset(root_data_folder + 'nsidc-seaice/sic_pss25_' + str(yr) + '0101-' + str(yr) + '1231_v06r00.nc')
        ice_dict[yr] = yearly_ice
    
    platDF =  mod_ocean.expand_datetime(platDF, type='dataframe') # shipDF
    platDF['datetime'] = platDF['datetime'].astype('datetime64[ns]')
    platDF_annual = {k:df for k, df in platDF.groupby('year')}
    
    # transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3412", always_xy=True)

    for yr in tqdm(year_range):
        temp = platDF_annual[yr].copy()

        if days_before == None:
            temp['sea_ice'] =  temp.apply(lambda row: get_row_seaice(row, ice_dict[yr]), axis=1)

        if days_before is not None:
            new_var_name = 'sea_ice_avg_' + str(days_before) + 'd'
            temp[new_var_name] = temp.apply(lambda row: get_row_avg_seaice(row, ice_dict[yr], 
                                                                                  days_before=days_before), axis=1)
        platDF_annual[yr] = temp

    platDF_added = pd.concat(platDF_annual.values(), axis=0)

    if days_before == None: print('Added column "sea_ice".')
    else: print('Added column ' + new_var_name)
    return platDF_added


