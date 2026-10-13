
# ======== internal 
import crusoe.oceantools as mod_ocean
import crusoe.argotools as mod_argo
import crusoe.preprocessing as mod_prep




# SOCCOM only 
# soccom_df = loader.import_soccom_20m_averages(
#                           start_date = '2013-12-31', 
#                           end_date = '2024-12-31',
#                           averaging_depth = 5)
# soccom_df = mod_prep.process_soccom_20m_averages(
#                       soccom_df, 
#                       start_date = '2013-12-31', 
#                       end_date = '2024-12-31', 
#                       use_var='pCO2_pHbias5_pK01')

# soccom_df = platDF.copy()
# print('Add "profid" variable name by comparing datetags...')
# # finderINDEX = xr.merge([bgcINDEX, bgcINDEX_test], join="outer") # combine ten-year and 2024 together
# soccomDF_processed, missing_ids = mod_prep.match_profid_from_datetag(soccom_df, coreINDEX)
# soccomDF_processed = mod_ocean.set_longitude_range(soccomDF_processed, end_type='360') # catch any longitude 
# soccomDF_processed['log_mld'] = np.log(soccomDF_processed['mld'])


# Adding ERA5 sea level pressure
platDF = mod_prep.add_sea_level_pressure(platDF) 

# Adding atmospheric CO2 from MBL marine boundary layer product
# Converting atmospheric CO2 (ppm) to pCO2 (uatm) using vapor pressure
platDF = mod_prep.add_regression_time_vars(platDF)
platDF = mod_prep.add_atmospheric_carbon_vars(platDF, convert_pco2=True)

# Add calculated features 
platDF = mod_prep.add_satellite_adt(platDF, year_range = range(2014,2025))
platDF = mod_prep.add_wind_speed(platDF, year_range = range(2014,2025))
platDF = mod_prep.add_sea_ice(platDF, year_range = range(2014,2025))
platDF = mod_prep.add_solar_radiation(platDF, daily_max=True)


platDF = mod_ocean.set_longitude_range(platDF, end_type='360')
platDF = mod_ocean.expand_datetime(platDF, type='dataframe')

platDF['delta_pco2'] = platDF['pco2_ocean'] - platDF['pco2_atmos']

platDF = platDF.reset_index()