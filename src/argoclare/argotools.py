# os tools

import os.path
from tqdm import tqdm

import numpy                 as np
import pandas                as pd
import xarray                as xr
from   datetime              import date, datetime, timedelta                 # for saving figures with today's date
import datetime
import gsw
import scipy
# import mod_cremas as crx
import matplotlib
import matplotlib.pyplot     as plt  

from argoclare import oceantools as myocn

# crplot (plotting helpers) is in the conda environment only; the processing functions don't need it
try:
    from crplot import figtools as mod_plot
    plt.rcParams.update(mod_plot.my_params(size=12))
except ImportError:
    mod_plot = None


def tester():
    print('testing editable twice install')
# %% Common slicing functions
# =============================================================================

def get_profile_DS(argoDS, profid=None):
    """ 
    Function to return an xr Dataset with data from one vertical profile.
    @ param: argoDS (dataset with unique profid's as index)
    @ return: xr Dataset
    """
    if profid == None:
        profid = argoDS.profid[np.random.randint(0, len(argoDS.profid))].values.astype(str).tolist()
    return argoDS.sel({'profid':profid}), profid

def get_wmo_DS(argoDS, wmo=None):
    """ 
    Function to return an xr Dataset with data from one float.
    @ param: argoDS (dataset with wmoid's as coordinate, profids as index)
    @ return: wmo_DS (xr Dataset), wmo (argo)
    """
    if wmo == None:
        all_wmoids = argoDS.to_dataframe().wmoid.unique()
        wmo = all_wmoids[np.random.randint(0, len(all_wmoids))]
    valid_profids = [x for x in argoDS.profid.values if x.startswith(str(wmo)[:7])]
    wmo_DS = argoDS.sel(profid=np.isin(argoDS.profid, valid_profids))
    return wmo_DS, str(wmo)[:7]

def get_wmo_DF(argoDF, wmo=None):
    """ 
    Function to return a pd Dataframe with data from one float.
    @ param: argoDF 
    @ return: wmoDF, wmo (argo)
    """
    argoDF = argoDF.reset_index().copy()
    if wmo == None:
        all_wmoids = argoDF.wmoid.unique()
        wmo = all_wmoids[np.random.randint(0, len(all_wmoids))]
    valid_profids = [x for x in argoDF.profid.values if x.startswith(str(wmo)[:7])]
    wmo_DF = argoDF[argoDF.profid.isin(valid_profids)]
    return wmo_DF, str(wmo)[:7]


def get_n_random_profiles(argoDF, n=10):
    """
    @param  argoDF:  DataFrame with multiindex (profid, pressure)
            n:        Number of random profiles to select
    @return DataFrame with n random profiles 
    """
    profids = argoDF.reset_index().profid.unique()
    return argoDF.loc[np.random.choice(profids, n)]


def list_profile_DFs(df):
    """ 
    @param df: dataframe with all profiles
    @return: list of dataframes, each with a unique profile
    """
    profids = pd.unique(df.profid)
    profile_DFs = []
    for i in range(len(profids)):
        profile_DFs.append(df[df['profid']==profids[i]].copy())
    return profile_DFs

def print_float_bounds(argo_DF):
    # print('Bounds of data: \n')
    print('Dates: \t\t' + str(argo_DF.datetime.min()) + ' to ' + str(argo_DF.datetime.max()))
    print('Latitude:\t' + str(argo_DF.latitude.min()) + ' to ' + str(argo_DF.latitude.max()))
    print('Longitude:\t' + str(argo_DF.longitude.min()) + ' to ' + str(argo_DF.longitude.max()))

# BGC_DF['profid'] = BGC_DF['profid'].apply(lambda x: f"{x[:7]}_id{x[7:]}")




# %% Functions to process Argo data (use with Argopy)
# =============================================================================

# Main method
def process_argo_float(float_df, bgc_list = ['pH']):
    """
    Return dataframe from a single core or BGC float dataset, accessed with Argopy. 
    Assumed to be used in 'expert' mode, i.e. not quality-controlled yet for BGC.
    (From Argopy, download and use .point2profile() and then .to_dataframe() to get input float dataframe)

    @param:
            float_df (dict):  single float Dataframe, accessed in 'research' mode for core
                                only 'expert' mode available for BGC floats
            bgc_list : list of BGC variables to include in the final dataframe
                        default ['pH'] adds only pH data
    @return:
            float_df (pd.DataFrame): 
    """
    float_df = float_df.reset_index()

    # Default columns to rename, starting with necessary properties across core/bgc
    # Note that Argopy "research mode" has removed "ADJUSTED" from column names
    new_columns = {'LATITUDE':'latitude','LONGITUDE':'longitude', 'TIME':'datetime', 
                'CYCLE_NUMBER':'cycle_number', 'PLATFORM_NUMBER':'wmoid', 
                'PRES_ADJUSTED':'pressure', 'TEMP_ADJUSTED':'temperature', 'PSAL_ADJUSTED':'salinity'}
    # Rename QC and error columns
    new_columns.update({'TIME_QC': 'time_qc', 'POSITION_QC': 'position_qc', 
                        'PRES_ADJUSTED_QC': 'pressure_qc', 
                        'TEMP_ADJUSTED_QC': 'temperature_qc','PSAL_ADJUSTED_QC': 'salinity_qc'})
    new_columns.update({'PRES_ADJUSTED_ERROR': 'pres_error', 
                        'PSAL_ADJUSTED_ERROR': 'psal_error', 'TEMP_ADJUSTED_ERROR': 'temp_error'})
    
    # output_vars = new_columns.values()

    # ==================
    # Add BGC variables to the new column names
    if 'pH' in bgc_list: # expert mode
        new_columns.update({'PH_IN_SITU_TOTAL_ADJUSTED': 'pH', 'PH_IN_SITU_TOTAL_ADJUSTED_QC': 'pH_qc',
                            'PH_IN_SITU_TOTAL_ADJUSTED_ERROR': 'pH_error'})
    if 'oxygen' in bgc_list: 
        new_columns.update({'DOXY_ADJUSTED': 'oxygen', 'DOXY_ADJUSTED_QC': 'oxygen_qc',
                            'DOXY_ADJUSTED_ERROR': 'oxygen_error'})
    # if 'nitrate' in bgc_list:
    #     new_columns.update({'NITRATE_ADJUSTED': 'nitrate', 'NITRATE_ADJUSTED_QC': 'nitrate_qc',
    #                         'NITRATE_ADJUSTED_ERROR': 'nitrate_error'})
    # ==================
        
    float_df.rename(columns=new_columns, inplace=True)
    # float_df['yearday'] = myocn.datetime2ytd(float_df['datetime'], ref_time = ref_time)

    # Create a unique profile id to be a useful index
    # Make sure strings are filled so 1st and 10th profile are different
    float_df['profid'] = float_df.apply(lambda x: str(x.wmoid) + 'cyc' + str(x.cycle_number).zfill(3), axis=1)

    # Add calculated variables using gsw
    float_df['SA']= gsw.SA_from_SP(float_df['salinity'],float_df['pressure'],float_df['longitude'],float_df['latitude'])
    float_df['CT'] = gsw.CT_from_t(float_df['SA'], float_df['temperature'], float_df['pressure']) 
    float_df['sigma0'] = gsw.sigma0(float_df.SA.values, float_df.CT.values)
    float_df['spice'] = gsw.spiciness0(float_df["SA"].values, float_df["CT"].values)

    # Turn all QC flags into strings
    qc_vars = [var for var in float_df.columns.tolist() if '_qc' in var]
    for k in qc_vars:
        float_df[k] = float_df[k].astype(str)


    # Standard variable list to return (core)
    # Can reorder by changing the output_vars list 
    output_vars = ['wmoid', 'profid', 'cycle_number', 'latitude', 'longitude', 'datetime',
            'pressure', 'CT', 'SA', 'sigma0', 'spice',
            'temperature', 'salinity',
            'temperature_qc', 'salinity_qc', 'pressure_qc',
            'time_qc', 'position_qc',
            'temp_error', 'psal_error', 'pres_error']
    
    for x in bgc_list:
        output_vars = output_vars + [x, x+'_qc', x+'_error']

    return float_df[output_vars]

def reindex_by_datetag(platData, type='dataset'):
    ndigits_time = 13 # 10 for YYYYMMDD, 13 for hour
    if type == 'dataset':
        platform_id = [str(pid) for pid in platData['wmoid'].values]
        datetag = [x[:ndigits_time].replace('-','').replace(' ','') for x in platData['datetime'].values]

        prof_datetag = [f"{p}_{d}" for p, d in zip(platform_id, datetag)]

        platData = platData.assign_coords(prof_datetag=("profid", prof_datetag))
        platData = platData.swap_dims({"profid": "prof_datetag"})
        platData = platData.drop_vars("profid")

    # if type == 'dataframe':
    else:
        temp = platData.reset_index()
        temp['prof_datetag'] = temp.apply(lambda row: str(row.wmoid) + '_' + 
                                          row.datetime[:ndigits_time].replace('-','').replace(' ',''), axis=1)
        # temp['prof_datetag'] = temp.wmoid.apply(lambda x: )


        temp = temp.drop(columns='index')
        platData = temp.set_index(['prof_datetag', 'pressure'])

    return platData

def datenum_to_datetime(matlab_datenum):
  """ For float matlab Datenum conversion"""
  days = matlab_datenum % 1
  dt = datetime.fromordinal(int(matlab_datenum)) + timedelta(days=days) - timedelta(days=366)
  return dt


# def process_core_float(float_df, var_list = 'phys', ref_time = '2014-01-01'):
#     """
#     (REPLACING WITH GENERAL FUNCTION, process_argo_float(). Outdated Jan 30 2025)
#     Return dataframe from a single float dataset accessed with Argopy. 
#     Assumed to be used in 'research' mode, i.e. already quality-controlled for core Argo. 

#     @param:
#             float_df (dict):  single float Dataframe, accessed in 'research' mode
#             var_list : list of variables to include in the final dataframe, or 'default'
#             ref_time : reference time for yearday calculation
#     @return:
#             float_df (pd.DataFrame): 
#     """
#     float_df = float_df.reset_index()
#     float_df.rename(columns={'LATITUDE':'latitude','LONGITUDE':'longitude', 'TIME':'datetime', 'CYCLE_NUMBER':'cycle_number',
#                             'PLATFORM_NUMBER':'wmoid', 'PRES':'pressure', 'TEMP':'temperature', 'PSAL':'salinity',
#                             'PRES_ERROR': 'pres_error', 'PSAL_ERROR': 'psal_error', 'TEMP_ERROR': 'temp_error',}, inplace=True)
#     float_df['yearday'] = myocn.datetime2ytd(float_df['datetime'], ref_time = ref_time)

#     # Create a unique profile id so easy to sort later
#     # Make sure strings are filled so 1st and 10th profile are different
#     prof = [tag.zfill(3) for tag in float_df['cycle_number'].astype(str)]
#     float_df['profid'] = [str(float_df.wmoid[0]) + '_id' + tag for tag in prof]

#     # Add calculated variables using gsw
#     float_df['SA']= gsw.SA_from_SP(float_df['salinity'],float_df['pressure'],float_df['longitude'],float_df['latitude'])
#     float_df['CT'] = gsw.CT_from_t(float_df['SA'], float_df['temperature'], float_df['pressure']) # Make sure to use SA here
#     float_df['sigma0'] = gsw.sigma0(float_df.SA.values, float_df.CT.values)
#     float_df['spice'] = gsw.spiciness0(float_df["SA"].values, float_df["CT"].values)

#     if var_list == 'phys':
#             var_list = ['wmoid', 'profid', 'latitude', 'longitude', 'datetime', 'yearday',
#             'pressure', 'CT', 'SA', 'sigma0', 'spice',
#             'temperature', 'salinity',
#             'temp_error', 'psal_error', 'pres_error', 'cycle_number']

#     return float_df[var_list]


def filter_qc_flags(float_df, qc_vars = 'all', use_flags=['1', '2', '5', '8']):
        """
        Filter a dataframe based on QC flags.
        Can choose different QC flags for different variables by calling the function multiple times.
        Note Argopy has this function, but this one allows you to track #obs, filter on position QC.
        @param: float_df (pd.DataFrame): dataframe of float data
                qc_vars (list): list of QC variables to filter
                        default 'all' filters on any variable with '_qc' in the name
                        ['temperature_qc', 'salinity_qc', 'pressure_qc', 'time_qc', 'position_qc', 'pH_qc']
                use_flags : flags that pass QC; default are standard argo QC flags 1, 2, and 8
                        '1' for 'good' data (only '1' returned in 'research' mode)
                        '2' for 'probably good' data
                        '5' for 'changed' data (rare; for position qc where lat/lon was adjusted)
                        '8' for 'interpolated/estimated' data
        @return: float_qc (pd.DataFrame)
        """ 
        print('Using flags: ', use_flags)
        float_qc = float_df.copy().reset_index()
        print ('# of profiles before QC filtering: \t', len(float_qc.profid.unique()))
        print('# of obs before QC filtering: \t\t', len(float_qc), '\n')

        if qc_vars == 'all':
                qc_vars = [var for var in float_qc.columns.tolist() if '_qc' in var]
        
        # for var in qc_vars:
        #         float_qc = float_qc[float_qc[var].isin(use_flags)]
        #         print('# of obs after ', var, ': \t\t', len(float_qc))
        

        qc_table = pd.DataFrame(columns= (use_flags + ['nobs_dropped', 'nobs_remaining']), index=qc_vars)
        for var in qc_vars:
                prevlen = len(float_qc) # store length before filtering
                for flag in use_flags:
                        qc_table.loc[var, flag] = len(float_qc[float_qc[var] == flag])

                # Filter based on use_flags
                float_qc = float_qc[float_qc[var].isin(use_flags)]
                qc_table.loc[var, 'nobs_dropped'] = int(prevlen - len(float_qc))
                qc_table.loc[var, 'nobs_remaining'] = len(float_qc)
        
        print(qc_table)

        print ('\n# of profiles after QC filtering: \t', str(len(float_qc.profid.unique())) + '\n')
        return float_qc
        

def to_xr_dataset(argoDF, nc_attrs={'date':str(datetime.datetime.now())}):
    """ 
    Convert float Dataframe to Dataset, and assign title and source attributes.
    """
    temp = xr.Dataset.from_dataframe(argoDF) # .set_index(["profid", 'pressure']))
    # argo_INDEX = temp.mean(dim='pressure')

    # argo_DS = temp.set_coords([ 'wmoid', 'datetime', 'yearday', 'latitude', 'longitude'])
    argo_DS = temp.set_coords([ 'profid', 'datetime', 'latitude', 'longitude'])
    argo_DS = argo_DS.assign_attrs(nc_attrs)

    return argo_DS
    

# %% Adding calculated variables

def add_mixedlayer_pressure(platINDEX, platDS):
    """ Originally from 2.0_process_pco2_data
    """
    # Calculate mixed layer pressure 
    plat_mlps, no_data = calc_mlp(platDS, threshold=0.03)
    plat_mlps['mld'] = gsw.z_from_p(plat_mlps.mlp.values, platINDEX.sel(profid=plat_mlps.index).latitude.values)

    # Select out only valid
    platINDEX = platINDEX.sel(profid=plat_mlps.index)
    platDS = platDS.sel(profid=plat_mlps.index)

    platINDEX['mld']= -xr.DataArray(plat_mlps.mld.values, dims='profid') # Notice negative
    platDS['mld']= -xr.DataArray(plat_mlps.mld.values, dims='profid') # Notice negative
    platINDEX['mlp'] = xr.DataArray(plat_mlps.mlp.values, dims='profid')
    platDS['mlp'] = xr.DataArray(plat_mlps.mlp.values, dims='profid')

    return platINDEX, platDS


def calc_mlp(platDS, threshold=0.03):  
    """  
    Calculating mixed layer pressure (MLP) for each Argo profile 
    Updated Aug 12 2025
    Use linear interpolation to find mixed layer pressure between two pressure levels

    @param:     platDS: xr Dataset with 'profid' as a dimension
                        (Current version uses 5dbar incremented data)
                threshold: Density threshold for mixed layer pressure calculation (default is 0.03 kg/m^3)
    @return:    prof_mlps: DataFrame with 'profid' as index and 'mlp' as column
                no_data: List of profids without valid data 
    """
    prof_mlps = pd.DataFrame(index=pd.unique(platDS.profid.values), columns=['mlp'], dtype=float) # float so gsw.z_from_p accepts it
    no_data = [] 
    
    for id, prof in platDS.to_dataframe().groupby('profid'):
        prof_df = prof.reset_index().copy()

        try: # general catch for missing data, technically don't need for our L3 data
            dens10 = prof_df.loc[prof_df.pressure==10].sigma0.values[0]
            dens_tofind = dens10 + threshold 
            mask = prof_df.sigma0.values > dens_tofind

            if mask.any(): # If any values meet threshold density condition
                idx = prof_df.sigma0.index[np.argmax(mask)]
                (p0,d0)= (prof_df.pressure[np.argmax(mask)-1], prof_df.sigma0[np.argmax(mask)-1]) # obs just above mlp
                (p1,d1)= (prof_df.pressure[np.argmax(mask)], prof_df.sigma0[np.argmax(mask)]) # first obs below mlp

                mlp = p1 + (dens_tofind - d1) * ((p1-p0)/(d1-d0)) # Linear interpolation
                prof_mlps.loc[id, 'mlp'] = mlp

            else: 
                prof_mlps.loc[id, 'mlp'] = np.nan
        except: 
            no_data.append(id)
        
    return prof_mlps, no_data

# %% Interpolation functions
# =============================================================================
def get_max_gap(x_val):
    """
    Return the maximum allowed gap based on the value of x.
    """
    if x_val < 100:
        return 25
    elif x_val < 500:
        return 50
    elif x_val < 1000:
        return 100
    else:
        return 250
    

def interpolate_z_profile(prof, pres_levels, var_list, 
                          z_gap = 200, surface_fill = 0,
                          ref_time = '2014-01-01') -> pd.DataFrame:
    """  
    Function to interpolate single float profile data (pchip) to chosen pressure levels,
    Does not fit over vertical gaps in the data, with cutoff defined by z_gap. 

    @param      prof: pd DataFrame with data from a single profile
                pres_levels: list of pressure levels to interpolate to
                var_list: list of variables to interpolate
                z_gap: z (m) threshold for large gaps in the data, can define own dbar
                    or use 'dynamic' to use get_max_gap() function
                surface_fill: (dbar) pressure above which to fill surface values with nearest neighbor,
                                 if first observed value is shallower than surface_fill
    @return     output: pd DataFrame with interpolated data
    """

    # If there is data, separate profile out into continuous vertical parts
    # Useful catch when iterating over a lot of profiles using regrid_pressure_levels() 
    try:
        # Sort by pressure and drop duplicates
        prof = prof.reset_index().sort_values(by='pressure')
        prof = prof.drop_duplicates(subset='pressure', keep="last") 
        prof['pres_diff'] = [np.nan] + np.diff(prof.pressure).tolist()

        # Dynamic gap filling
        if z_gap == 'dynamic':
            prof['max_allowed_gap'] = ([get_max_gap(x) for x in prof['pressure'].values])
            subID_index = prof[prof['pres_diff'] > prof['max_allowed_gap']].index
        else: # Static, define own
            subID_index = prof[prof['pres_diff'] > z_gap].index

        # print(subID_index)
        # Separate out continuous ID's
        prof.loc[subID_index, 'marker'] = 1
        prof['continuous_id'] = prof['marker'].cumsum().ffill().fillna(0).astype(int)
    except: 
        # print('no data in profid' + str(prof.profid.iloc[0]))
        return None
    
    # Initialize interpolated DataFrame to return, with pressure as index
    output = pd.DataFrame(index=pres_levels, columns=var_list, dtype=float)
    output.index.name = 'pressure'

    # If there are no gaps, all points will be assigned to same continuous_id
    # Fit pchip to each continuous_id 
    for _, subprof in prof.groupby('continuous_id'):
        if len(subprof) > 1: # If there is more than 1 point to fit pchip over
            xmin = subprof.pressure.min(); xmax = subprof.pressure.max()
            # Inclusive, so a level that falls exactly on the first/last observation is kept
            # (pchip passes through the observations, so it returns the observed value there)
            subpres_levels = [x for x in pres_levels if x >= xmin and x <= xmax] # valid pres levels
            for var in var_list:
                varprof = subprof.dropna(subset=[var]) # y must have finite values
                if len(varprof) > 1: # If there is more than 1 point to fit pchip over
                    f = scipy.interpolate.PchipInterpolator(x=varprof['pressure'], y=varprof[var], extrapolate = False)
                    output.loc[subpres_levels, var] = f(subpres_levels)

    if surface_fill > 0:
        # # ==== HANDLE SURFACE GAPS 
        # Fill surface with the nearest neighbor if the first observed value is below x_fill = 25
        first_obs = prof.iloc[0]
        if first_obs['pressure'] <= surface_fill: # <= to match the surface filter (first obs <= 25 dbar)
            for var in var_list:
                # Decide here if you want to fill with first interpolated value, or first observed value
                # fill_value = output.dropna().iloc[0].y_interp # first interp
                fill_value = prof.iloc[0][var] # first observed

                fill_index = [x for x in output.index if x < first_obs.pressure]
                output.loc[fill_index, var] = np.tile(fill_value, len(fill_index))

    # Reset index, drop nans (needed before datetime calculation)
    output = output.reset_index().dropna() 

    # Add profid, wmoid, datetime
    output['profid'] = np.tile(prof.profid.iloc[0], len(output))
    output['wmoid'] = np.tile(prof.wmoid.iloc[0], len(output))
    
    if 'yearday' in var_list:
        output['datetime'] = pd.to_datetime(myocn.ytd2datetime(output.yearday, ref_time = ref_time))
    
    return output
    

def regrid_pressure_levels(argoDF, pres_levels = np.arange(0,1001,5), var_list = 'all', bgc_list = [], 
                           z_gap = 200, surface_fill =0, ref_time = '2014-01-01',
                           grouping_col = 'profid'):
    ''' 
    Interpolate variables to a new pressure grid, without extrapolation.
    
    Replaces old function 'interpolate_float_pressure' from mod_argo in 0.4_bgcArgo 
    Adding middle catch for large gaps in the data

    @ param  argoDF: dataframe with "profid" as a variable column
                      can hold profiles from one float, or from multiple floats
             pres_levels: pressure levels to interpolate to
             bgc_list: (default) [] empty list for core Argo
                        ['pH'] or list of bgc variables to interpolate
                z_gap: z (m) threshold for large gaps in the data, can define own dbar
                        Set as very large value if you want to interpolate smoothly over gaps
                surface_fill: fill surface with nearest neighbor if first observed value is below x_fill
                            see interpolate_z_profile() for details
                ref_time: reference time for yearday calculation
    @ return argoDF_regular: dataframe with interpolated variables
                            can be converted to xr Dataset with .to_xr_dataset()
    '''
    # Default variables to interpolate (CT, SA) from core Argo 
    if var_list == 'all':
        var_list = ['latitude', 'longitude', 'linear_time',
                    'CT', 'SA', 'sigma0', 'spice', 'temperature', 'salinity']
    var_list = var_list + bgc_list

    # Initialize list to hold interpolated DFs, one for each profile
    profile_interp_list = []
    for key, profDF in argoDF.groupby(grouping_col):
        if len(profDF)>=2:
            profile_interp_list.append(interpolate_z_profile(profDF, pres_levels, var_list, 
                                                z_gap = z_gap,  # avoid interpolating over large gaps 
                                                surface_fill = surface_fill,
                                                ref_time = ref_time))
        # except: continue # if there is no data in the profile, skip it 

        # Combine all profiles into one DataFrame
    argoDF_regular = pd.concat(profile_interp_list) 
    argoDF_regular = argoDF_regular.set_index(["profid", 'pressure'])
    # argoDF_regular = argoDF_regular.dropna(subset=['CT', 'SA'])

    return argoDF_regular


# def interpolate_float_pressure(argoDF, pres_levels = np.arange(0,1001,5), bgc_list = [], ref_time = '2014-01-01'):
#     ''' 
#     OUTDATED as of Feb 22 2025, see regrid_pressure_levels()
#     Interpolate variables to a new pressure grid, without extrapolation.

#     @ param  argoDF: dataframe with "profid" as a variable column
#                       can hold profiles from one float, or from multiple floats
#              pres_levels: (regular) pressure levels to interpolate to
#              bgc_list: (default) [] empty list for core Argo
#                         ['pH'] or list of bgc variables to interpolate
#     @ return argoDF_regular: dataframe with interpolated variables
#                             can be converted to xr Dataset with .to_xr_dataset()
#     '''
#     # Default variables to interpolate (CT, SA) from core Argo 
#     var_list = ['CT', 'SA', 'sigma0', 'spice', 'temperature', 
#                     'salinity', 'yearday', 'latitude', 'longitude']
#     var_list = var_list + bgc_list

#     # # Initialize a list to hold interpolated DFs, one for each profile
#     profile_interp_list = []

#     # For each profile, interpolate variables and store in a dictionary
#     interpolated_variables = {k:None for k in var_list} # For each 
#     for profid, group in argoDF.groupby('profid'):
        
#         # print('Processing profile:', profid)
#         group = group.dropna(subset=['CT']).sort_values(by='pressure').reset_index()
#         group = group.drop_duplicates(subset='pressure', keep="last") # why are there pressure duplicates? 
#         for var in var_list:
#             try: # if there is valid data for the variable

#                 # # catch any large gaps in the data
#                 # group['pres_diff'] = np.diff(group.pressure)
#                 # # group['marker'] = np.tile(np.nan, len(group))

#                 # if group['pres_diff'].any()>200:
#                 #     subID_index = group[group['pres_diff']>200].index
#                 #     group.loc[subID_index, 'marker'] = 1
#                 #     group['continuous_id'] = group['marker'].cumsum().ffill()

#                 # temp = [] # list of interpolated variables
#                 # for subid, subgroup in group.groupby('continuous_id'):
#                 #     subpres_levels = pres_levels[(pres_levels >= subgroup['pressure'].min()) & (pres_levels <= subgroup['pressure'].max())]
#                 #     f = scipy.interpolate.PchipInterpolator(x=group['pressure'], y=group[var], extrapolate = False)
#                 #     temp = f(subpres_levels)

#                 # else: # large gaps
#                 f = scipy.interpolate.PchipInterpolator(x=group['pressure'], y=group[var], extrapolate = False)
#                 interpolated_variables[var] = f(pres_levels)

#             except: # if no valid data for the variable
#                 interpolated_variables[var] = np.tile(np.nan, len(pres_levels))
        
#         # Combine profile information into one DF and add to list
#         # prof_interp = pd.DataFrame({var: interpolated_variables[var] for var in var_list})
#         prof_interp = pd.DataFrame(interpolated_variables)
#         prof_interp['profid'] = profid
#         prof_interp['wmoid'] = group['wmoid'] #.values[0]
#         prof_interp['pressure'] = pres_levels

#         profile_interp_list.append(prof_interp)

#     argoDF_regular = pd.concat(profile_interp_list).dropna(subset=['CT', 'SA'])
#     argoDF_regular = argoDF_regular.set_index(["profid", 'pressure'])
#     argoDF_regular['datetime'] = pd.to_datetime(myocn.ytd2datetime(argoDF_regular['yearday'], ref_time = ref_time))

#     # Make sure there are values for CT, SA
#     argoDF_regular = argoDF_regular.dropna(subset=['CT', 'SA'])

#     return argoDF_regular



# %% colocation functions

def weighted_distance(lon_arr, lat_arr, Lx, Ly):
    """ 
    Updated, fixed longitude difference. 
    @param:
        Calculate a distance metric that incorporates ratio of decorrelation scales  
                (zonal Lx, meridional Ly)
        lon_arr: [lon1, lon2]
        lat_arr: [lat1, lat2]
    """
    delta_lon = lon_arr[1] - lon_arr[0]  # maximum difference of 180, either direction
    if np.abs(delta_lon) > 180:
        delta_lon = 360 - np.abs(delta_lon)
    delta_lat = lat_arr[1] - lat_arr[0]

    return np.sqrt( (delta_lon**2 /Lx**2) + (delta_lat**2 / Ly**2) )


# # METHOD updated Apr 23 2025
# Originally from 0.5_diagnostic_socatv2024
# Both time and space window, faster with using INDEX rather than pressure data
def find_nearest_float(platDF, argoINDEX, yd_window = 7, ref_time = '2014-01-01', Lx=2.5) -> pd.DataFrame:
    """  
    Find nearest argo profile to each SOCAT observation, calculate distance and yearday separation
    @param    platDF:       DataFrame of SOCAT observations with latitude, longitude
              argoINDEX:    xr Dataset with "profid" coordinates (pre-averaged along pressure)
              yd_window:    buffer number of days to search for nearest profile
              ref_time:     reference time for yearday conversion
              Lx:           longitude/latitude decorrelation scale for x/y (ex. 3 km ~ longitude, 1km ~ latitude)
                            Set Lx=1 for equal weighting
    """

    # argoINDEX = myocn.expand_datetime(argoINDEX, type='dataset') # adds month as coordinate
    # platDF = myocn.expand_datetime(platDF, type='dataframe').reset_index()

    colocation = pd.DataFrame(index=platDF.index, columns=
                            ['nearest_profid', 'prof_datetime', 'prof_lat', 'prof_lon',
                                'yd_sep', 'km_sep'])
    for idx in tqdm(platDF.index):
        lower = platDF.loc[idx, 'linear_time'] - yd_window
        upper = platDF.loc[idx, 'linear_time'] + yd_window
        argoset = (argoINDEX.where(argoINDEX.linear_time > lower)
                    .where(argoINDEX.linear_time < upper)
                    )

        argoDF = (argoset.reset_coords().to_dataframe() # .mean(dim='pressure')
                    .dropna(subset = ['CT', 'SA'])
                    )

        if len(argoDF) > 0:
            # Recalculate distance and yearday separation for each socat obs
            argoDF['yd_sep'] = argoDF.linear_time.map(lambda x: abs(x - platDF.loc[idx, 'linear_time']))
            argoDF['km_sep'] = argoDF.apply(lambda x: gsw.distance([platDF.loc[idx,'longitude'], x.longitude], 
                                                                [platDF.loc[idx,'latitude'], x.latitude])/1000, axis=1) # km

            argoDF['weighted_dist'] = argoDF.apply(lambda x: weighted_distance(
                                            [platDF.loc[idx,'longitude'], x.longitude], 
                                            [platDF.loc[idx,'latitude'], x.latitude], 
                                            Lx=Lx, Ly=1), axis=1)
            # imin = argoDF.idxmin(skipna=True).weighted_dist   # old version jan 2026
            imin = argoDF.sort_values(by='weighted_dist').index.values[0]   # new version mar 2026
            
            # Store metrics of nearest profile ID
            colocation.loc[idx, 'nearest_profid'] = imin
            colocation.loc[idx, 'prof_datetime'] = myocn.ytd2datetime(argoDF.loc[imin].linear_time, ref_time=ref_time)
            colocation.loc[idx, 'prof_lat'] = argoDF.loc[imin].latitude
            colocation.loc[idx, 'prof_lon'] = argoDF.loc[imin].longitude
            colocation.loc[idx, 'yd_sep'] = argoDF.loc[imin].yd_sep
            colocation.loc[idx, 'km_sep'] = argoDF.loc[imin].km_sep

    return pd.concat([platDF, colocation], axis=1)



# Co-location of other data to nearest argo float 
# Try slicing argoDS to same season as the SOCAT observation and searching for min. distance
def colocate_to_floats(platDF, argoDS, ref_time = '2014-01-01', minimize_var='km_sep') -> pd.DataFrame:
    """  
    OUTDATED
    For each SOCAT obs, find nearest (spatial) Argo profile that falls in same month +/- 1,
    ignoring year (assumes that seasonal cycle more important). 
    Calculate spatial separation ('km_sep') and day of year separation ('yd_sep', between 0,365)
    Spatial distance calculated using gsw.distance function. 

    @param  platDF: DataFrame of SOCAT observations with latitude, longitude
            argoDS: xr Dataset with "pressure", "profid" coordinates
            ref_time: reference time for yearday conversions
    @return DataFrame with SOCAT obs, and nearest Argo profile info
            columns:    prof_datetime, lat, lon are from the nearest Argo matchup

    """

    argoDS = myocn.expand_datetime(argoDS, type='dataset') # adds month as coordinate
    platDF = myocn.expand_datetime(platDF, type='dataframe')

    colocation = pd.DataFrame(index=platDF.index, columns=
                                ['nearest_profid', 'prof_datetime', 'prof_lat', 'prof_lon',
                                  'km_sep', 'yd_sep'])

    molist = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1]

    for idx in tqdm(platDF.index):
        # For each socat obs, subset Argo data to same month +/- 1 month
        row = platDF.loc[idx]
        valid_months = molist[row['month']-1:row['month']+2]
        argoset = argoDS.where(argoDS.month.isin(valid_months), drop=True)
        argoDF = argoset.reset_coords().mean(dim='pressure').to_dataframe()
        
        if len(argoDF) > 0:
            # Recalculate distance and yearday separation for each socat obs
            argoDF['yd_sep'] = argoDF.yearday.map(lambda x: abs(x - row['yearday'])%365)  # note mod 365 
            argoDF['km_sep'] = argoDF.apply(lambda x: gsw.distance([row['longitude'], x.longitude], 
                                                                    [row['latitude'], x.latitude]), axis=1) # m
            # Find minimum dist separation
            imin = argoDF.idxmin()[minimize_var]
            colocation.loc[idx, 'nearest_profid'] = imin
            colocation.loc[idx, 'prof_datetime'] = myocn.ytd2datetime(argoDF.loc[imin].yearday, ref_time=ref_time)
            colocation.loc[idx, 'prof_lat'] = argoDF.loc[imin].latitude
            colocation.loc[idx, 'prof_lon'] = argoDF.loc[imin].longitude
            colocation.loc[idx, 'yd_sep'] = argoDF.loc[imin].yd_sep
            colocation.loc[idx, 'km_sep'] = argoDF.loc[imin].km_sep/1000

    return pd.concat([platDF, colocation], axis=1)



# %% Plotting
# =============================================================================

from   cmocean               import cm as cmo
import cartopy
import cartopy.crs           as     ccrs   

# Prepare for plotting histogram of profile locations
# Longitude and latitude range
lon_min = -180
lon_max =  180
lat_min = -85
lat_max = -30

# depth range
zmin = 20.0
zmax = 1000.0

# ranges
lon_range   = (lon_min, lon_max)
lat_range   = (lat_min, lat_max)
depth_range = (zmin, zmax)



def plot_profiles_from_WMO(floatDF, var = 'CT', ax=None, dotsize=2, ylims=[1000,0]):
    """  
    See a selection of profiles as fxn of depth. 
    @param: floatDF: dataframe, either with 'profid' column (multiple profiles) 
                    or representing a single profile
    """
    if ax == None:
        fig = plt.figure(figsize=(5, 8))
        ax = fig.gca()

    try: floatDF = floatDF.reset_index()
    except: pass
    if 'profid' in floatDF.columns: # If floatDF has multiple profiles
        for name, group in floatDF.groupby('profid'):
            ax.scatter(group[var], group['pressure'], label=name, s=dotsize, marker='o', alpha=0.5, zorder=3)
    else: # If you indexed by profid (e.g. floatDF.loc['5906030_id010'])
        ax.scatter(floatDF[var], floatDF['pressure'], s=dotsize, marker='o', alpha=0.5, zorder=3)

    ax.invert_yaxis()
    ax.set_ylim(ylims)
    ax.set_ylabel('pressure')
    ax.set_xlabel(var)
    ax.grid(alpha=0.2, zorder=1)

    return ax


def plot_TS_diagnostics(wmo_DF, figsize=(12,8), dot_small=2, dot_medium = 8, dot_large = 20, add_mlp=False):
    """ 
    Plot diagnostic figures for a single WMO dataframe.
    @param: wmo_DF: dataframe with 'profid' column
            dot_small: size of dots for depth profiles, TS plot
            dot_medium: size of dots for float locations
            dot_large: size of dots for CT, SA sections
    @return: fig, axs
            ax1,2 : Sections of CT, SA
            ax3  : Map of float locations
            ax4,5 : Profiles of CT, SA
            ax6   : T-S diagram
    """
    wmo_DF = wmo_DF.reset_index().dropna(axis=0, subset=['CT', 'pressure'])
    # my_wmo = wmo_DF['wmoid'][0]
    my_wmo = wmo_DF.wmoid.dropna().unique()[0].astype(int)
    print_float_bounds(wmo_DF)
    # ==== 

    fig  = plt.figure(figsize=figsize, layout='tight')
    # axs = axs.flatten()
    gs = fig.add_gridspec(3,3)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[1, 0:2])
    ax3 = fig.add_subplot(gs[2, 0:2], projection = ccrs.PlateCarree())

    ax4 = fig.add_subplot(gs[0, 2])
    ax5 = fig.add_subplot(gs[1, 2])
    ax6 = fig.add_subplot(gs[2, 2])

    # ax1 = plt.subplot(311)
    # ax2 = plt.subplot(312)
    # ax3 = plt.subplot(313, projection = ccrs.PlateCarree())

    for ax in [ax1]:
        sca_CT = ax.scatter(wmo_DF.datetime, wmo_DF.pressure, s=dot_large, c=wmo_DF.CT, cmap=cmo.thermal)
        plt.colorbar(sca_CT)
        ax.invert_yaxis()
        ax.set_ylabel('pressure')
        ax.set_title('WMO ' + str(my_wmo)[:7] + ', CT') #' + '$\Theta$'

    for ax in [ax2]:
        sca_SA = ax.scatter(wmo_DF.datetime, wmo_DF.pressure, s=dot_large, c=wmo_DF.SA, cmap=cmo.haline)
        plt.colorbar(sca_SA)
        ax.invert_yaxis()
        ax.set_ylabel('pressure')
        ax.set_title('WMO ' + str(my_wmo)[:7] + ', SA')

    if add_mlp:
        for ax in [ax1, ax2]:
            # ax.scatter(wmo_DF['datetime'], wmo_DF['mlp'], c='w', marker='X', s=dot_large, alpha=1)
            ax.scatter(wmo_DF['datetime'], wmo_DF['mlp'], c='r', marker='_', s=dot_medium, alpha=0.7)
    for ax in [ax3]:
        ax.scatter(wmo_DF.longitude, wmo_DF.latitude, c='r', s=dot_medium, transform=ccrs.PlateCarree())
        ax.coastlines(resolution='50m',color='gray')
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=1, color='gray', alpha=0.5, linestyle='--')
        ax.add_feature(cartopy.feature.LAND)

        ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], ccrs.PlateCarree())

        # plt.colorbar(sca_map)


    pres_max = wmo_DF.pressure.max()
    for ax in [ax4]:
        plot_profiles_from_WMO(wmo_DF, var='CT', ax=ax, dotsize=dot_small)
        ax.set_ylim([pres_max+pres_max*0.05, 0])
        ax.set_title('Temperature Profiles')

    for ax in [ax5]:
        plot_profiles_from_WMO(wmo_DF, var='SA', ax=ax, dotsize=dot_small)
        ax.set_ylim([pres_max+pres_max*0.05, 0])
        ax.set_title('Salinity Profiles')

    for ax in [ax6]:
        if mod_plot is None:
            raise ImportError('plot_TS_diagnostics needs crplot (in the conda environment, not pixi)')
        mod_plot.setup_TS_contours(wmo_DF, ax6, contour_font_size=10)
        ax.scatter(wmo_DF.SA, wmo_DF.CT, c='r', s=dot_small, alpha=0.5)
    
    return fig, [ax1, ax2, ax3, ax4, ax5, ax6]

