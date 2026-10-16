import os.path
import numpy                 as np
import pandas                as pd
import xarray                as xr
from   datetime              import date, datetime, timedelta                 # for saving figures with today's date
import datetime
import scipy
import gsw
from tqdm import tqdm

# %% Time functions

def datetime2ytd(time, ref_time):
    """" Return time in YTD format from datetime format."""
    return (time - np.datetime64(ref_time))/np.timedelta64(1, 'D')

def ytd2datetime(num, ref_time):
    """" Return datetime format to YTD.
    @ param num: (int) number of days since ref_time
      ref_time: (str) reference time in 'YYYY-MM-DD' format
    """
    return (num * np.timedelta64(1,'D')) + np.datetime64(ref_time)

def add_decimalyr(df):
    """ for colocating atmospheric pco2 from MBL"""
    df = df.copy()
    df['datetime'] = df['datetime'].astype('datetime64[ns]') # pd.to_datetime(df['datetime'])
    df['decimalyr'] = (
        df['datetime'].dt.year +
        (df['datetime'].dt.dayofyear - 1) /
        (df['datetime'].dt.is_leap_year.replace({True: 366, False: 365}))
    )
    return df

def get_ydsines(yearday):
    """ For adding seasonal variable in Training_RandomForest.ipynb"""
    yearday = yearday%365.25
    ydcos = np.cos(2*np.pi*np.array(yearday)/365.25)
    ydsin = np.sin(2*np.pi*np.array(yearday)/365.25)

    return [ydcos, ydsin]

def expand_datetime(data, type='dataframe'):
    """ Choose "dataframe" or "dataset" type to expand datetime into year, month, day"""
    out = data.copy()
    if type == 'dataframe':
        out['year'] = data.datetime.astype('datetime64[ns]').map(lambda x: x.year)
        out['month'] = data.datetime.astype('datetime64[ns]').map(lambda x: x.month)
        out['day'] = data.datetime.astype('datetime64[ns]').map(lambda x: x.day)
    elif type == 'dataset':
        out['year'] = data.datetime.astype('datetime64[ns]').dt.year
        out['month'] = data.datetime.astype('datetime64[ns]').dt.month
        out['day'] = data.datetime.astype('datetime64[ns]').dt.day
        out = out.set_coords(['year', 'month', 'day'])
    return out


# %% Spatial functions (from 0.1_SOCAT)

def set_longitude_range(df, end_type = '360'):
    """ Switch from 0-360 to -180,180 range"""
    df = df.copy()
    if end_type == '180':
        # inds = df['longitude']<180
        # df.loc[inds, 'longitude'] = df.loc[inds, 'longitude'].apply(lambda x: x+360)
        
        inds = df['longitude']>180
        df.loc[inds, 'longitude'] = df.loc[inds, 'longitude'].apply(lambda x: x-360)
    elif end_type == '360':
        inds = df['longitude']<0
        df.loc[inds, 'longitude'] = df.loc[inds, 'longitude'].apply(lambda x: x+360)
    return df

def weighted_distance(lon_arr, lat_arr, Lx, Ly):
    """  used in Diagnostic_SOCAT
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

def did_cross_lon180(df):
    """ 
    Returns True if the platform crossed the 180 longitude line,
    i.e. when the longitude delta is large (> 270 degrees)"""
    df = df.copy()
    df.loc[:,'longitude_diff'] = df['longitude'].diff().values
    return (np.abs(df['longitude_diff']) > 270).any()

def get_avg_longitude(df, type='mean'):
    """ 
    Function to fix longitude averaging when crossing longitude 180 to -180
    @param df: dataframe with longitude values
    @return longitude_avg: (float) average longitude value 
    """
    # If any longitude differences are large,
    # Then add 360 to all the negative values of longitude
    # This will work even if the ship crosses the -180 bound several times
    if did_cross_lon180(df):
        index = df[df['longitude'] < 0].index
        df.loc[index, 'longitude'] = df.loc[index, 'longitude'] + 360

    if type == 'mean':
        longitude_avg = df['longitude'].mean()
    elif type == 'median':
        longitude_avg = df['longitude'].median()

    # If the average longitude is greater than 180, 
    # then subtract 360 (full revolution)
    if longitude_avg > 180:
        longitude_avg = longitude_avg - 360

    return longitude_avg 

# %% Bathymetry

# bathy_path = '/Volumes/cremas-repo/data/etopo/gebco_2024_sub_ice_topo/GEBCO_2024_sub_ice_topo.nc'
def get_bathymetry_DS(plat_index, bathy_nc, dropvars = [], bathy_lim = -1000,
                   save_path = None, save_tag = '', nc_attrs = {'title': ''}):
    """ 
    Find nearest bathymetry value for each float profile. 
    Return Dataset with open ocean profile index (averaged along pressure). 

    @param 
        plat_index: dataframe with 'latitude' and 'longitude' columns,
                index should be 'profid' (should be pre-averaged along pressure)
                [if not averaged, use `plat_DF.groupby('expoID').first()` during call]
        drop_vars: variables to drop when making final .nc file
                  for floats, dropvars = ['temperature_qc', 'salinity_qc', 'pressure_qc', 
                                         'temp_error', 'psal_error', 'pres_error']
        bathy_lim: (negative) limit for shallow profiles 
        bathy_nc : bathymetry Dataset
        save_path: path to save new dataframe, None by default means no save
    """
    print('Masking with bathymetry limit ' + str(bathy_lim))
    print('* Note long run time for large datasets *')

    # bathy_nc = xr.open_dataset(bathy_path).elevation    
    # plat_index = plat_df.groupby('profid').first()
    plat_index['bathymetry'] = bathy_nc.sel(
            lat=xr.DataArray(plat_index['latitude']), 
            lon=xr.DataArray(plat_index['longitude']), 
            method='nearest').values
    
    # Make xr Dataset to store profile-averaged core index by year
    INDEX_DS = (xr.Dataset.from_dataframe(plat_index[plat_index.bathymetry < bathy_lim])
                                                        .drop_vars(dropvars)
                                                        .assign_attrs(nc_attrs))
    if save_path != None:
        INDEX_DS.to_netcdf(save_path + save_tag + '.nc')
        print('Saved to ' + save_path + save_tag + '.nc' + '\n')

    return INDEX_DS



# %% Calculated variables

# def add_mixedlayer_pressure(platINDEX, platDS):
#     """ Originally from 2.0_process_pco2_data
# moved to mod_argo
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


def add_Pchip_buoyancy(plat_DF):
    """
    Calculate buoyancy (actually Nsquared using gsw) 
    @param      plat_DF: dataframe with profiles 
                ---> make new variable: profiles (list): list of dataframes, each dataframe is a profile
    @return     list of dataframes, each dataframe is a profile with a buoyancy column added    
    
    Version 09.06.2023
    """
    new_DF = pd.DataFrame()

    profids = pd.unique(plat_DF.profid) # list of profile ids
    profile_DFs = []
    for i in range(len(profids)):
        profile_DFs.append(plat_DF[plat_DF['profid'] == profids[i]].copy())

    for profile in profile_DFs:
        Nsquared, mid_pres = gsw.Nsquared(profile.SA.values, profile.CT.values, 
                                            profile.pressure.values, profile.lat.values)

        df = pd.DataFrame.from_dict({"Ns": Nsquared, "mp": mid_pres})
        df = df.dropna()

        if np.isnan(df.mp).all():
            nans = np.empty(len(profile['P'])); nans[:] = np.NaN
            profile.loc[:, 'buoyancy'] = nans
        else:
            f = scipy.interpolate.PchipInterpolator(x=df.mp, y=df.Ns, extrapolate = False)
            vertN2 = f(profile["pressure"].values)

            surf = np.where(~np.isnan(profile.SA.values))[0][0]
            bottom = np.where(~np.isnan(profile.SA.values))[0][-1]
            vertN2[surf] = vertN2[surf+1]
            vertN2[bottom] = vertN2[bottom-1]
            profile.loc[:, 'buoyancy'] = vertN2
        
        # Take vert N2 and find the maximum in the profile. 
        
        new_DF = pd.concat([new_DF, profile])

    return new_DF


# %% Isopycnal functions for profiler data (float or glider)

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


def get_isopycnal_signal(platDF, ave_isopycnal, var_thresh=0.01, var_list = ['yearday', 'pressure', 'sigma0', 'nitrate', 'spice']):
    """ 
    Originally from RandomForest_SG mod_DFproc.py

    @param: prof_list: list of glider DF's, using list_profile_DFs 
            ave_isopycnal: list of isopycnal values to find in each profile
            var_thresh: threshold for finding sigma in each profile
            var_list: which variables to keep track of
    @return: Dictionary object containing along-isopycnal variables. 
    """
    prof_list = list_profile_DFs(platDF)
    dLine = dict.fromkeys(ave_isopycnal)

    for sig in ave_isopycnal:
        temp = pd.DataFrame()

        for prof in prof_list:
            # Find all sigma points that are within that threshold
            rangeDF = pd.DataFrame() 
            rangeDF = prof[(prof['sigma0']< (sig+var_thresh)) & (prof['sigma0'] > (sig-var_thresh))].copy()

            # Choose mean of values
            rowdat = rangeDF[var_list].copy().dropna()
            rowdat = np.mean(rowdat, axis=0) #nanmean avoided if you drop nans above.
            temp = pd.concat([temp, rowdat], axis=1)
            
        temp = temp.T
        dLine[sig] = temp

    return dLine


# %% T-S diagrams and binning functions 
def setup_TS_contours(df, ax=None):
    """  
    For plotting; add T-S contours to a given axis.
    @param: df: dataframe with 'CT' and 'SA' as columns

    Example call for anyfloat (Dataframe):
    # crplot.setup_TS_contours(anyfloat, axs[0])
    # ax.scatter(anyfloat.SA, anyfloat.CT, c='r', s=2, alpha=0.5)
    """

    if ax == None:
        fig = plt.figure(figsize=(9,7))
        ax = fig.gca()

    # Add density contours
    # Figure out boudaries (mins and maxs)
    smin = df.SA.min() -.2
    smax = df.SA.max() +.2

    tmin= df.CT.min() - 1.2
    tmax = df.CT.max() + 0.5

    # Calculate how many gridcells we need in the x and y dimensions
    xdim = int(round((smax-smin)/0.1+1,0))
    ydim = int(round((tmax-tmin)+1,0))
    
    # Create empty grid of zeros
    dens = np.zeros((ydim,xdim))
    
    # Create temp and salt vectors of appropiate dimensions
    ti = np.linspace(1,ydim-1,ydim)+tmin
    si = np.linspace(1,xdim-1,xdim)*0.1+smin

    # Loop to fill in grid with densities
    for j in range(0,int(ydim)):
        for i in range(0, int(xdim)):
            dens[j,i]=gsw.sigma0(si[i],ti[j])

    CS = ax.contour(si,ti,dens, linestyles='dashed', colors='k', alpha=0.4, zorder=1)
    ax.clabel(CS, fmt='%1.2f')
    ax.set_xlabel('SA')
    ax.set_ylabel('CT')

    return ax


def TSbin(df, nbins):
    """ 
    Used in coords_TSbin function
    """
    nobs, bin_temp = np.histogram(df.CT, nbins)
    nobs, bin_sal = np.histogram(df.SA, nbins)   
    return [bin_temp, bin_sal]

def coords_TSbin(df, nbins): 
    """ Manually give bin coordinates to each obs. row in the dataframe
    Allows you to plot more complex quantities in T-S space. 
    """

    [bin_temp, bin_sal] = TSbin(df, nbins)
    # initialize empty rows which will hold new bin coordinates for T and S
    coordT = np.empty(len(df))
    coordS = np.empty(len(df))

    df = df.sort_values(by='CT')
    dfind = []  # find index limits where observations fall into each bin
    for i in range(len(bin_temp)): 
        dfind.append(np.searchsorted(df.CT, bin_temp[i], side='right'))
    for i in range(len(bin_temp)-1):  # notice -1 = nbins
        coordT[dfind[i]:dfind[i+1]] = i
    df['y_temp'] = coordT.astype(int)


    df = df.sort_values(by='SA')
    dfind = []
    for i in range(len(bin_sal)):
        dfind.append(np.searchsorted(df.SA, bin_sal[i], side='right'))
    for i in range(len(bin_sal)-1): 
        coordS[dfind[i]:dfind[i+1]] = i
    df['x_sal'] = coordS.astype(int)

    df = df.sort_values(by=['profid', 'pressure'])
    return df

def array_TSbin(df, nbins, var='oxygen', stat='count'):
    """" Calculates value for the TS-binned array.
    Note that coordT will be stored as "y_temp" and corresponds to row in the array."""
    arr = [ [np.NaN for i in range(nbins)] for j in range(nbins) ]
    for r in range(nbins):
        for c in range(nbins):
            subdf = df[(df['x_sal']==c) & (df['y_temp']==r)]

            if stat == 'mean':
                arr[r][c] = np.nanmean(subdf[var])
            elif stat == 'variance':
                arr[r][c] = np.var(subdf[var])
            elif stat == 'count':
                arr[r][c] = len(subdf[var])
            # change line here 

    return arr

# %% wind speed

def wind_speed(windstress,rho_air=1.2):
    '''
    Wind speed calculation from wind stress, using the formula:
    tau = rho_air * C_d * (w_speed)^2 where C_d = a/w_speed + b + c*w_speed

    Solves the roots of the equation: a*u^3 + b*u^2 + c*u - tau/rho_air

    where coefficients a, b, and c are based on Large & Yeager (2004)
    -----------------------------------------------------------------
    Input
     - windstress: single value of total wind stress 
     - rho_air: density of air, default = 1.2
    '''
    if np.sum(windstress)==0 or np.isnan(windstress):
        speed = np.array([np.nan])
    else:
        a0 = 0.0000764; a1 = 0.000142; a2 = 0.0027
        a3 = -windstress/rho_air
        r = np.roots([a0,a1,a2,a3])
        zro = abs(r.imag)<1e-7
        if np.sum(zro) > 0:
            speed = r.real[zro]
        else:
            speed = np.array([np.nan])
            
    return speed[0]





# # %% Interpolation function with gap handling
# # =============================================================================
# def custom_interp(x_data, y_data, x_levels, x_fill=25, x_gap = 'dynamic'):
#     """  
#     Updated Feb 24 2025
#     This function is adapted in mod_argo, regrid_pressure_levels(), for use on float data
#     (Adapting for Hannah, originally from in mod_argo interpolate_z_profile())
#     Function to interpolate single float profile data (pchip) to chosen pressure levels,
#     Does not fit over vertical gaps in the data, with cutoff defined by x_gap. 

#     @param      x_data: 1D array of x-values (pressure), length N
#                 y_data: 1D array of y-values (T, S, NO3, O2, etc), length N
#                 x_levels: 1D array of pressure levels to interpolate to, length N
#                 x_fill: (default 25) fill pressures up to surface with uppermost value in profile
#                         if that value has pressure < 25dbar
#     @return     output: pd DataFrame with interpolated data
#     """
#     # Initialize a dataframe for observed values
#     # May want to do some exception handling here (pressure sorting, drop duplicates)
#     prof = pd.DataFrame({'x': x_data, 'y': y_data}).dropna()
#     prof = prof.sort_values(by='x')
#     prof = prof.drop_duplicates(subset='x', keep='first')
#     prof['x_diff'] = [np.nan] + np.diff(prof.pressure).tolist()

#     # Initialize dataFrame to return interpolated values, with x_levels as index
#     output = pd.DataFrame(index=x_levels, columns=['y_interp'], dtype=float)
#     output.index.name='x_interp'

#     # ===== GAP FILLING
#     if x_gap == 'dynamic':
#         prof['max_allowed_gap'] = ([get_max_gap(x) for x in prof['x'].values])
#         subID_index = prof[prof['x_diff'] > prof['max_allowed_gap']].index
#     else: # Static, define own
#         subID_index = prof[prof['x_diff'] > x_gap].index

#     prof.loc[subID_index, 'marker'] = 1
#     prof['continuous_id'] = prof['marker'].cumsum().ffill().fillna(0).astype(int)

#     # If there are no gaps, all points will be assigned to same continuous_id
#     # Fit pchip to each continuous_id 
#     for _, subprof in prof.groupby('continuous_id'):
#         xmin = subprof['x'].min(); xmax = subprof['x'].max()
#         subx_levels = [x for x in x_levels if x > xmin and x < xmax] # valid pres levels
#         if len(subprof) > 1: # If there is more than 1 point to fit pchip over
#             f = scipy.interpolate.PchipInterpolator(x=subprof['x'], y=subprof['y'], extrapolate = False)
#             output.loc[subx_levels, 'y_interp'] = f(subx_levels)

#     # ==== HANDLE SURFACE GAPS 
#     # Note that by default, pchip without extrapolation will fill surface values < minimum
#     # Fill surface with the nearest neighbor if the first observed value is below x_fill = 25
#     first_obs = prof.iloc[0]
#     if first_obs['x'] < x_fill:
#         # Decide here if you want to fill with first interpolated value, or first observed value
#         # fill_value = output.dropna().iloc[0].y_interp
#         fill_value = prof.iloc[0].y
        
#         # Determine surface index values to fill
#         fill_index = [x for x in output.index if x < first_obs.x]
#         output.loc[fill_index, 'y_interp'] = np.tile(fill_value, len(fill_index))

        
#     return output.y_interp.values

