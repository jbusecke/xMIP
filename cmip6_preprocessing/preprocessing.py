# Preprocessing for CMIP6 models
import xarray as xr
import pandas as pd
import numpy as np
import warnings

def cmip6_renaming_dict():
    # I could probably simplify this with a generalized single dict, 
    # which has every single possible `wrong` name and then for each model
    # the renaming function just goes through them...
    
    """central database for renaming the dataset."""
    dim_name_dict = {
        "AWI-CM-1-1-MR":{},
        "BCC-CSM2-MR": {
            "x": "lon",
            "y": "lat",
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "bnds":"bnds",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            "vertex": None,
            'time_bounds': "time_bnds",
        },
        "BCC-ESM1": {
            "x": "lon",
            "y": "lat",
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "bnds":"bnds",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            "vertex": "vertex",
            'time_bounds': "time_bnds",
        },
        "CAMS-CSM1-0": {
            "x": ["i", 'lon'],
            "y": ["j", 'lat'],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": None,
            "lat_bounds": None,
            "vertex": 'vertices',
            'time_bounds': "time_bnds",
        },
        "CanESM5": {
            "x": ["i", 'lon'],
            "y": ["j", 'lat'],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": None,
            "lat_bounds": None,
            "time_bounds": "time_bnds",
            "vertex": "vertices",
        },
        "CNRM-CM6-1": {
            "x": ["x", 'lon'],
            "y": ["y", 'lat'],
            "lon": "lon",
            "lat": "lat",
            "lev": "lev",
            "bnds": "axis_nbounds",
            "lev_bounds": "lev_bounds",
            "lon_bounds": "bounds_lon",
            "lat_bounds": "bounds_lat",
            'vertex': "nvertex",
            'time_bounds': "time_bnds",
        },
        "CNRM-ESM2-1": {
            "x": ["x", "lon"],
            "y": ["y", "lat"],
            "lon": "lon",
            "lat": "lat",
            "lev": "lev",
            "lev_bounds": "lev_bounds",
            "lon_bounds": "bounds_lon",
            "lat_bounds": "bounds_lat",
            "bnds":"axis_nbounds",
            'vertex': None,
            'time_bounds': "time_bnds",
        },
        "E3SM-1-0": {
            "x": "lon",
            "y": "lat",
            "lon": None,
            "lat": None,
            "lev": "lev",
            "bnds":"bnds",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            "time_bounds":"time_bounds",
            'vertex': None,
        },
        "EC-Earth3-LR": {
            "x": ["i", 'lon'],
            "y": ["j", 'lat'],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": None,
            "lat_bounds": None,
            'time_bounds': "time_bnds",
            #         'vertex': 'vertices',
            #         'dzt': 'thkcello',
        },
        "EC-Earth3-Veg": {
            "x": ["i", 'lon'],
            "y": ["j", 'lat'],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": None,
            "lat_bounds": None,
            'vertex': 'vertices',
            'time_bounds': "time_bnds",
            #         'dzt': 'thkcello',
        },
        "EC-Earth3": {
            "x": ["i", 'lon'],
            "y": ["j", 'lat'],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": None,
            "lat_bounds": None,
            'time_bounds': "time_bnds",
            #         'vertex': 'vertices',
            #         'dzt': 'thkcello',
        },
        "FGOALS-f3-L": {
            "x": ["i", 'lon'],
            "y": ["j", 'lat'],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": None,
            "lat_bounds": None,
            'time_bounds': "time_bnds",
            #         'vertex': 'vertices',
            #         'dzt': 'thkcello',
        },
        "FGOALS-g3": {
            "x": ["i", 'lon'],
            "y": ["j", 'lat'],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": None,
            "lat_bounds": None,
            'time_bounds': "time_bnds",
            #         'vertex': 'vertices',
            #         'dzt': 'thkcello',
        },
        "MIROC-ES2L": {
            "x": ["x", 'lon'],
            "y": ["y", 'lat'],
            "lon": "longitude",
            "lat": "latitude",
            "lev": ["lev", "zlev"],
            "lev_bounds": ["lev_bnds", "zlev_bnds"],
            "lon_bounds": "x_bnds",
            "lat_bounds": "y_bnds",
            "time_bounds": "time_bnds",
            'vertex': 'vertices',
        },
        "MIROC6": {
            "x": ["x", 'lon'],
            "y": ["y", 'lat'],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "x_bnds",
            "lat_bounds": "y_bnds",
            'time_bounds': "time_bnds",
        },
        "HadGEM3-GC31-LL": {
            "x": ["i", 'lon'],
            "y": ["j", 'lat'],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bounds",
            "lon_bounds": None,
            "lat_bounds": None,
            'time_bounds': "time_bnds",
        },
        "HadGEM3-GC31-MM": {
            "x": ["i", 'lon'],
            "y": ["j", 'lat'],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bounds",
            "lon_bounds": None,
            "lat_bounds": None,
            'time_bounds': "time_bnds",
        },
        "UKESM1-0-LL": {
            "x": ["i", 'lon'],
            "y": ["j", 'lat'],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": None,
            "lat_bounds": None,
            "time_bounds":"time_bnds",
            #         'vertex': 'vertices',
            #         'dzt': 'thkcello',
        },
        "GISS-E2-1-G-CC": {
            "x": "lon",
            "y": "lat",
            "lon": None,
            "lat": None,
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            'time_bounds': "time_bnds",
            #         'vertex': None,
            #         'dzt': 'thkcello',
        },
        "GISS-E2-1-G": {
            "x": "lon",
            "y": "lat",
            "lon": None,
            "lat": None,
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            'time_bounds': "time_bnds",
            #         'vertex': None,
            #         'dzt': 'thkcello',
        },
        "GISS-E2-1-H": {
            "x": "lon",
            "y": "lat",
            "lon": None,
            "lat": None,
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            'time_bounds': "time_bnds",
            #         'vertex': None,
            #         'dzt': 'thkcello',
        },
        "CESM2-WACCM": {
            "x": ["nlon", "lon"],
            "y": ["nlat", "lat"],
            "lon": "lon",
            "lat": "lat",
            "lev": "lev",
            "bnds":"d2",
            "time_bounds":"time_bnds",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            'vertex': 'vertices',
        },
        "CESM2": {
            "x": ["nlon", "lon"],
            "y": ["nlat", "lat"],
            "lon": "lon",
            "lat": "lat",
            "lev": "lev",
            "bnds":'d2',
            "time_bounds":"time_bnds",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            'vertex': 'vertices',
        },
        "GFDL-CM4": {
            "x": ["x","lon"],
            "y": ["y", "lat"],
            "lon": "lon",
            "lat": "lat",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            "time_bounds": "time_bnds",
            #         'vertex': 'vertex',
            #         'dzt': 'thkcello',
        },
        "GFDL-ESM4": {
            "x": ["x","lon"],
            "y": ["y", "lat"],
            "lon": "lon",
            "lat": "lat",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            "time_bounds": "time_bnds",
            #         'vertex': 'vertex',
            #         'dzt': 'thkcello',
        },
        "NESM3": {
            "x": ['i', "lon"],
            "y": ['j', "lat"],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": None,
            "lat_bounds": None,
            'vertex': 'vertices',
            'time_bounds': "time_bnds",
            #         'dzt': 'thkcello',
        },
        "MRI-ESM2-0": {
            "x": ['x', "lon"],
            "y": ['y', "lat"],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "bnds":'bnds',
            "lev_bounds": "lev_bnds",
            "lon_bounds": ["x_bnds", 'lon_bnds'],
            "lat_bounds": ["y_bnds", 'lat_bnds'],
            "time_bounds": "time_bnds",
            'vertex': 'vertices',
        },
        "SAM0-UNICON": {
            "x": ["i", 'lon'],
            "y": ["j", 'lat'],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": None,
            "lat_bounds": None,
            'vertex': 'vertices',
            'time_bounds': "time_bnds",
            #         'dzt': 'thkcello',
        },
        "MCM-UA-1-0": {
            "x": "longitude",
            "y": "latitude",
            "lon": None,
            "lat": None,
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            'time_bounds': "time_bnds",
            #         'vertex': 'vertices',
            #         'dzt': 'thkcello',
         },  
        'IPSL-CM6A-LR': {
            "x": ['x', "lon"],
            "y": ['y', "lat"],
            "lon": 'nav_lon',
            "lat": 'nav_lat',
            "lev": ["lev","deptht", "olevel"],
            "lev_bounds": ["lev_bounds", "deptht_bounds",'olevel_bounds'],
            "lon_bounds": "bounds_nav_lon",
            "lat_bounds": "bounds_nav_lat",
            'vertex': 'nvertex',
            "bnds":"axis_nbounds",
            'time_bounds': "time_bnds",
            #         'dzt': 'thkcello',
        },
        'NorCPM1': {
            "x": ["i", 'lon'],
            "y": ["j", 'lat'],
            "lon": 'longitude',
            "lat": 'latitude',
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": None,
            "lat_bounds": None,
            'vertex': 'vertices',
            'time_bounds': "time_bnds",
        },
        'NorESM1-F': {
            "x": ["i", 'lon'],
            "y": ["j", 'lat'],
            "lon": 'longitude',
            "lat": 'latitude',
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": None,
            "lat_bounds": None,
            'vertex': 'vertices',
            'time_bounds': "time_bnds",
        },
        'MPI-ESM1-2-HR': {
            "x": ["i", 'lon'],
            "y": ["j", 'lat'],
            "lon": 'longitude',
            "lat": 'latitude',
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": None,
            "lat_bounds": None,
            'vertex': 'vertices',
            'time_bounds': "time_bnds",
        },
    }
    # cast all str into lists
    for model in dim_name_dict.keys():
        for field in dim_name_dict[model].keys():
            if isinstance(dim_name_dict[model][field], str) or dim_name_dict[model][field] is None :
                dim_name_dict[model][field] = [dim_name_dict[model][field]]
    return dim_name_dict

def rename_cmip6_raw(ds, dim_name_di, printing=False, debug=False, verbose=False):
    """Homogenizes cmip6 dadtasets to common naming and e.g. vertex order"""
    ds = ds.copy()
    source_id = ds.attrs['source_id']
    
    # Check if there is an entry in the dict matching the source id
    if debug:
        print(dim_name_di.keys())
    # rename variables
    if len(dim_name_di) == 0:
        warnings.warn('input dictionary empty for source_id: `%s`. Please add values to https://github.com/jbusecke/cmip6_preprocessing/blob/master/cmip6_preprocessing/preprocessing.py' %ds.attrs['source_id'])
    else:
        
        for di in dim_name_di.keys():
            
            if debug or printing or verbose:
                print(di)
                print(dim_name_di[di])
            
            # make sure the input is a list
            if isinstance(dim_name_di[di], str):
                dim_name_di[di] = [dim_name_di[di]]
            
            
            if di in ds.variables:
                # if the desired key is already present do nothing
                if printing:
                    print("Skipped renaming for [%s]. Name already correct." % di)
                
            else:
                
                # Track if the dimension was renamed already...
                # For some source ids (e.g. CNRM-ESM2-1) the key 'x':['x', 'lon'], leads to problems
                # because it renames the 2d lon in the gr grid into x. Ill try to fix this, below.
                # But longterm its probably better to go and put out another rename dict for gn and 
                # go back to not support lists of dim names. 
            
                # For now just stop in the list if the dimension is already there, or one 'hit'
                # was already encountered.
                trigger = False
                for wrong in dim_name_di[di]:
                    if printing:
                        print('Processing %s. Trying to replace %s' %(di, wrong))
                    if wrong in ds.variables or wrong in ds.dims:
                        if not trigger:
                            if debug:
                                print('Changing %s to %s' %(wrong, di))
                            ds = ds.rename({wrong: di})
                            trigger = True
                            if printing:
                                print('Renamed.')
                    else:
                        if wrong is None:
                            if printing:
                                print("No variable available for [%s]" % di)
    return ds

def rename_cmip6(ds, **kwargs):
    """rename dataset to a uniform dim/coords naming structure"""
    modelname = ds.attrs['source_id']
    rename_dict = cmip6_renaming_dict()
    if modelname not in rename_dict.keys():
        warnings.warn('No input dictionary entry for source_id: `%s`. Please add values to https://github.com/jbusecke/cmip6_preprocessing/blob/master/cmip6_preprocessing/preprocessing.py' %ds.attrs['source_id'])
        return ds
    else:
        return rename_cmip6_raw(ds, rename_dict[modelname],**kwargs)

def promote_empty_dims(ds):
    """Convert empty dimensions to actual coordinates"""
    ds = ds.copy()
    for di in ds.dims:
        if di not in ds.coords:
            ds.coords[di] = ds[di]
    return ds

def replace_x_y_nominal_lat_lon(ds):
    """Approximate the dimensional values of x and y with mean lat and lon at the equator"""
    ds = ds.copy()
    if 'x' in ds.dims and 'y' in ds.dims:
        
        nominal_y = ds.lat.mean('x')
        # extract the equatorial lat and take those lon values as nominal lon
        eq_ind = abs(ds.lat.mean('x')).load().argmin().data
        nominal_x = ds.lon.isel(y=eq_ind)
        ds.coords['x'].data = nominal_x.data
        ds.coords['y'].data = nominal_y.data

        ds = ds.sortby('x')
        ds = ds.sortby('y')
    
    else:
        warnings.warn('No x and y found in dimensions for source_id:%s. This likely means that you forgot to rename the dataset or this is the German unstructured model' %ds.attrs['source_id'])
    return ds


# some of the models do not have 2d lon lats, correct that.
def broadcast_lonlat(ds, verbose=True):
    """Some models (all `gr` grid_labels) have 1D lon lat arrays
    This functions broadcasts those so lon/lat are always 2d arrays."""
    if 'lon' not in ds.variables:
        ds.coords['lon'] = ds['x']
    if 'lat' not in ds.variables:
        ds.coords['lat'] = ds['y']
        
    if len(ds['lon'].dims) < 2:
        ds.coords["lon"] = ds["lon"] * xr.ones_like(ds["lat"])
    if len(ds['lat'].dims) < 2:
        ds.coords["lat"] = xr.ones_like(ds["lon"]) * ds["lat"]
    return ds

def unit_conversion_dict():
    """Units conversion database"""
    unit_dict = {
        'm':{'centimeters':1/100}
    }
    return unit_dict

def correct_units(ds, verbose=False, stric=False):
    "Converts coordinates into SI units using `unit_conversion_dict`"
    unit_dict = unit_conversion_dict()
    ds = ds.copy()
    # coordinate conversions
    for co, expected_unit in [('lev','m')]:
        if co in ds.coords:
            if 'units' in ds.coords[co].attrs.keys():
                unit = ds.coords[co].attrs['units']
                if unit != expected_unit:
                    print('%s: Unexpected unit (%s) for coordinate `%s` detected.' %(ds.attrs['source_id'],unit, co))
                    if unit in unit_dict[expected_unit].keys():
                        factor = unit_dict[expected_unit][unit]
                        ds.coords[co] = ds.coords[co] * factor
                        ds.coords[co].attrs['units'] = expected_unit
                        print('\t Converted to `m`')
                    else:
                        print('No conversion found in unit_dict')
            else:
                print('%s: No units found' %ds.attrs['source_id'])
                      
        else:
            if verbose:
                print('`%s` not found as coordinate' %co)
    return ds

def correct_coordinates(ds, verbose=False):
    """converts wrongly assigned data_vars to coordinates"""
    ds = ds.copy()
    for co in ['x', 'y', 'lon', 'lat', 'lev',
               "bnds", "lev_bounds", "lon_bounds", "lat_bounds", "time_bounds",
               'vertices_latitude', 'vertices_longitude',
              ]:
        if co in ds.variables:
            if verbose:
                print('setting %s as coord' %(co))
            ds = ds.set_coords(co)
    return ds

def correct_lon(ds):
    """Wraps negative x and lon values around to have 0-360 lons.
    longitude names expected to be corrected with `rename_cmip6`"""
    ds = ds.copy()
    x = ds['x'].data
    ds['x'].data = np.where(x < 0 , 360 + x, x)

    lon = ds['lon'].data
    ds['lon'].data = np.where(lon < 0 , 360 + lon, lon)
    
    ds = ds.sortby('x')
    return ds

def combined_preprocessing(ds):
    if 'AWI' not in ds.attrs['source_id']:
        ds = ds.copy()
        # fix naming
        ds = rename_cmip6(ds)
        # promote empty dims to actual coordinates
        ds = promote_empty_dims(ds)
        # broadcast lon/lat
        ds = broadcast_lonlat(ds)
        # replace x,y with nominal lon,lat
        ds = replace_x_y_nominal_lat_lon(ds)
        # shift all lons to consistent 0-360
        ds = correct_lon(ds)        
        # demote coordinates from data_variables (this is somehow reversed in intake)
        ds = correct_coordinates(ds)
        # fix the units
        ds = correct_units(ds)
    return ds


# These are all the old high level ones... they should be deprecated

# def read_data(col, preview=False, required_variable_id=True, to_dataset_kwargs={}, **kwargs):
#     """Read data from catalogue with correct reconstruction of staggered grid"""
    
#     data_dict = import_data(col, preview=preview, required_variable_id=required_variable_id, **kwargs, to_dataset_kwargs=to_dataset_kwargs)
        
#     data_final = {}
#     for modelname, v_dict in {k:v for k,v in data_dict.items() if k not in ['AWI-CM-1-1-MR']}.items():
#         print(modelname)
#         data_final[modelname] = full_preprocessing(v_dict, modelname, plot=False)
#     return data_final
    

# def import_data(col, preview=False, required_variable_id=True,  to_dataset_kwargs={}, **kwargs):
#     if preview:
#         print(col.search(**kwargs).df)
    
#     variables = kwargs.pop('variable_id', None)
#     if variables is None:
#         raise ValueError('You need to pass at least one value for `variable_id`')
#     elif isinstance(variables, str):
#         variables = [variables]
    
#     # read each variables in separate catalogue
#     cat = {}
#     for var in variables:
#         cat[var] = col.search(variable_id=var, **kwargs)
    
#     # determine unique model names
#     models = pd.concat([k.df for k in cat.values()])['source_id'].unique()

#     data_dict = {}
#     for model in models:
#         data_dict[model] = {var:cat[var].df[cat[var].df.source_id == model]['zstore'].values for var in variables}
#     to_dataset_kwargs.setdefault(zarr_kwargs,{'consolidated': True})
#     ds_dict = {var: cat[var].to_dataset_dict(**to_dataset_kwargs) for var in variables}

    
#     data_dict = {}
#     for model in models:
#         data_dict[model] = {
#             var:[ds_dict[var][key] for key in ds_dict[var].keys() if '.'+model+'.' in key] for var in variables
#         }
#         # clean up the entries
#         #erase empty cells
#         data_dict[model] = {k:v for k,v in data_dict[model].items() if len(v) > 0}
#         #Check for double entries
#         for k, v in data_dict[model].items():
#             if len(v) == 1:
#                 data_dict[model][k] = v[0]
#             elif len(v) > 1:
#                 print(cat[k].df[cat[k].df.source_id == model])
#                 print(v)
#                 raise ValueError('You loaded two different datasets for model [%s] variable [%s]. Tighten your criteria and loop over them.' %(model,k))
    
#     # pick models that have all the specified variables           
#     if required_variable_id:
#         if required_variable_id == True:
#             required_variable_id = variables
#         data_dict = {k:v for k,v in data_dict.items() if set(required_variable_id).issubset(set(v.keys()))}
#     return data_dict

# def full_preprocessing(dat_dict, modelname,
#                        tracer_ref="thetao",
#                        u_ref="uo",
#                        v_ref="vo",
#                        plot=True,
#                        verbose=False):
#     """Fully preprocess data for one model ensemble .
#     The input needs to be a dictionary in the form:
#     {'<source_id>':{'<varname_a>':'<uri_a>', '<varname_b>':'<uri_b>', ...}}
#     """
#     renaming_dict = cmip6_renaming_dict()
#     # homogenize the naming
#     dat_dict = {
#         var: rename_cmip6_raw(data, renaming_dict[modelname])
#         for var, data in dat_dict.items()
#     }

#     # broadcast lon and lat values if they are 1d
#     if renaming_dict[modelname]["lon"] is None:
#         dat_dict = {var: broadcast_lonlat(data) for var, data in dat_dict.items()}
    
#     # merge all variables together on the correct staggered grid
#     ds = merge_variables_on_staggered_grid(
#         dat_dict, modelname, u_ref=u_ref, v_ref=v_ref,plot=plot, verbose=verbose
#     )
#     try:
#         grid_temp = Grid(ds) 
#     except:
#         print(ds)

#     ds = recreate_metrics(ds, grid_temp)
#     return ds