# Preprocessing for CMIP6 models
import xarray as xr
import pandas as pd
import warnings
from xgcm import Grid
from .recreate_grids import merge_variables_on_staggered_grid, recreate_metrics

def rename_cmip6(ds, **kwargs):
    modelname = ds.attrs['source_id']
    return cmip6_homogenization(ds, cmip6_renaming_dict()[modelname], printing=False)


def read_data(col, preview=False, required_variable_id=True, **kwargs):
    """Read data from catalogue with correct reconstruction of staggered grid"""
    
    data_dict = import_data(col, preview=preview, required_variable_id=required_variable_id, **kwargs)
        
    data_final = {}
    for modelname, v_dict in {k:v for k,v in data_dict.items() if k not in ['AWI-CM-1-1-MR']}.items():
        print(modelname)
        data_final[modelname] = full_preprocessing(v_dict, modelname, plot=False)
    return data_final
    

def import_data(col, preview=False, required_variable_id=True, **kwargs):
    if preview:
        print(col.search(**kwargs).df)
    
    variables = kwargs.pop('variable_id', None)
    if variables is None:
        raise ValueError('You need to pass at least one value for `variable_id`')
    elif isinstance(variables, str):
        variables = [variables]
    
    # read each variables in separate catalogue
    cat = {}
    for var in variables:
        cat[var] = col.search(variable_id=var, **kwargs)
    
    # determine unique model names
    models = pd.concat([k.df for k in cat.values()])['source_id'].unique()

    data_dict = {}
    for model in models:
        data_dict[model] = {var:cat[var].df[cat[var].df.source_id == model]['zstore'].values for var in variables}
        
    ds_dict = {var: cat[var].to_dataset_dict(zarr_kwargs={'consolidated': True}, 
                                cdf_kwargs={'chunks': {20}}) for var in variables}

    
    data_dict = {}
    for model in models:
        data_dict[model] = {
            var:[ds_dict[var][key] for key in ds_dict[var].keys() if '.'+model+'.' in key] for var in variables
        }
        # clean up the entries
        #erase empty cells
        data_dict[model] = {k:v for k,v in data_dict[model].items() if len(v) > 0}
        #Check for double entries
        for k, v in data_dict[model].items():
            if len(v) == 1:
                data_dict[model][k] = v[0]
            elif len(v) > 1:
                print(cat[k].df[cat[k].df.source_id == model])
                print(v)
                raise ValueError('You loaded two different datasets for model [%s] variable [%s]. Tighten your criteria and loop over them.' %(model,k))
    
    # pick models that have all the specified variables           
    if required_variable_id:
        if required_variable_id == True:
            required_variable_id = variables
        data_dict = {k:v for k,v in data_dict.items() if set(required_variable_id).issubset(set(v.keys()))}
    return data_dict

def full_preprocessing(dat_dict, modelname,
                       tracer_ref="thetao",
                       u_ref="uo",
                       v_ref="vo",
                       plot=True,
                       verbose=False):
    """Fully preprocess data for one model ensemble .
    The input needs to be a dictionary in the form:
    {'<source_id>':{'<varname_a>':'<uri_a>', '<varname_b>':'<uri_b>', ...}}
    """
    renaming_dict = cmip6_renaming_dict()
    # homogenize the naming
    dat_dict = {
        var: cmip6_homogenization(data, renaming_dict[modelname])
        for var, data in dat_dict.items()
    }

    # broadcast lon and lat values if they are 1d
    if renaming_dict[modelname]["lon"] is None:
        dat_dict = {var: broadcast_lonlat(data) for var, data in dat_dict.items()}
    
    # merge all variables together on the correct staggered grid
    ds = merge_variables_on_staggered_grid(
        dat_dict, modelname, u_ref=u_ref, v_ref=v_ref,plot=plot, verbose=verbose
    )
    try:
        grid_temp = Grid(ds) 
    except:
        print(ds)

    ds = recreate_metrics(ds, grid_temp)
    return ds


def cmip6_homogenization(ds, dim_name_di, printing=False):
    """Homogenizes cmip6 dadtasets to common naming and e.g. vertex order"""
    ds = ds.copy()
    source_id = ds.attrs['source_id']
    # rename variables
    for di in dim_name_di.keys():
        if isinstance(dim_name_di[di], str):
            dim_name_di[di] = [dim_name_di[di]]
        for wrong in dim_name_di[di]:
            if di != wrong and di not in ds.variables:
                if wrong in ds.variables:
                    ds = ds.rename({wrong: di})
                else:
                    if wrong is None:
                        if printing:
                            print("No variable available for [%s]" % di)
                        if di == "lon":
                            ds["lon"] = ds["x"]
                            if printing:
                                print("Filled lon with x")
                        elif di == "lat":
                            ds["lat"] = ds["y"]
                            if printing:
                                print("Filled lat with y")
# I think this is not necessary to warn? Maybe warn about unused values....then I can remove some entries from the dict...for later
#                     else:
#                         warnings.warn(
#                             "[%s]Variable [%s] not found in %s" % (source_id,wrong, list(ds.variables))
#                         )
        else:
            if printing:
                print("Skipped renaming for [%s]. Name already correct." % di)
    return ds


# some of the models do not have 2d lon lats, correct that.
def broadcast_lonlat(ds):
    ds.coords["lon"] = ds["lon"] * xr.ones_like(ds["lat"])
    ds.coords["lat"] = xr.ones_like(ds["lon"]) * ds["lat"]
    return ds


# preprocess the data (this is manual and annoying. We should have one central `renamer` function)
def cmip6_renaming_dict():
    dim_name_dict = {
        "BCC-CSM2-MR": {
            "x": "lon",
            "y": "lat",
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bounds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            "vertex": "vertex",
            #         'dzt': 'thkcello',
        },
        "BCC-ESM1": {
            "x": "lon",
            "y": "lat",
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bounds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            "vertex": "vertex",
            #         'dzt': 'thkcello',
        },
        "CAMS-CSM1-0": {
            "x": "i",
            "y": "j",
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bounds",
            "lon_bounds": None,
            "lat_bounds": None,
            "vertex": None,
            #         'dzt': 'thkcello',
        },
        "CanESM5": {
            "x": "i",
            "y": "j",
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": None,
            "lat_bounds": None,
            "vertex": None,
            #         'dzt': 'thkcello',
        },
        "CNRM-CM6-1": {
            "x": "x",
            "y": "y",
            "lon": "lon",
            "lat": "lat",
            "lev": "lev",
            "lev_bounds": "lev_bounds",
            "lon_bounds": None,
            "lat_bounds": None,
            #         'vertex': 'nvertex',
            #         'dzt': 'thkcello',
        },
        "CNRM-ESM2-1": {
            "x": "x",
            "y": "y",
            "lon": "lon",
            "lat": "lat",
            "lev": "lev",
            "lev_bounds": "lev_bounds",
            "lon_bounds": None,
            "lat_bounds": None,
            #         'vertex': 'nvertex',
            #         'dzt': 'thkcello',
        },
        "EC-Earth3-LR": {
            "x": "i",
            "y": "j",
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bounds",
            "lon_bounds": None,
            "lat_bounds": None,
            #         'vertex': 'vertices',
            #         'dzt': 'thkcello',
        },
        "EC-Earth3-Veg": {
            "x": "i",
            "y": "j",
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bounds",
            "lon_bounds": None,
            "lat_bounds": None,
            #         'vertex': 'vertices',
            #         'dzt': 'thkcello',
        },
        "EC-Earth3": {
            "x": "i",
            "y": "j",
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bounds",
            "lon_bounds": None,
            "lat_bounds": None,
            #         'vertex': 'vertices',
            #         'dzt': 'thkcello',
        },
        "MIROC-ES2L": {
            "x": "x",
            "y": "y",
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bounds",
            "lon_bounds": "x_bnds",
            "lat_bounds": "y_bnds",
            #         'vertex': 'vertices',
            #         'dzt': 'thkcello',
        },
        "MIROC6": {
            "x": "x",
            "y": "y",
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bounds",
            "lon_bounds": "x_bnds",
            "lat_bounds": "y_bnds",
            #         'vertex': 'vertices',
            #         'dzt': 'thkcello',
        },
        "HadGEM3-GC31-LL": {
            "x": "i",
            "y": "j",
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bounds",
            "lon_bounds": None,
            "lat_bounds": None,
            #         'vertex': 'vertices',
            #         'dzt': 'thkcello',
        },
        "UKESM1-0-LL": {
            "x": "i",
            "y": "j",
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bounds",
            "lon_bounds": None,
            "lat_bounds": None,
            #         'vertex': 'vertices',
            #         'dzt': 'thkcello',
        },
        "GISS-E2-1-G-CC": {
            "x": "lon",
            "y": "lat",
            "lon": None,
            "lat": None,
            "lev": "lev",
            "lev_bounds": "lev_bounds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            #         'vertex': None,
            #         'dzt': 'thkcello',
        },
        "GISS-E2-1-G": {
            "x": "lon",
            "y": "lat",
            "lon": None,
            "lat": None,
            "lev": "lev",
            "lev_bounds": "lev_bounds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            #         'vertex': None,
            #         'dzt': 'thkcello',
        },
        "CESM2-WACCM": {
            "x": "nlon",
            "y": "nlat",
            "lon": "lon",
            "lat": "lat",
            "lev": "lev",
            "lev_bounds": "lev_bounds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            #         'vertex': 'vertices',
            #         'dzt': 'thkcello',
        },
        "CESM2": {
            "x": "nlon",
            "y": "nlat",
            "lon": "lon",
            "lat": "lat",
            "lev": "lev",
            "lev_bounds": "lev_bounds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            #         'vertex': 'vertices',
            #         'dzt': 'thkcello',
        },
        "GFDL-CM4": {
            "x": "x",
            "y": "y",
            "lon": "lon",
            "lat": "lat",
            "lev": "lev",
            "lev_bounds": "lev_bounds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            #         'vertex': 'vertex',
            #         'dzt': 'thkcello',
        },
        "GFDL-ESM4": {
            "x": "x",
            "y": "y",
            "lon": "lon",
            "lat": "lat",
            "lev": "lev",
            "lev_bounds": "lev_bounds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            #         'vertex': 'vertex',
            #         'dzt': 'thkcello',
        },
        "NESM3": {
            "x": "i",
            "y": "j",
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bounds",
            "lon_bounds": None,
            "lat_bounds": None,
            #         'vertex': 'vertices',
            #         'dzt': 'thkcello',
        },
        "SAM0-UNICON": {
            "x": "i",
            "y": "j",
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bounds",
            "lon_bounds": None,
            "lat_bounds": None,
            #         'vertex': 'vertices',
            #         'dzt': 'thkcello',
        },
        "MCM-UA-1-0": {
            "x": "longitude",
            "y": "latitude",
            "lon": None,
            "lat": None,
            "lev": "lev",
            "lev_bounds": "lev_bounds",
            "lon_bounds": "lon_bounds",
            "lat_bounds": "lat_bounds",
            #         'vertex': 'vertices',
            #         'dzt': 'thkcello',
         },  
        'IPSL-CM6A-LR': {
            "x": "x",
            "y": "y",
            "lon": 'nav_lon',
            "lat": 'nav_lat',
            "lev": ["lev","deptht", "olevel"],
            "lev_bounds": "lev_bounds",
            "lon_bounds": "bounds_nav_lon",
            "lat_bounds": "bounds_nav_lat",
            'vertex': 'nvertex',
            #         'dzt': 'thkcello',
        },
        'NorCPM1': {
            "x": "i",
            "y": "j",
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
