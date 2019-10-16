# Preprocessing for CMIP6 models
import xarray as xr
import warnings
from xgcm import Grid
from .recreate_grids import merge_variables_on_staggered_grid, recreate_metrics


def full_preprocessing(dat_dict, modelname, plot=False, verbose=False):
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
        dat_dict, modelname, plot=plot, verbose=verbose
    )

    grid_temp = Grid(ds)
    ds = recreate_metrics(ds, grid_temp)
    return ds


def cmip6_homogenization(ds, dim_name_di, printing=False):
    """Homogenizes cmip6 dadtasets to common naming and e.g. vertex order"""
    ds = ds.copy()
    # rename variables
    for di in dim_name_di.keys():
        if di != dim_name_di[di]:
            if dim_name_di[di] in ds.variables:
                ds = ds.rename({dim_name_di[di]: di})
            else:
                if dim_name_di[di] is None:
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
                else:
                    warnings.warn(
                        "Variable [%s] not found in %s" % (dim_name_di[di], ds.coords)
                    )
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
            "lev_bounds": "lev_bounds",
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
    }
    return dim_name_dict
