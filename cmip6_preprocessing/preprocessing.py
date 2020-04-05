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
        "AWI-CM-1-1-MR": {},
        "AWI-CM-1-1-LR": {},
        "AWI-ESM-1-1-LR": {},
        "BCC-CSM2-MR": {
            "x": "lon",
            "y": "lat",
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "bnds": "bnds",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            "vertex": None,
            "time_bounds": "time_bnds",
        },
        "BCC-ESM1": {
            "x": "lon",
            "y": "lat",
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "bnds": "bnds",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            "vertex": "vertex",
            "time_bounds": "time_bnds",
        },
        "CAMS-CSM1-0": {
            "x": ["i", "lon"],
            "y": ["j", "lat"],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": None,
            "lat_bounds": None,
            "vertex": "vertices",
            "time_bounds": "time_bnds",
        },
        "CanESM5": {
            "x": ["i", "lon"],
            "y": ["j", "lat"],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": None,
            "lat_bounds": None,
            "time_bounds": "time_bnds",
            "vertex": "vertices",
        },
        "CanESM5-CanOE": {
            "x": "i",
            "y": "j",
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "vertices_longitude",
            "lat_bounds": "vertices_latitude",
            "vertex": "vertices",
        },
        "CNRM-CM6-1": {
            "x": ["x", "lon"],
            "y": ["y", "lat"],
            "lon": "lon",
            "lat": "lat",
            "lev": "lev",
            "bnds": "axis_nbounds",
            "lev_bounds": "lev_bounds",
            "lon_bounds": "bounds_lon",
            "lat_bounds": "bounds_lat",
            "vertex": "nvertex",
            "time_bounds": "time_bnds",
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
            "bnds": "axis_nbounds",
            "vertex": None,
            "time_bounds": "time_bnds",
        },
        "E3SM-1-0": {
            "x": "lon",
            "y": "lat",
            "lon": None,
            "lat": None,
            "lev": "lev",
            "bnds": "bnds",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            "time_bounds": "time_bounds",
            "vertex": None,
        },
        "E3SM-1-1": {
            "x": "lon",
            "y": "lat",
            "lon": None,
            "lat": None,
            "lev": "lev",
            "bnds": "bnds",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            "time_bounds": "time_bounds",
            "vertex": None,
        },
        "E3SM-1-1-ECA": {
            "x": "lon",
            "y": "lat",
            "lon": None,
            "lat": None,
            "lev": "lev",
            "bnds": "bnds",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            "time_bounds": "time_bounds",
            "vertex": None,
        },
        "EC-Earth3-LR": {
            "x": ["i", "lon"],
            "y": ["j", "lat"],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": None,
            "lat_bounds": None,
            "time_bounds": "time_bnds",
            #         'vertex': 'vertices',
            #         'dzt': 'thkcello',
        },
        "EC-Earth3-Veg": {
            "x": ["i", "lon"],
            "y": ["j", "lat"],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": None,
            "lat_bounds": None,
            "vertex": "vertices",
            "time_bounds": "time_bnds",
            #         'dzt': 'thkcello',
        },
        "EC-Earth3": {
            "x": ["i", "lon"],
            "y": ["j", "lat"],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": None,
            "lat_bounds": None,
            "time_bounds": "time_bnds",
            #         'vertex': 'vertices',
            #         'dzt': 'thkcello',
        },
        "FGOALS-f3-L": {
            "x": ["i", "lon"],
            "y": ["j", "lat"],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": None,
            "lat_bounds": None,
            "time_bounds": "time_bnds",
            #         'vertex': 'vertices',
            #         'dzt': 'thkcello',
        },
        "NICAM16-7S": {
            "x": "lon",
            "y": "lat",
            "lon": None,
            "lat": None,
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            "time_bounds": "time_bnds",
            "vertex": "vertices",
        },
        "MIROC-ES2L": {
            "x": ["x", "lon"],
            "y": ["y", "lat"],
            "lon": "longitude",
            "lat": "latitude",
            "lev": ["lev", "zlev"],
            "lev_bounds": ["lev_bnds", "zlev_bnds"],
            "lon_bounds": "x_bnds",
            "lat_bounds": "y_bnds",
            "time_bounds": "time_bnds",
            "vertex": "vertices",
        },
        "MIROC6": {
            "x": ["x", "lon"],
            "y": ["y", "lat"],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "x_bnds",
            "lat_bounds": "y_bnds",
            "time_bounds": "time_bnds",
        },
        "HadGEM3-GC31-LL": {
            "x": ["i", "lon"],
            "y": ["j", "lat"],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bounds",
            "lon_bounds": None,
            "lat_bounds": None,
            "time_bounds": "time_bnds",
        },
        "HadGEM3-GC31-MM": {
            "x": ["i", "lon"],
            "y": ["j", "lat"],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bounds",
            "lon_bounds": None,
            "lat_bounds": None,
            "time_bounds": "time_bnds",
        },
        "UKESM1-0-LL": {
            "x": ["i", "lon"],
            "y": ["j", "lat"],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": None,
            "lat_bounds": None,
            "time_bounds": "time_bnds",
            #         'vertex': 'vertices',
            #         'dzt': 'thkcello',
        },
        "GISS-E2-2-G": {
            "x": "lon",
            "y": "lat",
            "lon": None,
            "lat": None,
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            "vertex": None,
            "time_bounds": "time_bnds",
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
            "vertex": None,
            "time_bounds": "time_bnds",
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
            "vertex": None,
            "time_bounds": "time_bnds",
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
            "vertex": None,
            "time_bounds": "time_bnds",
        },
        "CESM1-1-CAM5-CMIP5": {
            "x": ["nlon", "lon"],
            "y": ["nlat", "lat"],
            "lon": "lon",
            "lat": "lat",
            "lev": "lev",
            "bnds": "d2",
            "time_bounds": "time_bnds",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            "vertex": "vertices",
        },
        "CESM2-WACCM": {
            "x": ["nlon", "lon"],
            "y": ["nlat", "lat"],
            "lon": "lon",
            "lat": "lat",
            "lev": "lev",
            "bnds": "d2",
            "time_bounds": "time_bnds",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            "vertex": "vertices",
        },
        "CESM2-WACCM-FV2": {
            "x": ["nlon", "lon"],
            "y": ["nlat", "lat"],
            "lon": "lon",
            "lat": "lat",
            "lev": "lev",
            "bnds": "d2",
            "time_bounds": "time_bnds",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            "vertex": "vertices",
        },
        "CESM2": {
            "x": ["nlon", "lon"],
            "y": ["nlat", "lat"],
            "lon": "lon",
            "lat": "lat",
            "lev": "lev",
            "bnds": "d2",
            "time_bounds": "time_bnds",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            "vertex": "vertices",
        },
        "CESM2-FV2": {
            "x": ["nlon", "lon"],
            "y": ["nlat", "lat"],
            "lon": "lon",
            "lat": "lat",
            "lev": "lev",
            "bnds": "d2",
            "time_bounds": "time_bnds",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            "vertex": "vertices",
        },
        "GFDL-CM4": {
            "x": ["x", "lon"],
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
        "GFDL-OM4p5B": {
            "x": ["x", "lon"],
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
            "x": ["x", "lon"],
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
            "x": ["i", "lon"],
            "y": ["j", "lat"],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": None,
            "lat_bounds": None,
            "vertex": "vertices",
            "time_bounds": "time_bnds",
            #         'dzt': 'thkcello',
        },
        "MRI-ESM2-0": {
            "x": ["x", "lon"],
            "y": ["y", "lat"],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "bnds": "bnds",
            "lev_bounds": "lev_bnds",
            "lon_bounds": ["x_bnds", "lon_bnds"],
            "lat_bounds": ["y_bnds", "lat_bnds"],
            "time_bounds": "time_bnds",
            "vertex": "vertices",
        },
        "SAM0-UNICON": {
            "x": ["i", "lon"],
            "y": ["j", "lat"],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": None,
            "lat_bounds": None,
            "vertex": "vertices",
            "time_bounds": "time_bnds",
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
            "time_bounds": "time_bnds",
            #         'vertex': 'vertices',
            #         'dzt': 'thkcello',
        },
        "IPSL-CM6A-LR": {
            "x": ["x", "lon"],
            "y": ["y", "lat"],
            "lon": "nav_lon",
            "lat": "nav_lat",
            "lev": ["lev", "deptht", "olevel"],
            "lev_bounds": ["lev_bounds", "deptht_bounds", "olevel_bounds"],
            "lon_bounds": "bounds_nav_lon",
            "lat_bounds": "bounds_nav_lat",
            "vertex": "nvertex",
            "bnds": "axis_nbounds",
            "time_bounds": "time_bnds",
            #         'dzt': 'thkcello',
        },
        "NorCPM1": {
            "x": ["i", "lon"],
            "y": ["j", "lat"],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": None,
            "lat_bounds": None,
            "vertex": "vertices",
            "time_bounds": "time_bnds",
        },
        "NorESM1-F": {
            "x": ["i", "lon"],
            "y": ["j", "lat"],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "vertices_longitude",
            "lat_bounds": "vertices_latitude",
            "vertex": "vertices",
            "time_bounds": "time_bnds",
        },
        "NorESM2-LM": {
            "x": "i",
            "y": "j",
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "vertices_longitude",
            "lat_bounds": "vertices_latitude",
            "vertex": "vertices",
            "time_bounds": "time_bnds",
        },
        "NorESM2-MM": {
            "x": "i",
            "y": "j",
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",  # i leave this here because the names are the same as for the other Nor models.
            "lev_bounds": "lev_bnds",
            "lon_bounds": "vertices_longitude",
            "lat_bounds": "vertices_latitude",
            "vertex": "vertices",
            "time_bounds": "time_bnds",
        },
        "MPI-ESM1-2-HR": {
            "x": ["i", "lon"],
            "y": ["j", "lat"],
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "vertices_longitude",
            "lat_bounds": "vertices_latitude",
            "vertex": "vertices",
            "time_bounds": "time_bnds",
        },
        "MPI-ESM1-2-LR": {
            "x": "i",
            "y": "j",
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "vertices_longitude",
            "lat_bounds": "vertices_latitude",
            "vertex": "vertices",
            "time_bounds": "time_bnds",
        },
        "MPI-ESM-1-2-HAM": {
            "x": "i",
            "y": "j",
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "vertices_longitude",
            "lat_bounds": "vertices_latitude",
            "vertex": "vertices",
            "time_bounds": "time_bnds",
        },
        "CNRM-CM6-1-HR": {
            "x": "x",
            "y": "y",
            "lon": "lon",
            "lat": "lat",
            "lev": "lev",
            "lev_bounds": "lev_bounds",
            "lon_bounds": "bounds_lon",
            "lat_bounds": "bounds_lat",
            "vertex": None,
            "time_bounds": "time_bounds",
        },
        "FIO-ESM-2-0": {
            "x": "i",
            "y": "j",
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "vertices_longitude",
            "lat_bounds": "vertices_latitude",
            "vertex": "vertices",
            "time_bounds": "time_bnds",
        },
        "ACCESS-ESM1-5": {
            "x": "i",
            "y": "j",
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "vertices_longitude",
            "lat_bounds": "vertices_latitude",
            "vertex": "vertices",
            "time_bounds": "time_bnds",
        },
        "ACCESS-CM2": {
            "x": "i",
            "y": "j",
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "vertices_longitude",
            "lat_bounds": "vertices_latitude",
            "vertex": "vertices",
            "time_bounds": "time_bnds",
        },
        "INM-CM4-8": {  # this is a guess.
            "x": "lon",
            "y": "lat",
            "lon": None,
            "lat": None,
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            "vertex": None,
            "time_bounds": "time_bnds",
        },
        "INM-CM5-0": {  # this is a guess.
            "x": "lon",
            "y": "lat",
            "lon": None,
            "lat": None,
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            "lon_bounds": "lon_bnds",
            "lat_bounds": "lat_bnds",
            "vertex": None,
            "time_bounds": "time_bnds",
        },
        "MRI-ESM2-0": {
            "x": "x",
            "y": "y",
            "lon": "longitude",
            "lat": "latitude",
            "lev": "lev",
            "lev_bounds": "lev_bnds",
            #             "lon_bounds": 'x_bnds',
            #             "lat_bounds": 'y_bnds',
            #             'vertex': None, # this is a mess. there is yet another convention. Will have to deal with this once I wrap xgcm into here.
            "time_bounds": "time_bnds",
        },
        "CIESM": {  # this is a guess.
            "x": "i",
            "y": "j",
            "lon": "longitude",
            "lat": "latitude",
            #             "lev": "lev", # no 3d data available as of now
            #             "lev_bounds": "lev_bnds",
            "lon_bounds": "vertices_longitude",
            "lat_bounds": "vertices_latitude",
            "vertex": "vertices",
            "time_bounds": "time_bnds",
        },
        "KACE-1-0-G": {  # this is a guess.
            "x": "lon",
            "y": "lat",
            "lon": "longitude",
            "lat": "latitude",
            #             "lev": "lev", # no 3d data available as of now
            #             "lev_bounds": "lev_bnds",
            #             "lon_bounds": "vertices_longitude",
            #             "lat_bounds": "vertices_latitude",
            #             "lon_bounds": "vertices_longitude",
            #             "lat_bounds": "vertices_latitude",
            "vertex": "vertices",
            "time_bounds": "time_bnds",
        },
    }
    # cast all str into lists
    for model in dim_name_dict.keys():
        for field in dim_name_dict[model].keys():
            if (
                isinstance(dim_name_dict[model][field], str)
                or dim_name_dict[model][field] is None
            ):
                dim_name_dict[model][field] = [dim_name_dict[model][field]]
        #     add 'lon' and 'lat' as possible logical indicies for all models. This should take care of all regridded ocean output and all atmosphere models.
        if "x" in dim_name_dict[model].keys():
            if not "lon" in dim_name_dict[model]["x"]:
                dim_name_dict[model]["x"].append("lon")

        if "y" in dim_name_dict[model].keys():
            if not "lat" in dim_name_dict[model]["y"]:
                dim_name_dict[model]["y"].append("lat")
    return dim_name_dict


def rename_cmip6_raw(ds, dim_name_di, printing=False, debug=False, verbose=False):
    """Homogenizes cmip6 dadtasets to common naming and e.g. vertex order"""
    ds = ds.copy()
    source_id = ds.attrs["source_id"]

    # Check if there is an entry in the dict matching the source id
    if debug:
        print(dim_name_di.keys())
    # rename variables
    if len(dim_name_di) == 0:
        warnings.warn(
            "input dictionary empty for source_id: `%s`. Please add values to https://github.com/jbusecke/cmip6_preprocessing/blob/master/cmip6_preprocessing/preprocessing.py"
            % ds.attrs["source_id"],
            UserWarning,
        )
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
                        print("Processing %s. Trying to replace %s" % (di, wrong))
                    if wrong in ds.variables or wrong in ds.dims:
                        if not trigger:
                            if debug:
                                print("Changing %s to %s" % (wrong, di))
                            ds = ds.rename({wrong: di})
                            trigger = True
                            if printing:
                                print("Renamed.")
                    else:
                        if wrong is None:
                            if printing:
                                print("No variable available for [%s]" % di)
    return ds


def rename_cmip6(ds, rename_dict=None, **kwargs):
    """rename dataset to a uniform dim/coords naming structure"""
    modelname = ds.attrs["source_id"]
    if rename_dict is None:
        rename_dict = cmip6_renaming_dict()

    if modelname not in rename_dict.keys():
        msg = f"No input dictionary entry for source_id: `{ds.attrs['source_id']}`. \
            Please add values to https://github.com/jbusecke/cmip6_preprocessing/blob/master/cmip6_preprocessing/preprocessing.py"
        warnings.warn(msg, UserWarning)
        return ds
    else:
        return rename_cmip6_raw(ds, rename_dict[modelname], **kwargs)


def promote_empty_dims(ds):
    """Convert empty dimensions to actual coordinates"""
    ds = ds.copy()
    for di in ds.dims:
        if di not in ds.coords:
            ds = ds.assign_coords({di: ds[di]})
            # ds.coords[di] = ds[di]
    return ds


def replace_x_y_nominal_lat_lon(ds):
    """Approximate the dimensional values of x and y with mean lat and lon at the equator"""
    ds = ds.copy()
    if "x" in ds.dims and "y" in ds.dims:

        nominal_y = ds.lat.mean("x")
        # extract the equatorial lat and take those lon values as nominal lon
        eq_ind = abs(ds.lat.mean("x")).load().argmin().data
        nominal_x = ds.lon.isel(y=eq_ind)

        ds = ds.assign_coords(x=nominal_x, y=nominal_y)

        ds = ds.sortby("x")
        ds = ds.sortby("y")

    else:
        warnings.warn(
            "No x and y found in dimensions for source_id:%s. This likely means that you forgot to rename the dataset or this is the German unstructured model"
            % ds.attrs["source_id"]
        )
    return ds


# some of the models do not have 2d lon lats, correct that.
def broadcast_lonlat(ds, verbose=True):
    """Some models (all `gr` grid_labels) have 1D lon lat arrays
    This functions broadcasts those so lon/lat are always 2d arrays."""
    if "lon" not in ds.variables:
        ds.coords["lon"] = ds["x"]
    if "lat" not in ds.variables:
        ds.coords["lat"] = ds["y"]

    if len(ds["lon"].dims) < 2:
        ds.coords["lon"] = ds["lon"] * xr.ones_like(ds["lat"])
    if len(ds["lat"].dims) < 2:
        ds.coords["lat"] = xr.ones_like(ds["lon"]) * ds["lat"]
    return ds


def unit_conversion_dict():
    """Units conversion database"""
    unit_dict = {"m": {"centimeters": 1 / 100}}
    return unit_dict


def correct_units(ds, verbose=False, stric=False):
    "Converts coordinates into SI units using `unit_conversion_dict`"
    unit_dict = unit_conversion_dict()
    ds = ds.copy()
    # coordinate conversions
    for co, expected_unit in [("lev", "m")]:
        if co in ds.coords:
            if "units" in ds.coords[co].attrs.keys():
                unit = ds.coords[co].attrs["units"]
                if unit != expected_unit:
                    print(
                        "%s: Unexpected unit (%s) for coordinate `%s` detected."
                        % (ds.attrs["source_id"], unit, co)
                    )
                    if unit in unit_dict[expected_unit].keys():
                        factor = unit_dict[expected_unit][unit]
                        ds.coords[co] = ds.coords[co] * factor
                        ds.coords[co].attrs["units"] = expected_unit
                        print("\t Converted to `m`")
                    else:
                        print("No conversion found in unit_dict")
            else:
                print("%s: No units found" % ds.attrs["source_id"])

        else:
            if verbose:
                print("`%s` not found as coordinate" % co)
    return ds


def correct_coordinates(ds, verbose=False):
    """converts wrongly assigned data_vars to coordinates"""
    ds = ds.copy()
    for co in [
        "x",
        "y",
        "lon",
        "lat",
        "lev",
        "bnds",
        "lev_bounds",
        "lon_bounds",
        "lat_bounds",
        "time_bounds",
        "vertices_latitude",
        "vertices_longitude",
    ]:
        if co in ds.variables:
            if verbose:
                print("setting %s as coord" % (co))
            ds = ds.set_coords(co)
    return ds


def correct_lon(ds):
    """Wraps negative x and lon values around to have 0-360 lons.
    longitude names expected to be corrected with `rename_cmip6`"""
    ds = ds.copy()

    x = ds["x"].data
    x = np.where(x < 0, 360 + x, x)

    lon = ds["lon"].data
    lon = np.where(lon < 0, 360 + lon, lon)

    ds = ds.assign_coords(x=x, lon=(ds.lon.dims, lon))
    ds = ds.sortby("x")
    return ds


def combined_preprocessing(ds):
    if "AWI" not in ds.attrs["source_id"]:
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
