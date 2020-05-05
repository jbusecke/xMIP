# Preprocessing for CMIP6 models
import xarray as xr
import pandas as pd
import numpy as np
import warnings


def cmip6_renaming_dict():
    """a universal renaming dict. Keys correspond to source id (model name)
    and valuse are a dict of target name (key) and a list of variables that
    should be renamed into the target."""
    rename_dict = {
        # dim labels
        "x": ["x", "i", "nlon", "lon", "longitude"],
        "y": ["y", "j", "nlat", "lat", "latitude"],
        "lev": ["lev", "deptht", "olevel", "zlev", "olev"],
        "bnds": ["bnds", "axis_nbounds", "d2"],
        "vertex": ["vertex", "nvertex", "vertices"],
        # coordinate labels
        "lon": ["lon", "longitude", "nav_lon"],
        "lat": ["lat", "latitude", "nav_lat"],
        "lev_bounds": [
            "lev_bounds",
            "deptht_bounds",
            "lev_bnds",
            "olevel_bounds",
            "zlev_bnds",
        ],
        # i rename both vertex and bounds based coordinates the same here.
        # they will be transformed later
        "lon_bounds": [
            "bounds_lon",
            "bounds_nav_lon",
            "lon_bnds",
            "vertices_longitude" "x_bnds",
        ],
        "lat_bounds": [
            "bounds_lat" "bounds_nav_lat" "lat_bnds" "vertices_latitude" "y_bnds"
        ],
        "time_bounds": ["time_bounds", "time_bnds"],
    }
    return rename_dict


#
# def cmip6_renaming_dict():
#     # I could probably simplify this with a generalized single dict,
#     # which has every single possible `wrong` name and then for each model
#     # the renaming function just goes through them...
#
#     """central database for renaming the dataset."""
#     dim_name_dict = {
#         "AWI-CM-1-1-MR": {},
#         "AWI-CM-1-1-LR": {},
#         "AWI-ESM-1-1-LR": {},
#         "BCC-CSM2-MR": {
#             "x": "lon",
#             "y": "lat",
#             "lon": "longitude",
#             "lat": "latitude",
#             "lev": "lev",
#             "bnds": "bnds",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "lon_bnds",
#             "lat_bounds": "lat_bnds",
#             "vertex": None,
#             "time_bounds": "time_bnds",
#         },
#         "BCC-ESM1": {
#             "x": "lon",
#             "y": "lat",
#             "lon": "longitude",
#             "lat": "latitude",
#             "lev": "lev",
#             "bnds": "bnds",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "lon_bnds",
#             "lat_bounds": "lat_bnds",
#             "vertex": "vertex",
#             "time_bounds": "time_bnds",
#         },
#         "CAMS-CSM1-0": {
#             "x": ["i", "lon"],
#             "y": ["j", "lat"],
#             "lon": "longitude",
#             "lat": "latitude",
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": None,
#             "lat_bounds": None,
#             "vertex": "vertices",
#             "time_bounds": "time_bnds",
#         },
#         "CanESM5": {
#             "x": ["i", "lon"],
#             "y": ["j", "lat"],
#             "lon": "longitude",
#             "lat": "latitude",
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": None,
#             "lat_bounds": None,
#             "time_bounds": "time_bnds",
#             "vertex": "vertices",
#         },
#         "CanESM5-CanOE": {
#             "x": "i",
#             "y": "j",
#             "lon": "longitude",
#             "lat": "latitude",
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "vertices_longitude",
#             "lat_bounds": "vertices_latitude",
#             "vertex": "vertices",
#         },
#         "CNRM-CM6-1": {
#             "x": ["x", "lon"],
#             "y": ["y", "lat"],
#             "lon": "lon",
#             "lat": "lat",
#             "lev": "lev",
#             "bnds": "axis_nbounds",
#             "lev_bounds": "lev_bounds",
#             "lon_bounds": "bounds_lon",
#             "lat_bounds": "bounds_lat",
#             "vertex": "nvertex",
#             "time_bounds": "time_bnds",
#         },
#         "CNRM-ESM2-1": {
#             "x": ["x", "lon"],
#             "y": ["y", "lat"],
#             "lon": "lon",
#             "lat": "lat",
#             "lev": "lev",
#             "lev_bounds": "lev_bounds",
#             "lon_bounds": "bounds_lon",
#             "lat_bounds": "bounds_lat",
#             "bnds": "axis_nbounds",
#             "vertex": None,
#             "time_bounds": "time_bnds",
#         },
#         "E3SM-1-0": {
#             "x": "lon",
#             "y": "lat",
#             "lon": None,
#             "lat": None,
#             "lev": "lev",
#             "bnds": "bnds",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "lon_bnds",
#             "lat_bounds": "lat_bnds",
#             "time_bounds": "time_bounds",
#             "vertex": None,
#         },
#         "E3SM-1-1": {
#             "x": "lon",
#             "y": "lat",
#             "lon": None,
#             "lat": None,
#             "lev": "lev",
#             "bnds": "bnds",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "lon_bnds",
#             "lat_bounds": "lat_bnds",
#             "time_bounds": "time_bounds",
#             "vertex": None,
#         },
#         "E3SM-1-1-ECA": {
#             "x": "lon",
#             "y": "lat",
#             "lon": None,
#             "lat": None,
#             "lev": "lev",
#             "bnds": "bnds",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "lon_bnds",
#             "lat_bounds": "lat_bnds",
#             "time_bounds": "time_bounds",
#             "vertex": None,
#         },
#         "EC-Earth3-LR": {
#             "x": ["i", "lon"],
#             "y": ["j", "lat"],
#             "lon": "longitude",
#             "lat": "latitude",
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": None,
#             "lat_bounds": None,
#             "time_bounds": "time_bnds",
#             #         'vertex': 'vertices',
#             #         'dzt': 'thkcello',
#         },
#         "EC-Earth3-Veg": {
#             "x": ["i", "lon"],
#             "y": ["j", "lat"],
#             "lon": "longitude",
#             "lat": "latitude",
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": None,
#             "lat_bounds": None,
#             "vertex": "vertices",
#             "time_bounds": "time_bnds",
#             #         'dzt': 'thkcello',
#         },
#         "EC-Earth3": {
#             "x": ["i", "lon"],
#             "y": ["j", "lat"],
#             "lon": "longitude",
#             "lat": "latitude",
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": None,
#             "lat_bounds": None,
#             "time_bounds": "time_bnds",
#             #         'vertex': 'vertices',
#             #         'dzt': 'thkcello',
#         },
#         "FGOALS-f3-L": {
#             "x": ["i", "lon"],
#             "y": ["j", "lat"],
#             "lon": "longitude",
#             "lat": "latitude",
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": None,
#             "lat_bounds": None,
#             "time_bounds": "time_bnds",
#             #         'vertex': 'vertices',
#             #         'dzt': 'thkcello',
#         },
#         "NICAM16-7S": {
#             "x": "lon",
#             "y": "lat",
#             "lon": None,
#             "lat": None,
#             "lon_bounds": "lon_bnds",
#             "lat_bounds": "lat_bnds",
#             "time_bounds": "time_bnds",
#             "vertex": "vertices",
#         },
#         "MIROC-ES2L": {
#             "x": ["x", "lon"],
#             "y": ["y", "lat"],
#             "lon": "longitude",
#             "lat": "latitude",
#             "lev": ["lev", "zlev"],
#             "lev_bounds": ["lev_bnds", "zlev_bnds"],
#             "lon_bounds": "x_bnds",
#             "lat_bounds": "y_bnds",
#             "time_bounds": "time_bnds",
#             "vertex": "vertices",
#         },
#         "MIROC6": {
#             "x": ["x", "lon"],
#             "y": ["y", "lat"],
#             "lon": "longitude",
#             "lat": "latitude",
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "x_bnds",
#             "lat_bounds": "y_bnds",
#             "time_bounds": "time_bnds",
#         },
#         "HadGEM3-GC31-LL": {
#             "x": ["i", "lon"],
#             "y": ["j", "lat"],
#             "lon": "longitude",
#             "lat": "latitude",
#             "lev": "lev",
#             "lev_bounds": "lev_bounds",
#             "lon_bounds": None,
#             "lat_bounds": None,
#             "time_bounds": "time_bnds",
#         },
#         "HadGEM3-GC31-MM": {
#             "x": ["i", "lon"],
#             "y": ["j", "lat"],
#             "lon": "longitude",
#             "lat": "latitude",
#             "lev": "lev",
#             "lev_bounds": "lev_bounds",
#             "lon_bounds": None,
#             "lat_bounds": None,
#             "time_bounds": "time_bnds",
#         },
#         "UKESM1-0-LL": {
#             "x": ["i", "lon"],
#             "y": ["j", "lat"],
#             "lon": "longitude",
#             "lat": "latitude",
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": None,
#             "lat_bounds": None,
#             "time_bounds": "time_bnds",
#             #         'vertex': 'vertices',
#             #         'dzt': 'thkcello',
#         },
#         "GISS-E2-2-G": {
#             "x": "lon",
#             "y": "lat",
#             "lon": None,
#             "lat": None,
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "lon_bnds",
#             "lat_bounds": "lat_bnds",
#             "vertex": None,
#             "time_bounds": "time_bnds",
#         },
#         "GISS-E2-1-G-CC": {
#             "x": "lon",
#             "y": "lat",
#             "lon": None,
#             "lat": None,
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "lon_bnds",
#             "lat_bounds": "lat_bnds",
#             "vertex": None,
#             "time_bounds": "time_bnds",
#         },
#         "GISS-E2-1-G": {
#             "x": "lon",
#             "y": "lat",
#             "lon": None,
#             "lat": None,
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "lon_bnds",
#             "lat_bounds": "lat_bnds",
#             "vertex": None,
#             "time_bounds": "time_bnds",
#         },
#         "GISS-E2-1-H": {
#             "x": "lon",
#             "y": "lat",
#             "lon": None,
#             "lat": None,
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "lon_bnds",
#             "lat_bounds": "lat_bnds",
#             "vertex": None,
#             "time_bounds": "time_bnds",
#         },
#         "CESM1-1-CAM5-CMIP5": {
#             "x": ["nlon", "lon"],
#             "y": ["nlat", "lat"],
#             "lon": "lon",
#             "lat": "lat",
#             "lev": "lev",
#             "bnds": "d2",
#             "time_bounds": "time_bnds",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "lon_bnds",
#             "lat_bounds": "lat_bnds",
#             "vertex": "vertices",
#         },
#         "CESM2-WACCM": {
#             "x": ["nlon", "lon"],
#             "y": ["nlat", "lat"],
#             "lon": "lon",
#             "lat": "lat",
#             "lev": "lev",
#             "bnds": "d2",
#             "time_bounds": "time_bnds",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "lon_bnds",
#             "lat_bounds": "lat_bnds",
#             "vertex": "vertices",
#         },
#         "CESM2-WACCM-FV2": {
#             "x": ["nlon", "lon"],
#             "y": ["nlat", "lat"],
#             "lon": "lon",
#             "lat": "lat",
#             "lev": "lev",
#             "bnds": "d2",
#             "time_bounds": "time_bnds",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "lon_bnds",
#             "lat_bounds": "lat_bnds",
#             "vertex": "vertices",
#         },
#         "CESM2": {
#             "x": ["nlon", "lon"],
#             "y": ["nlat", "lat"],
#             "lon": "lon",
#             "lat": "lat",
#             "lev": "lev",
#             "bnds": "d2",
#             "time_bounds": "time_bnds",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "lon_bnds",
#             "lat_bounds": "lat_bnds",
#             "vertex": "vertices",
#         },
#         "CESM2-FV2": {
#             "x": ["nlon", "lon"],
#             "y": ["nlat", "lat"],
#             "lon": "lon",
#             "lat": "lat",
#             "lev": "lev",
#             "bnds": "d2",
#             "time_bounds": "time_bnds",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "lon_bnds",
#             "lat_bounds": "lat_bnds",
#             "vertex": "vertices",
#         },
#         "GFDL-CM4": {
#             "x": ["x", "lon"],
#             "y": ["y", "lat"],
#             "lon": "lon",
#             "lat": "lat",
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "lon_bnds",
#             "lat_bounds": "lat_bnds",
#             "time_bounds": "time_bnds",
#             #         'vertex': 'vertex',
#             #         'dzt': 'thkcello',
#         },
#         "GFDL-OM4p5B": {
#             "x": ["x", "lon"],
#             "y": ["y", "lat"],
#             "lon": "lon",
#             "lat": "lat",
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "lon_bnds",
#             "lat_bounds": "lat_bnds",
#             "time_bounds": "time_bnds",
#             #         'vertex': 'vertex',
#             #         'dzt': 'thkcello',
#         },
#         "GFDL-ESM4": {
#             "x": ["x", "lon"],
#             "y": ["y", "lat"],
#             "lon": "lon",
#             "lat": "lat",
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "lon_bnds",
#             "lat_bounds": "lat_bnds",
#             "time_bounds": "time_bnds",
#             #         'vertex': 'vertex',
#             #         'dzt': 'thkcello',
#         },
#         "NESM3": {
#             "x": ["i", "lon"],
#             "y": ["j", "lat"],
#             "lon": "longitude",
#             "lat": "latitude",
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": None,
#             "lat_bounds": None,
#             "vertex": "vertices",
#             "time_bounds": "time_bnds",
#             #         'dzt': 'thkcello',
#         },
#         "MRI-ESM2-0": {
#             "x": ["x", "lon"],
#             "y": ["y", "lat"],
#             "lon": "longitude",
#             "lat": "latitude",
#             "lev": "lev",
#             "bnds": "bnds",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": ["x_bnds", "lon_bnds"],
#             "lat_bounds": ["y_bnds", "lat_bnds"],
#             "time_bounds": "time_bnds",
#             "vertex": "vertices",
#         },
#         "SAM0-UNICON": {
#             "x": ["i", "lon"],
#             "y": ["j", "lat"],
#             "lon": "longitude",
#             "lat": "latitude",
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": None,
#             "lat_bounds": None,
#             "vertex": "vertices",
#             "time_bounds": "time_bnds",
#             #         'dzt': 'thkcello',
#         },
#         "MCM-UA-1-0": {
#             "x": "longitude",
#             "y": "latitude",
#             "lon": None,
#             "lat": None,
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "lon_bnds",
#             "lat_bounds": "lat_bnds",
#             "time_bounds": "time_bnds",
#             #         'vertex': 'vertices',
#             #         'dzt': 'thkcello',
#         },
#         "IPSL-CM6A-LR": {
#             "x": ["x", "lon"],
#             "y": ["y", "lat"],
#             "lon": "nav_lon",
#             "lat": "nav_lat",
#             "lev": ["lev", "deptht", "olevel"],
#             "lev_bounds": ["lev_bounds", "deptht_bounds", "olevel_bounds"],
#             "lon_bounds": "bounds_nav_lon",
#             "lat_bounds": "bounds_nav_lat",
#             "vertex": "nvertex",
#             "bnds": "axis_nbounds",
#             "time_bounds": "time_bnds",
#             #         'dzt': 'thkcello',
#         },
#         "NorCPM1": {
#             "x": ["i", "lon"],
#             "y": ["j", "lat"],
#             "lon": "longitude",
#             "lat": "latitude",
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": None,
#             "lat_bounds": None,
#             "vertex": "vertices",
#             "time_bounds": "time_bnds",
#         },
#         "NorESM1-F": {
#             "x": ["i", "lon"],
#             "y": ["j", "lat"],
#             "lon": "longitude",
#             "lat": "latitude",
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "vertices_longitude",
#             "lat_bounds": "vertices_latitude",
#             "vertex": "vertices",
#             "time_bounds": "time_bnds",
#         },
#         "NorESM2-LM": {
#             "x": "i",
#             "y": "j",
#             "lon": "longitude",
#             "lat": "latitude",
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "vertices_longitude",
#             "lat_bounds": "vertices_latitude",
#             "vertex": "vertices",
#             "time_bounds": "time_bnds",
#         },
#         "NorESM2-MM": {
#             "x": "i",
#             "y": "j",
#             "lon": "longitude",
#             "lat": "latitude",
#             "lev": "lev",  # i leave this here because the names are the same as for the other Nor models.
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "vertices_longitude",
#             "lat_bounds": "vertices_latitude",
#             "vertex": "vertices",
#             "time_bounds": "time_bnds",
#         },
#         "MPI-ESM1-2-HR": {
#             "x": ["i", "lon"],
#             "y": ["j", "lat"],
#             "lon": "longitude",
#             "lat": "latitude",
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "vertices_longitude",
#             "lat_bounds": "vertices_latitude",
#             "vertex": "vertices",
#             "time_bounds": "time_bnds",
#         },
#         "MPI-ESM1-2-LR": {
#             "x": "i",
#             "y": "j",
#             "lon": "longitude",
#             "lat": "latitude",
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "vertices_longitude",
#             "lat_bounds": "vertices_latitude",
#             "vertex": "vertices",
#             "time_bounds": "time_bnds",
#         },
#         "MPI-ESM-1-2-HAM": {
#             "x": "i",
#             "y": "j",
#             "lon": "longitude",
#             "lat": "latitude",
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "vertices_longitude",
#             "lat_bounds": "vertices_latitude",
#             "vertex": "vertices",
#             "time_bounds": "time_bnds",
#         },
#         "CNRM-CM6-1-HR": {
#             "x": "x",
#             "y": "y",
#             "lon": "lon",
#             "lat": "lat",
#             "lev": "lev",
#             "lev_bounds": "lev_bounds",
#             "lon_bounds": "bounds_lon",
#             "lat_bounds": "bounds_lat",
#             "vertex": None,
#             "time_bounds": "time_bounds",
#         },
#         "FIO-ESM-2-0": {
#             "x": "i",
#             "y": "j",
#             "lon": "longitude",
#             "lat": "latitude",
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "vertices_longitude",
#             "lat_bounds": "vertices_latitude",
#             "vertex": "vertices",
#             "time_bounds": "time_bnds",
#         },
#         "ACCESS-ESM1-5": {
#             "x": "i",
#             "y": "j",
#             "lon": "longitude",
#             "lat": "latitude",
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "vertices_longitude",
#             "lat_bounds": "vertices_latitude",
#             "vertex": "vertices",
#             "time_bounds": "time_bnds",
#         },
#         "ACCESS-CM2": {
#             "x": "i",
#             "y": "j",
#             "lon": "longitude",
#             "lat": "latitude",
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "vertices_longitude",
#             "lat_bounds": "vertices_latitude",
#             "vertex": "vertices",
#             "time_bounds": "time_bnds",
#         },
#         "INM-CM4-8": {  # this is a guess.
#             "x": "lon",
#             "y": "lat",
#             "lon": None,
#             "lat": None,
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "lon_bnds",
#             "lat_bounds": "lat_bnds",
#             "vertex": None,
#             "time_bounds": "time_bnds",
#         },
#         "INM-CM5-0": {  # this is a guess.
#             "x": "lon",
#             "y": "lat",
#             "lon": None,
#             "lat": None,
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             "lon_bounds": "lon_bnds",
#             "lat_bounds": "lat_bnds",
#             "vertex": None,
#             "time_bounds": "time_bnds",
#         },
#         "MRI-ESM2-0": {
#             "x": "x",
#             "y": "y",
#             "lon": "longitude",
#             "lat": "latitude",
#             "lev": "lev",
#             "lev_bounds": "lev_bnds",
#             #             "lon_bounds": 'x_bnds',
#             #             "lat_bounds": 'y_bnds',
#             #             'vertex': None, # this is a mess. there is yet another convention. Will have to deal with this once I wrap xgcm into here.
#             "time_bounds": "time_bnds",
#         },
#         "CIESM": {  # this is a guess.
#             "x": "i",
#             "y": "j",
#             "lon": "longitude",
#             "lat": "latitude",
#             #             "lev": "lev", # no 3d data available as of now
#             #             "lev_bounds": "lev_bnds",
#             "lon_bounds": "vertices_longitude",
#             "lat_bounds": "vertices_latitude",
#             "vertex": "vertices",
#             "time_bounds": "time_bnds",
#         },
#         "KACE-1-0-G": {  # this is a guess.
#             "x": "lon",
#             "y": "lat",
#             "lon": "longitude",
#             "lat": "latitude",
#             #             "lev": "lev", # no 3d data available as of now
#             #             "lev_bounds": "lev_bnds",
#             #             "lon_bounds": "vertices_longitude",
#             #             "lat_bounds": "vertices_latitude",
#             #             "lon_bounds": "vertices_longitude",
#             #             "lat_bounds": "vertices_latitude",
#             "vertex": "vertices",
#             "time_bounds": "time_bnds",
#         },
#         "TaiESM1": {  # this is a guess.
#             "x": ["i", "lon"],
#             "y": ["j", "lat"],
#             "lon": "longitude",
#             "lat": "latitude",
#             #             "lev": "lev", # no 3d data available as of now
#             #             "lev_bounds": "lev_bnds",
#             #             "lon_bounds": "vertices_longitude",
#             #             "lat_bounds": "vertices_latitude",
#             #             "lon_bounds": "vertices_longitude",
#             #             "lat_bounds": "vertices_latitude",
#             "vertex": "vertices",
#             "time_bounds": "time_bnds",
#         },
#     }
#     # cast all str into lists
#     for model in dim_name_dict.keys():
#         for field in dim_name_dict[model].keys():
#             if (
#                 isinstance(dim_name_dict[model][field], str)
#                 or dim_name_dict[model][field] is None
#             ):
#                 dim_name_dict[model][field] = [dim_name_dict[model][field]]
#         #     add 'lon' and 'lat' as possible logical indicies for all models. This should take care of all regridded ocean output and all atmosphere models.
#         if "x" in dim_name_dict[model].keys():
#             if not "lon" in dim_name_dict[model]["x"]:
#                 dim_name_dict[model]["x"].append("lon")
#
#         if "y" in dim_name_dict[model].keys():
#             if not "lat" in dim_name_dict[model]["y"]:
#                 dim_name_dict[model]["y"].append("lat")
#     return dim_name_dict
#


def rename_cmip6(ds, rename_dict=None):
    """Homogenizes cmip6 dadtasets to common naming"""
    ds = ds.copy()
    source_id = ds.attrs["source_id"]

    if rename_dict is None:
        rename_dict = cmip6_renaming_dict()

    # rename variables
    if len(rename_dict) == 0:
        warnings.warn(
            "input dictionary empty.", UserWarning,
        )
    else:
        for di in rename_dict.keys():

            # make sure the input is a list
            if isinstance(rename_dict[di], str):
                rename_dict[di] = [rename_dict[di]]

            if di not in ds.variables:

                # For now just stop in the list if the dimension is already there, or one 'hit'
                # was already encountered.
                trigger = False
                for wrong in rename_dict[di]:
                    if wrong in ds.variables or wrong in ds.dims:
                        if not trigger:
                            ds = ds.rename({wrong: di})
                            trigger = True
    return ds


# def rename_cmip6(ds, rename_dict=None):
#     """rename dataset to a uniform dim/coords naming structure"""
#     modelname = ds.attrs["source_id"]
#
#
#     # if modelname not in rename_dict.keys():
#     #     msg = f"No input dictionary entry for source_id: `{ds.attrs['source_id']}`. \
#     #         Please add values to https://github.com/jbusecke/cmip6_preprocessing/blob/master/cmip6_preprocessing/preprocessing.py"
#     #     warnings.warn(msg, UserWarning)
#     #     return ds
#     # else:
#     return rename_cmip6_raw(ds, rename_dict[modelname])


def promote_empty_dims(ds):
    """Convert empty dimensions to actual coordinates"""
    ds = ds.copy()
    for di in ds.dims:
        if di not in ds.coords:
            ds = ds.assign_coords({di: ds[di]})
            # ds.coords[di] = ds[di]
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


def replace_x_y_nominal_lat_lon(ds):
    """Approximate the dimensional values of x and y with mean lat and lon at the equator"""
    ds = ds.copy()

    def maybe_fix_non_unique(data, pad=False):
        """remove duplicate values by linear interpolation
        if values are non-unique. `pad` if the last two points are the same
        pad with -90 or 90. This is only applicable to lat values"""
        if len(data) == len(np.unique(data)):
            return data
        else:
            # pad each end with the other end.
            if pad:
                if len(np.unique([data[0:2]])) < 2:
                    data[0] = -90
                if len(np.unique([data[-2:]])) < 2:
                    data[-1] = 90

            ii_range = np.arange(len(data))
            _, indicies = np.unique(data, return_index=True)
            double_idx = np.array([ii not in indicies for ii in ii_range])
            # print(f"non-unique values found at:{ii_range[double_idx]})")
            data[double_idx] = np.interp(
                ii_range[double_idx], ii_range[~double_idx], data[~double_idx]
            )
            return data

    if "x" in ds.dims and "y" in ds.dims:

        # pick the nominal lon/lat values from the eastern
        # and southern edge, and eliminate non unique values
        # these occour e.g. in "MPI-ESM1-2-HR"
        max_lat_idx = ds.lat.isel(y=-1).argmax("x").load().data
        nominal_y = maybe_fix_non_unique(ds.isel(x=max_lat_idx).lat.load().data)
        eq_idx = len(ds.y) // 2
        nominal_x = maybe_fix_non_unique(ds.isel(y=eq_idx).lon.load().data)

        ds = ds.assign_coords(x=nominal_x, y=nominal_y)
        ds = ds.sortby("x")
        ds = ds.sortby("y")

        # do one more interpolation for the x values, in case the boundary values were
        # affected
        ds = ds.assign_coords(
            x=maybe_fix_non_unique(ds.x.load().data),
            y=maybe_fix_non_unique(ds.y.load().data, pad=True),
        )

    else:
        warnings.warn(
            "No x and y found in dimensions for source_id:%s. This likely means that you forgot to rename the dataset or this is the German unstructured model"
            % ds.attrs["source_id"]
        )
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

    # remove out of bounds values found in some
    # models as missing values
    ds["lon"] = ds["lon"].where(abs(ds["lon"]) <= 1e35)
    ds["lat"] = ds["lat"].where(abs(ds["lat"]) <= 1e35)

    # only correct the actual longitude
    lon = ds["lon"].data

    # then adjust lon convention
    lon = np.where(lon < 0, 360 + lon, lon)
    lon = ds["lon"].where(ds["lon"] > 0, 360 + ds["lon"])
    ds = ds.assign_coords(lon=lon)
    return ds


def combined_preprocessing(ds):
    if "AWI" not in ds.attrs["source_id"]:
        ds = ds.copy()
        # fix naming
        ds = rename_cmip6(ds)
        # promote empty dims to actual coordinates
        ds = promote_empty_dims(ds)
        # demote coordinates from data_variables (this is somehow reversed in intake)
        ds = correct_coordinates(ds)
        # broadcast lon/lat
        ds = broadcast_lonlat(ds)
        # shift all lons to consistent 0-360
        ds = correct_lon(ds)
        # fix the units
        ds = correct_units(ds)
        # replace x,y with nominal lon,lat
        ds = replace_x_y_nominal_lat_lon(ds)

    return ds
