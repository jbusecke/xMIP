from xgcm.autogenerate import generate_grid_ds
import xarray as xr
import numpy as np
from xgcm import Grid
import pyproj
import matplotlib.pyplot as plt

# now sort all other variables in accordingly
def rename(da, da_tracer, da_u, da_v, grid_type, position, verbose=False):
    # check with which variable the lon and lat agree
    rename_dict = {
        "B": {
            "u": {
                "x": "x_" + position["X"][1],
                "y": "y_" + position["Y"][1],
                "lon": "lon_ne",
                "lat": "lat_ne",
                "vertices_latitude": "vertices_latitude_ne",
                "vertices_longitude": "vertices_longitude_ne",
            },
            "v": {
                "x": "x_" + position["X"][1],
                "y": "y_" + position["Y"][1],
                "lon": "lon_ne",
                "lat": "lat_ne",
                "vertices_latitude": "vertices_latitude_ne",
                "vertices_longitude": "vertices_longitude_ne",
            },
        },
        "C": {
            "u": {
                "x": "x_" + position["X"][1],
                "lon": "lon_e",
                "lat": "lat_e",
                "vertices_latitude": "vertices_latitude_e",
                "vertices_longitude": "vertices_longitude_e",
            },
            "v": {
                "y": "y_" + position["Y"][1],
                "lon": "lon_n",
                "lat": "lat_n",
                "vertices_latitude": "vertices_latitude_n",
                "vertices_longitude": "vertices_longitude_n",
            },
        },
    }

    loc = []

    for data, name in zip([da_tracer, da_u, da_v], ["tracer", "u", "v"]):
        if da.lon.equals(data.lon):
            loc.append(name)
    if len(loc) != 1:
        if grid_type == "B" and set(loc) == set(["u", "v"]):
            loc = ["u"]
        else:
            raise RuntimeError("somthing went wrong")

    loc = loc[0]
    if loc != "tracer":
        re_dict = {
            k: v
            for k, v in rename_dict[grid_type][loc].items()
            if k in da.variables
        }
        da = da.rename(re_dict)
    return da


def merge_variables_on_staggered_grid(
    data_dict,
    modelname,
    tracer_ref="thetao",
    u_ref="uo",
    v_ref="vo",
    plot=True,
    verbose=False,
):
    """Parses datavariables according to their staggered grid position.
    Should also work for gr variables, which are assumed to be on an A-grid."""
    
    if any([not a in data_dict.keys() for a in [tracer_ref, u_ref, v_ref]]):
        print('NON-REFERENCE MODE. This should just be used for a bunch of variables on the same grid')
        grid_type = 'A'
        
    else:
        
        # extract reference dataarrays (those need to be in there)
        tracer = data_dict[tracer_ref]
        u = data_dict[u_ref]
        v = data_dict[v_ref]

        # determine grid type
        if tracer.lon.equals(u.lon):
            grid_type = "A"
        else:
            if u.lon.equals(v.lon):
                grid_type = "B"
            else:
                grid_type = "C"

        print("Grid Type: %s detected" % grid_type)

    if grid_type == "A":
        # this should also work with interpolated and obs datasets
        # Just merge everything together
        ds_combined = xr.merge([v for v in data_dict.values()])
        ds_full = generate_grid_ds(
            ds_combined, {"X": "x", "Y": "y"}, position=("center", "right")
        )
    else:
        # now determine the axis shift
        lon = {}
        lat = {}

        lon["tracer"] = tracer.lon
        lon["u"] = u.lon
        lon["v"] = v.lon

        lat["tracer"] = tracer.lat
        lat["u"] = u.lat
        lat["v"] = v.lat

        # vizualize the position
        if plot:
            ref_idx = 3
            plt.figure()
            for vi, var in enumerate(["tracer", "u", "v"]):
                plt.plot(
                    lon[var].isel(x=ref_idx, y=ref_idx),
                    lat[var].isel(x=ref_idx, y=ref_idx),
                    marker="*",
                    markersize=25,
                )
                plt.text(
                    lon[var].isel(x=ref_idx, y=ref_idx),
                    lat[var].isel(x=ref_idx, y=ref_idx),
                    var,
                    ha="center",
                    va="center",
                )
            plt.title("Staggered Grid visualizaton")
            plt.show()

        if verbose:
            print("Determine grid shift")

        lon_diff = lon["tracer"] - lon["u"]
        # elinate large values due to boundry disc
        lon_diff = lon_diff.where(abs(lon_diff) < 180)

        lat_diff = lat["tracer"] - lat["v"]

        position = dict()
        for axis, diff in zip(["X", "Y"], [lon_diff, lat_diff]):
            if np.sign(diff.mean().load()) < 0:
                position[axis] = ("center", "left")
            else:
                position[axis] = ("center", "right")

        if verbose:
            print("Regenerate grid")

        ds_full = generate_grid_ds(tracer, {"X": "x", "Y": "y"}, position=position)

        if verbose:
            print("Renaming and Merging")
        for k, da in data_dict.items():
            #             print('Merging: %s' %k)
            da_renamed = rename(da, tracer, u, v, grid_type, position, verbose=verbose)
            # parse all the coordinate values from the reconstructed
            # dataset to the new dataarray
            # or the merging will create intermediate steps
            for co in da_renamed.coords:
                if co in ds_full.coords:
                    da_renamed.coords[co] = ds_full.coords[co]

            if verbose:
                print("Merge")

            # place all new data_variables into the dataset
            for dvar in da_renamed.data_vars:
                if dvar not in ds_full.data_vars:
                    ds_full[dvar] = da_renamed[dvar]
    #             ds_full = xr.merge([ds_full, da_renamed])
    return ds_full

def distance(lon0, lat0, lon1, lat1):
    geod = pyproj.Geod(ellps="WGS84")
    _, _, distance = geod.inv(lon0, lat0, lon1, lat1)
    return distance


def recreate_metrics(ds, grid):
    ds = ds.copy()

    # is the vel point on left or right?
    axis_vel_pos = {
        axis: list(set(grid.axes[axis].coords.keys()) - set(["center"]))[0]
        for axis in ["X", "Y"]
    }

    # infer dx at eastern bound from tracer points
    lon0, lon1 = grid.axes["X"]._get_neighbor_data_pairs(
        ds.lon.load(), axis_vel_pos["X"]
    )
    lat0 = lat1 = ds.lat.load().data
    dx = distance(lon0, lat0, lon1, lat1)
    ds.coords["dxe"] = xr.DataArray(dx, coords=grid.interp(ds.lon, "X").coords)

    # infer dy at northern bound from tracer points
    lat0, lat1 = grid.axes["Y"]._get_neighbor_data_pairs(
        ds.lat.load(), axis_vel_pos["Y"], boundary="extrapolate"
    )

    lon0 = lon1 = ds.lon.load().data
    dy = distance(lon0, lat0, lon1, lat1)
    ds.coords["dyn"] = xr.DataArray(
        dy, coords=grid.interp(ds.lat, "Y", boundary="extrapolate").coords
    )

    # now simply interpolate all the other metrics
    ds.coords["dxt"] = grid.interp(ds.coords["dxe"], "X")
    ds.coords["dxne"] = grid.interp(ds.coords["dxe"], "Y", boundary="extrapolate")
    ds.coords["dxn"] = grid.interp(ds.coords["dxt"], "Y", boundary="extrapolate")

    ds.coords["dyt"] = grid.interp(ds.coords["dyn"], "Y", boundary="extrapolate")
    ds.coords["dyne"] = grid.interp(ds.coords["dyn"], "X")
    ds.coords["dye"] = grid.interp(ds.coords["dyt"], "X")

    ds.coords["area_t"] = ds.coords["dxt"] * ds.coords["dyt"]
    ds.coords["area_e"] = ds.coords["dxe"] * ds.coords["dye"]
    ds.coords["area_ne"] = ds.coords["dxne"] * ds.coords["dyne"]
    ds.coords["area_n"] = ds.coords["dxn"] * ds.coords["dyn"]

    # should i return the coords to dask?
    return ds


# TODO: check if all these hardcoded lines work for each CMIP6 model...
def recreate_grid_simple(ds, lon_name="lon", lat_name="lat"):
    ds_full = generate_grid_ds(ds, {"X": "x", "Y": "y"}, position=("center", "right"))
    grid = Grid(ds_full, periodic=["X"])
    ds_full = recreate_metrics(ds, grid)
    return ds_full
