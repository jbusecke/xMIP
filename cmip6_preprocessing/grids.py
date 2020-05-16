from xgcm.autogenerate import generate_grid_ds
from xgcm import Grid
import warnings
import numpy as np
import xarray as xr
import yaml


def parse_bounds_vertex(da, dim="bnds", position=[0, 1]):
    return tuple([da.isel({dim: i}).load().data for i in position])


def interp_vertex_to_bounds(da, orientation):
    """
    little helper function to average 4 vertex points into two bound points.
    Helpful to recreate e.g. the latitude at the `lon_bounds` points.
    """
    if orientation == "x":
        datasets = [
            da.isel(vertex=[0, 1]).mean("vertex"),
            da.isel(vertex=[3, 2]).mean("vertex"),
        ]
    elif orientation == "y":
        datasets = [
            da.isel(vertex=[0, 3]).mean("vertex"),
            da.isel(vertex=[1, 2]).mean("vertex"),
        ]

    return xr.concat(datasets, dim="bnds")


def distance_deg(lon0, lat0, lon1, lat1):
    delta_lon = lon1 - lon0
    delta_lat = lat1 - lat0
    # very small differences can end up negative, so zero them out based on a simple
    # criterion
    # this should work for CMIP6 (no 1/1 deg models) but should be based on actual grid
    # info in the future
    small_crit = 1 / 10
    delta_lon = np.where(
        abs(delta_lon) < small_crit, 0.0, delta_lon
    )  # , np.nan, delta_lon)
    delta_lat = np.where(
        abs(delta_lat) < small_crit, 0.0, delta_lat
    )  # , np.nan, delta_lat)

    #     # some bounds are wrapped aroud the lon discontinuty.
    delta_lon = np.where(delta_lon < (-small_crit * 2), 360 + delta_lon, delta_lon)  #
    delta_lon = np.where(
        delta_lon > (360 + small_crit * 2), -360 + delta_lon, delta_lon
    )

    return delta_lon, delta_lat


def distance(lon0, lat0, lon1, lat1):
    Re = 6.378e6
    delta_lon, delta_lat = distance_deg(lon0, lat0, lon1, lat1)
    dy = Re * (np.pi * delta_lat / 180)
    dx = Re * (np.pi * delta_lon / 180) * np.cos(np.pi * lat0 / 180)
    return np.sqrt(dx ** 2 + dy ** 2)


def recreate_metrics(ds, grid):
    """Recreate a full set of horizontal distance metrics by using spherical geometry"""
    ds = ds.copy()

    # Since this puts out numpy arrays, the arrays need to be transposed correctly
    transpose_dims = ["y", "x"]
    dims = [di for di in ds.dims if di not in transpose_dims]

    ds = ds.transpose(*tuple(transpose_dims + dims))

    # is the vel point on left or right?
    axis_vel_pos = {
        axis: list(set(grid.axes[axis].coords.keys()) - set(["center"]))[0]
        for axis in ["X", "Y"]
    }
    # determine the appropriate vertex position for the north/south and east/west edge,
    # based on the grid config
    if axis_vel_pos["Y"] in ["left"]:
        ns_vertex_idx = [0, 3]
    elif axis_vel_pos["Y"] in ["right"]:
        ns_vertex_idx = [1, 2]

    if axis_vel_pos["Y"] in ["left"]:
        ew_vertex_idx = [0, 1]
    elif axis_vel_pos["Y"] in ["right"]:
        ew_vertex_idx = [3, 2]

    # infer dx at tracer points
    if "lon_bounds" in ds.coords and "lat_verticies" in ds.coords:
        lon0, lon1 = parse_bounds_vertex(ds["lon_bounds"])
        lat0, lat1 = parse_bounds_vertex(
            interp_vertex_to_bounds(ds["lat_verticies"], "x")
        )
        dist = distance(lon0, lat0, lon1, lat1)
        ds.coords["dx_t"] = xr.DataArray(dist, coords=ds.lon.coords)

    # infer dy at tracer points
    if "lat_bounds" in ds.coords and "lon_verticies" in ds.coords:
        lat0, lat1 = parse_bounds_vertex(ds["lat_bounds"])
        lon0, lon1 = parse_bounds_vertex(
            interp_vertex_to_bounds(ds["lon_verticies"], "y")
        )
        dist = distance(lon0, lat0, lon1, lat1)
        ds.coords["dy_t"] = xr.DataArray(dist, coords=ds.lon.coords)

    if "lon_verticies" in ds.coords and "lat_verticies" in ds.coords:

        # infer dx at the north/south face
        lon0, lon1 = parse_bounds_vertex(
            ds["lon_verticies"], dim="vertex", position=ns_vertex_idx
        )
        lat0, lat1 = parse_bounds_vertex(
            ds["lat_verticies"], dim="vertex", position=ns_vertex_idx
        )
        dist = distance(lon0, lat0, lon1, lat1)
        ds.coords["dx_gy"] = xr.DataArray(
            dist, coords=grid.interp(ds.lon, "Y", boundary="extrapolate").coords
        )

        # infer dy at the east/west face
        lon0, lon1 = parse_bounds_vertex(
            ds["lon_verticies"], dim="vertex", position=ew_vertex_idx
        )
        lat0, lat1 = parse_bounds_vertex(
            ds["lat_verticies"], dim="vertex", position=ew_vertex_idx
        )
        dist = distance(lon0, lat0, lon1, lat1)
        ds.coords["dy_gx"] = xr.DataArray(
            dist, coords=grid.interp(ds.lon, "X", boundary="extrapolate").coords
        )

    # for the distances that dont line up with the cell boundaries we need some different logic

    # infer dx at eastern/western bound from tracer points
    lon0, lon1 = grid.axes["X"]._get_neighbor_data_pairs(
        ds.lon.load(), axis_vel_pos["X"]
    )
    lat0, lat1 = grid.axes["X"]._get_neighbor_data_pairs(
        ds.lat.load(), axis_vel_pos["X"]
    )
    dx = distance(lon0, lat0, lon1, lat1)
    ds.coords["dx_gx"] = xr.DataArray(dx, coords=grid.interp(ds.lon, "X").coords)

    # infer dy at northern bound from tracer points
    lat0, lat1 = grid.axes["Y"]._get_neighbor_data_pairs(
        ds.lat.load(), axis_vel_pos["Y"], boundary="extrapolate"
    )
    lon0, lon1 = grid.axes["Y"]._get_neighbor_data_pairs(
        ds.lon.load(), axis_vel_pos["Y"], boundary="extrapolate"
    )
    dy = distance(lon0, lat0, lon1, lat1)
    ds.coords["dy_gy"] = xr.DataArray(
        dy, coords=grid.interp(ds.lat, "Y", boundary="extrapolate").coords
    )

    metrics_dict = {
        "X": [co for co in ["dx_t", "dx_gy", "dx_gx"] if co in ds.coords],
        "Y": [co for co in ["dy_t", "dy_gy", "dy_gx"] if co in ds.coords],
    }

    # and now the last one, the metrics centered on the corner point corner point
    # these still need to be done
    return ds, metrics_dict


def detect_shift(ds_base, ds, axis):
    """ detects the shift of `ds` relative to `ds` on logical grid axes, using
    lon and lat positions."""
    ds_base = ds_base.copy()
    ds = ds.copy()
    axis = axis.lower()
    axis_coords = {"x": "lon", "y": "lat"}

    # check the shift only for one point, somewhat in the center to avoid the
    # distorted polar regions
    check_point = {"x": len(ds_base.x) // 2, "y": len(ds_base.y) // 2}
    check_point_diff = {k: [v, v + 1] for k, v in check_point.items()}

    shift = (
        ds.isel(**check_point)[axis_coords[axis]].load().data
        - ds_base.isel(**check_point)[axis_coords[axis]].load().data
    )
    diff = ds[axis].isel({axis: check_point_diff[axis]}).diff(axis).data.tolist()[0]
    threshold = 0.1
    # the fraction of full cell distance, that a point has to be shifted in order to
    # be recognized.
    # This avoids detection of shifts for very small differences that sometimes happen
    # if the coordinates were written e.g. by different modulel of a model

    axis_shift = "center"

    if shift > (diff * threshold):
        axis_shift = "right"
    elif shift < -(diff * threshold):
        axis_shift = "left"
    return axis_shift


def create_full_grid(base_ds, grid_dict=None):
    """Generate a full xgcm-compatible dataset from a reference datasets `base_ds`.
    This dataset should be representing a tracer fields, e.g. the cell center."""

    # load dict with grid shift info for each axis
    if grid_dict is None:
        yaml_path = "staggered_grid_config.yaml"
        ff = open(yaml_path, "r")
        grid_dict = yaml.load(ff)
        ff.close()

    source_id = base_ds.attrs["source_id"]
    grid_label = base_ds.attrs["grid_label"]

    # if source_id not in dict, and grid label is gn, warn and ask to submit an issue
    if source_id not in grid_dict.keys() and "gn" in grid_label:
        warnings.warn(
            f"Could not find the model ({source_id}) in `grid_dict`, returning `None`. Please submit an issue to github: https://github.com/jbusecke/cmip6_preprocessing/issues"
        )
        return None

    # if source_id in dict, but its gr, just print that default lefts are used for regridded data
    if "gr" in grid_label:
        warnings.warn(
            "For grid label `*gr*` the axis shift defaults to left for X and Y"
        )
        axis_shift = {"X": "left", "Y": "left"}
    else:
        axis_shift = grid_dict[source_id]["axis_shift"]

    position = {k: ("center", axis_shift[k]) for k in axis_shift.keys()}

    axis_dict = {"X": "x", "Y": "y"}

    ds_grid = generate_grid_ds(
        base_ds, axis_dict, position=position, boundary_discontinuity={"X": 360}
    )

    return ds_grid


def combine_staggered_grid(ds_base, other_ds=None, recalculate_metrics=False, **kwargs):
    """combine a reference datasets `base_ds` with a list of other datasets
    `other_ds` to a full xgcm-compatible staggered grid datasets. This can be a
    list of variables, regardless of their grid position, which will be
    automatically detected. Coordinates and attrs of these added datasets will be lost.
    `recalculate_metrics` enables the reconstruction of grid metrics usign simple
    spherical geometry.

    !!! Check your results carefully when using reconstructed values,
    these might differe substantially if the grid geometry is complicated.
    """
    ds_base = ds_base.copy()
    if isinstance(other_ds, str):
        other_ds = [other_ds]

    ds_g = create_full_grid(ds_base)

    if ds_g is None:
        warnings.warn("Staggered Grid creation failed. Returning `None`")
        return None, None

    # save attrs out for later (something during alignment destroys them)
    dim_attrs_dict = {}
    for di in ds_g.dims:
        dim_attrs_dict[di] = ds_g[di].attrs

    # TODO: metrics and interpolation of metrics if they are parsed

    # parse other variables
    if other_ds is not None:
        for ds_new in other_ds:
            ds_new = ds_new.copy()
            # strip everything but the variable_id (perhaps I would want to
            # loosen this in the future)
            ds_new = ds_new[ds_new.attrs["variable_id"]]

            if not all(
                [
                    len(ds_new[di]) == len(ds_g[di])
                    for di in ds_new.dims
                    if di not in ["member_id", "time"]
                ]
            ):
                warnings.warn(
                    f"Could not parse `{ds_new.name}`, due to a size mismatch. If this is the MRI model, the grid convention is currently not supported."
                )
            else:
                # detect shift and rename accordingly
                rename_dict = {}
                for axis in ["X", "Y"]:
                    shift = detect_shift(ds_base, ds_new, axis)

                    if shift != "center":
                        rename_dict[axis.lower()] = axis.lower() + "_" + shift
                ds_new = ds_new.rename(rename_dict)
                ds_new = ds_new.reset_coords(drop=True)
                # TODO: This needs to be coded more generally, for now hardcode x and y
                force_align_dims = [di for di in ds_new.dims if "x" in di or "y" in di]
                _, ds_new = xr.align(
                    ds_g.copy(),
                    ds_new,
                    join="override",
                    exclude=[di for di in ds_new.dims if di not in force_align_dims],
                )
                additional_dims = [di for di in ds_new.dims if di not in ds_g.dims]
                if len(additional_dims) > 0:
                    raise RuntimeError(
                        f"While trying to parse `{ds_new.name}`, detected dims that are not in the base dataset:[{additional_dims}]"
                    )
                ds_g[ds_new.name] = ds_new

    # Restore dims attrs from the beginning
    for di in ds_g.dims:
        ds_g.coords[di].attrs.update(dim_attrs_dict[di])

    grid_kwargs = {"periodic": ["X"]}
    grid_kwargs.update(kwargs)
    grid = Grid(ds_g, grid_kwargs)

    # if activated calculate metrics
    if recalculate_metrics:
        grid_kwargs.pop(
            "metrics", None
        )  # remove any passed metrics when recalculating them
        # I might be able to refine this more to e.g. allow axes that are not recreated.

        ds_g, metrics_dict = recreate_metrics(ds_g, grid)
        # this might fail in circumstances, where the
        grid_kwargs["metrics"] = metrics_dict
        grid = Grid(ds_g, **grid_kwargs)
    return grid, ds_g
