import numpy as np
import pytest
import xarray as xr
from xgcm import Grid
from xgcm.autogenerate import generate_grid_ds
from cmip6_preprocessing.grids import (
    distance,
    distance_deg,
    interp_vertex_to_bounds,
    parse_bounds_vertex,
    recreate_metrics,
    interp_vertex_to_bounds,
)


def test_parse_bounds_vertex():
    lon_b = xr.DataArray(np.array([0, 1, 2, 3]), dims=["vertex"])
    lat_b = xr.DataArray(np.array([10, 11, 12, 13]), dims=["vertex"])

    data = np.random.rand(4)

    da = xr.DataArray(
        data, dims=["vertex"], coords={"lon_verticies": lon_b, "lat_verticies": lat_b}
    )
    test = parse_bounds_vertex(da, "vertex", position=[0, 3])
    print(test)
    expected = (da.isel(vertex=0).load().data, da.isel(vertex=3).load().data)
    print(expected)
    assert test == expected


def test_interp_vertex_to_bounds():
    da = xr.DataArray(np.arange(4), dims=["vertex"])
    # test interp on the y axis
    expected = xr.DataArray(np.array([1.5, 1.5]), dims=["bnds"])
    xr.testing.assert_equal(interp_vertex_to_bounds(da, "y"), expected)
    # test interp on the x axis
    expected = xr.DataArray(np.array([0.5, 2.5]), dims=["bnds"])
    xr.testing.assert_equal(interp_vertex_to_bounds(da, "x"), expected)


def test_distance_deg():
    lon0, lat0, lon1, lat1 = 120, 30, 121, 31
    delta_lon, delta_lat = distance_deg(lon0, lat0, lon1, lat1)
    assert delta_lon == 1.0
    assert delta_lat == 1.0

    lon0, lat0, lon1, lat1 = 360, 30, 1, 31
    delta_lon, delta_lat = distance_deg(lon0, lat0, lon1, lat1)
    assert delta_lon == 1.0
    assert delta_lat == 1.0

    lon0, lat0, lon1, lat1 = 300, 30, 301, 30.09
    delta_lon, delta_lat = distance_deg(lon0, lat0, lon1, lat1)
    assert delta_lon == 1.0
    assert delta_lat == 0.0


@pytest.mark.parametrize("lon", [0, 90, 120])
@pytest.mark.parametrize("lat", [0, 10, 45])
def test_distance(lon, lat):
    Re = 6.378e6
    # test straight lat line
    lon0, lat0, lon1, lat1 = lon, lat, lon, lat + 1
    dist = distance(lon0, lat0, lon1, lat1)
    np.testing.assert_allclose(dist, Re * (np.pi * 1.0 / 180))

    # test straight lon line
    lon0, lat0, lon1, lat1 = lon, lat, lon + 1, lat
    dist = distance(lon0, lat0, lon1, lat1)
    np.testing.assert_allclose(
        dist, Re * (np.pi * 1.0 / 180) * np.cos(np.pi * lat0 / 180)
    )


def _add_small_rand(da):
    return da + (np.random.rand(*da.shape) * 0.05)


# TODO: inner and outer (needs to be implemented in xgcm autogenerate first)
@pytest.mark.parametrize("xshift", ["left", "right"])
@pytest.mark.parametrize("yshift", ["left", "right"])
def test_recreate_metrics(xshift, yshift):

    # reconstruct all the metrics by hand and compare to inferred output

    # * For now this is a regular lon lat grid. Might need to add some tests for more complex grids.
    # Then again. This will not do a great job for those....

    # create test dataset
    xt = np.arange(4) + 1

    yt = np.arange(5) + 1

    zt = np.arange(6) + 1

    lev = xr.DataArray(zt, coords=[("lev", zt)])

    # Need to add a tracer here to get the tracer dimsuffix
    tr = xr.DataArray(
        np.random.rand(len(xt), len(yt), len(zt)),
        coords=[("x", xt), ("y", yt), ("lev", zt)],
    )

    lon_raw = xr.DataArray(xt, coords=[("x", xt)])
    lat_raw = xr.DataArray(yt, coords=[("y", yt)])
    lon = lon_raw * xr.ones_like(lat_raw)
    lat = xr.ones_like(lon_raw) * lat_raw

    lon_bounds_e = lon + 0.5
    lon_bounds_w = lon - 0.5 + (np.random.rand(*lon.shape) * 0.05)
    lat_bounds_n = lat + 0.5 + (np.random.rand(*lon.shape) * 0.05)
    lat_bounds_s = lat - 0.5 + (np.random.rand(*lon.shape) * 0.05)

    lon_bounds = xr.concat(
        [_add_small_rand(lon_bounds_w), _add_small_rand(lon_bounds_w)], dim="bnds"
    )
    lat_bounds = xr.concat(
        [_add_small_rand(lat_bounds_s), _add_small_rand(lat_bounds_n)], dim="bnds"
    )

    lev_bounds = xr.concat(
        [_add_small_rand(lev - 0.5), _add_small_rand(lev + 0.5)], dim="bnds"
    )

    lon_verticies = xr.concat(
        [
            _add_small_rand(lon_bounds_e),
            _add_small_rand(lon_bounds_e),
            _add_small_rand(lon_bounds_w),
            _add_small_rand(lon_bounds_w),
        ],
        dim="vertex",
    )
    lat_verticies = xr.concat(
        [
            _add_small_rand(lat_bounds_s),
            _add_small_rand(lat_bounds_n),
            _add_small_rand(lat_bounds_n),
            _add_small_rand(lat_bounds_s),
        ],
        dim="vertex",
    )

    ds = xr.Dataset({"tracer": tr})
    ds = ds.assign_coords(
        lon=lon,
        lat=lat,
        lon_bounds=lon_bounds,
        lat_bounds=lat_bounds,
        lon_verticies=lon_verticies,
        lat_verticies=lat_verticies,
        lev_bounds=lev_bounds,
    )

    ds_full = generate_grid_ds(
        ds,
        {"X": "x", "Y": "y", "Z": "lev"},
        position={"X": ("center", xshift), "Y": ("center", yshift)},
    )

    grid = Grid(ds_full)
    print(grid)

    ds_metrics, metrics_dict = recreate_metrics(ds_full, grid)

    # compute the more complex metrics (I could wrap this into a function I guess?)
    lon0, lon1 = grid.axes["X"]._get_neighbor_data_pairs(ds.lon.load(), xshift)
    lat0, lat1 = grid.axes["X"]._get_neighbor_data_pairs(ds.lat.load(), xshift)
    dx_gx_expected = distance(lon0, lat0, lon1, lat1)

    lon0, lon1 = grid.axes["Y"]._get_neighbor_data_pairs(ds.lon.load(), yshift)
    lat0, lat1 = grid.axes["Y"]._get_neighbor_data_pairs(ds.lat.load(), yshift)
    dy_gy_expected = distance(lon0, lat0, lon1, lat1)

    # corner metrics
    # dx
    if yshift == "left":
        # dx
        lon0, lon1 = grid.axes["X"]._get_neighbor_data_pairs(
            interp_vertex_to_bounds(ds_metrics.lon_verticies, "y").isel(bnds=0), xshift,
        )
        lat0, lat1 = grid.axes["X"]._get_neighbor_data_pairs(
            ds_metrics.lat_bounds.isel(bnds=0), xshift
        )
    elif yshift == "right":
        lon0, lon1 = grid.axes["X"]._get_neighbor_data_pairs(
            interp_vertex_to_bounds(ds_metrics.lon_verticies, "y").isel(bnds=1), xshift,
        )
        lat0, lat1 = grid.axes["X"]._get_neighbor_data_pairs(
            ds_metrics.lat_bounds.isel(bnds=1), xshift
        )
    dx_gxgy_expected = distance(lon0, lat0, lon1, lat1)

    # dy
    if xshift == "left":
        # dx
        lat0, lat1 = grid.axes["Y"]._get_neighbor_data_pairs(
            interp_vertex_to_bounds(ds_metrics.lat_verticies, "x").isel(bnds=0), yshift,
        )
        lon0, lon1 = grid.axes["Y"]._get_neighbor_data_pairs(
            ds_metrics.lon_bounds.isel(bnds=0), yshift
        )
    elif xshift == "right":
        lat0, lat1 = grid.axes["Y"]._get_neighbor_data_pairs(
            interp_vertex_to_bounds(ds_metrics.lat_verticies, "x").isel(bnds=1), yshift,
        )
        lon0, lon1 = grid.axes["Y"]._get_neighbor_data_pairs(
            ds_metrics.lon_bounds.isel(bnds=1), yshift
        )
    dy_gxgy_expected = distance(lon0, lat0, lon1, lat1)

    if xshift == "left":
        vertex_points = [0, 1]
    else:
        vertex_points = [2, 3]
    lon0, lon1 = (
        ds_metrics.lon_verticies.isel(vertex=vertex_points[0]),
        ds_metrics.lon_verticies.isel(vertex=vertex_points[1]),
    )
    lat0, lat1 = (
        ds_metrics.lat_verticies.isel(vertex=vertex_points[0]),
        ds_metrics.lat_verticies.isel(vertex=vertex_points[1]),
    )
    dy_gx_expected = distance(lon0, lat0, lon1, lat1)

    if yshift == "left":
        vertex_points = [0, 3]
    else:
        vertex_points = [1, 2]
    lon0, lon1 = (
        ds_metrics.lon_verticies.isel(vertex=vertex_points[0]),
        ds_metrics.lon_verticies.isel(vertex=vertex_points[1]),
    )
    lat0, lat1 = (
        ds_metrics.lat_verticies.isel(vertex=vertex_points[0]),
        ds_metrics.lat_verticies.isel(vertex=vertex_points[1]),
    )
    dx_gy_expected = distance(lon0, lat0, lon1, lat1)

    for var, expected in [
        ("dz_t", lev_bounds.diff("bnds").squeeze().data),
        (
            "dx_t",
            distance(
                ds_metrics.lon_bounds.isel(bnds=0).data,
                ds_metrics.lat.data,
                ds_metrics.lon_bounds.isel(bnds=1).data,
                ds_metrics.lat.data,
            ),
        ),
        (
            "dy_t",
            distance(
                ds_metrics.lon.data,
                ds_metrics.lat_bounds.isel(bnds=0).data,
                ds_metrics.lon.data,
                ds_metrics.lat_bounds.isel(bnds=1).data,
            ),
        ),
        ("dx_gx", dx_gx_expected),
        ("dy_gy", dy_gy_expected),
        ("dy_gx", dy_gx_expected),
        ("dx_gy", dx_gy_expected),
        ("dy_gxgy", dy_gxgy_expected),
        ("dx_gxgy", dx_gxgy_expected),
    ]:
        print(var)
        control = ds_metrics[var].data
        if expected.shape != control.shape:
            control = control.T
        np.testing.assert_allclose(control, expected)
