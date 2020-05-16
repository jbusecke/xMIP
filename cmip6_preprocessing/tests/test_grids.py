import numpy as np
import xarray as xr
from cmip6_preprocessing.grids import distance_deg, parse_bounds_vertex


def test_distance_deg():
    assert distance_deg(0, 0, 0, 1) == (np.array(0), np.array(1))
    assert distance_deg(0, 0, 1, 0) == (np.array(1), np.array(0))


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
