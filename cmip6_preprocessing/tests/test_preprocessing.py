import pytest
import intake
import pandas as pd
import numpy as np
import xarray as xr
from cmip6_preprocessing.preprocessing import (
    cmip6_renaming_dict,
    rename_cmip6,
    broadcast_lonlat,
    promote_empty_dims,
    replace_x_y_nominal_lat_lon,
    correct_coordinates,
    correct_lon,
    correct_units,
)

# get all available ocean models from the cloud.
url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.csv"
df = pd.read_csv(url)
df_ocean = df[(df.table_id == "Omon") + (df.table_id == "Oyr")]
ocean_models = df_ocean.source_id.unique()

# TODO: Need to adapt atmos only models
all_models = ocean_models


def create_test_ds(xname, yname, zname, xlen, ylen, zlen):
    x = np.linspace(0, 359, 10)
    y = np.linspace(-90, 89, 5)
    z = np.linspace(0, 5000, 6)
    data = np.random.rand(len(x), len(y), len(z))
    ds = xr.DataArray(data, coords=[(xname, x), (yname, y), (zname, z)]).to_dataset(
        name="test"
    )
    ds.attrs["source_id"] = "test_id"
    # if x and y are not lon and lat, add lon and lat to make sure there are no conflicts
    lon = ds[xname] * xr.ones_like(ds[yname])
    lat = xr.ones_like(ds[xname]) * ds[yname]
    if xname != "lon" and yname != "lat":
        ds = ds.assign_coords(lon=lon, lat=lat)
    else:
        ds = ds.assign_coords(longitude=lon, latitude=lat)
    return ds


@pytest.mark.parametrize("xname", ["i", "x", "lon"])
@pytest.mark.parametrize("yname", ["j", "y", "lat"])
@pytest.mark.parametrize("zname", ["lev", "olev", "olevel", "deptht", "deptht"])
@pytest.mark.parametrize("missing_dim", [None, "x", "y", "z"])
def test_rename_cmip6(xname, yname, zname, missing_dim):
    xlen, ylen, zlen = (10, 5, 6)
    ds = create_test_ds(xname, yname, zname, xlen, ylen, zlen)

    if missing_dim == "x":
        ds = ds.drop_dims(xname)
    elif missing_dim == "y":
        ds = ds.drop_dims(yname)
    elif missing_dim == "z":
        ds = ds.drop_dims(zname)

    ds_renamed = rename_cmip6(ds, cmip6_renaming_dict())
    assert set(ds_renamed.dims).issubset(set(["x", "y", "lev"]))
    if missing_dim not in ["x", "y"]:
        assert (set(ds_renamed.coords) - set(ds_renamed.dims)) == set(["lon", "lat"])
    if not missing_dim == "x":
        assert xlen == len(ds_renamed.x)
    if not missing_dim == "y":
        assert ylen == len(ds_renamed.y)
    if not missing_dim == "z":
        assert zlen == len(ds_renamed.lev)


def test_broadcast_lonlat():
    x = np.arange(-180, 179, 5)
    y = np.arange(-90, 90, 6)
    data = np.random.rand(len(x), len(y))
    ds = xr.DataArray(data, dims=["x", "y"], coords={"x": x, "y": y}).to_dataset(
        name="test"
    )
    expected = ds.copy()
    expected.coords["lon"] = ds.x * xr.ones_like(ds.y)
    expected.coords["lat"] = xr.ones_like(ds.x) * ds.y

    ds_test = broadcast_lonlat(ds)
    xr.testing.assert_identical(expected, ds_test)


def test_promote_empty_dims():
    xlen, ylen, zlen = (10, 5, 6)
    ds = create_test_ds("x", "y", "z", xlen, ylen, zlen)
    ds = ds.drop(["x", "y", "z"])
    ds_promoted = promote_empty_dims(ds)
    assert set(["x", "y", "z"]).issubset(set(ds_promoted.coords))


@pytest.mark.parametrize("dask", [True, False])
def test_replace_x_y_nominal_lat_lon(dask):
    x = np.linspace(0, 720, 10)
    y = np.linspace(-200, 140, 5)
    lon = xr.DataArray(np.linspace(0, 360, len(x)), coords=[("x", x)])
    lat = xr.DataArray(np.linspace(-90, 90, len(y)), coords=[("y", y)])
    llon = lon * xr.ones_like(lat)
    llat = xr.ones_like(lon) * lat

    data = np.random.rand(len(x), len(y))
    ds = xr.DataArray(data, coords=[("x", x), ("y", y)]).to_dataset(name="data")
    ds.coords["lon"] = llon
    ds.coords["lat"] = llat

    if dask:
        ds = ds.chunk({"x": -1, "y": -1})
        ds.coords["lon"] = ds.coords["lon"].chunk({"x": -1, "y": -1})
        ds.coords["lat"] = ds.coords["lat"].chunk({"x": -1, "y": -1})

    replaced_ds = replace_x_y_nominal_lat_lon(ds)
    print(replaced_ds.x.data)
    print(lon.data)
    np.testing.assert_allclose(replaced_ds.x, lon)
    np.testing.assert_allclose(replaced_ds.y, lat)
    assert all(replaced_ds.x.diff("x") > 0)
    assert all(replaced_ds.y.diff("y") > 0)
    assert len(replaced_ds.lon.shape) == 2
    assert len(replaced_ds.lat.shape) == 2
    assert set(replaced_ds.lon.dims) == set(["x", "y"])
    assert set(replaced_ds.lat.dims) == set(["x", "y"])

    # test a dataset that would result in duplicates with current method
    x = np.linspace(0, 720, 4)
    y = np.linspace(-200, 140, 3)
    llon = xr.DataArray(
        np.array([[0, 50, 100, 150], [0, 50, 100, 150], [0, 50, 100, 150]]),
        coords=[("y", y), ("x", x)],
    )
    llat = xr.DataArray(
        np.array([[0, 0, 10, 0], [10, 0, 0, 0], [20, 20, 20, 20]]),
        coords=[("y", y), ("x", x)],
    )
    data = np.random.rand(len(x), len(y))
    ds = xr.DataArray(data, coords=[("x", x), ("y", y)]).to_dataset(name="data")
    ds.coords["lon"] = llon
    ds.coords["lat"] = llat

    if dask:
        ds = ds.chunk({"x": -1, "y": -1})
        ds.coords["lon"] = ds.coords["lon"].chunk({"x": -1, "y": -1})
        ds.coords["lat"] = ds.coords["lat"].chunk({"x": -1, "y": -1})

    replaced_ds = replace_x_y_nominal_lat_lon(ds)
    assert len(replaced_ds.y) == len(np.unique(replaced_ds.y))
    assert len(replaced_ds.x) == len(np.unique(replaced_ds.x))
    # make sure values are sorted in ascending order
    assert all(replaced_ds.x.diff("x") > 0)
    assert all(replaced_ds.y.diff("y") > 0)
    assert len(replaced_ds.lon.shape) == 2
    assert len(replaced_ds.lat.shape) == 2
    assert set(replaced_ds.lon.dims) == set(["x", "y"])
    assert set(replaced_ds.lat.dims) == set(["x", "y"])


@pytest.mark.parametrize(
    "coord",
    [
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
    ],
)
def test_correct_coordinates(coord):
    xlen, ylen, zlen = (10, 5, 6)
    ds = create_test_ds("xx", "yy", "zz", xlen, ylen, zlen)
    # set a new variable
    ds = ds.assign({coord: ds.test})

    ds_corrected = correct_coordinates(ds)
    assert coord in list(ds_corrected.coords)


@pytest.mark.parametrize("missing_values", [False, 1e36, -1e36])
@pytest.mark.parametrize("shift", [-70, -180, -360,])  # cant handle positive shifts yet
def test_correct_lon(missing_values, shift):
    xlen, ylen, zlen = (40, 20, 6)
    ds = create_test_ds("x", "y", "lev", xlen, ylen, zlen)
    ds = ds.assign_coords(x=ds.x.data + shift)
    lon = ds["lon"].reset_coords(drop=True)
    ds = ds.assign_coords(lon=lon + shift)
    if missing_values:
        # CESM-FV has some super high missing values. Test removal
        lon = ds["lon"].load().data
        lon[10:20, 10:20] = missing_values
        ds["lon"].data = lon
    ds_lon_corrected = correct_lon(ds)
    print(ds_lon_corrected.lon.load().data)
    assert ds_lon_corrected.lon.min() >= 0
    assert ds_lon_corrected.lon.max() <= 360


def test_correct_units():
    lev = np.arange(0, 200)
    data = np.random.rand(*lev.shape)
    ds = xr.DataArray(data, dims=["lev"], coords={"lev": lev}).to_dataset(name="test")
    ds.attrs["source_id"] = "something"
    ds.lev.attrs["units"] = "centimeters"

    ds_test = correct_units(ds)
    assert ds_test.lev.attrs["units"] == "m"
    np.testing.assert_allclose(ds_test.lev.data, ds.lev.data / 100.0)
