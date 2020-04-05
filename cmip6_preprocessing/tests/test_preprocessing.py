import pytest
import intake
import pandas as pd
import numpy as np
import xarray as xr
from cmip6_preprocessing.preprocessing import (
    cmip6_renaming_dict,
    rename_cmip6,
    promote_empty_dims,
    replace_x_y_nominal_lat_lon,
    correct_coordinates,
    correct_lon,
)

# get all available ocean models from the cloud.
url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.csv"
df = pd.read_csv(url)
df_ocean = df[(df.table_id == "Omon") + (df.table_id == "Oyr")]
ocean_models = df_ocean.source_id.unique()

# TODO: Need to adapt atmos only models
all_models = ocean_models

required_coords = ["x", "y", "lon", "lat"]
additonal_coords = [
    "lev",
    "lev_bounds",
    "bnds",
    "time_bounds",
    "vertex",
    "lat_bounds",
    "lon_bounds",
]


def test_renaming_dict_keys():
    # check that each model has an entry
    rename_dict = cmip6_renaming_dict()
    print(set(all_models) - set(rename_dict.keys()))
    assert set(all_models).issubset(set(rename_dict.keys()))


@pytest.mark.parametrize("model", all_models)
def test_renaming_dict_entry_keys(model):
    if "AWI" not in model:  # excluding the unstructured awi model
        # check that all required coords are there
        for co in required_coords:
            assert co in cmip6_renaming_dict()[model].keys()
        # check that (if present) x and y contain lon and lat entries
        if "x" in cmip6_renaming_dict()[model].keys():
            assert "lon" in cmip6_renaming_dict()[model]["x"]
        if "y" in cmip6_renaming_dict()[model].keys():
            assert "lat" in cmip6_renaming_dict()[model]["y"]

        # check that there are no extra entries
        assert (
            len(
                set(cmip6_renaming_dict()[model].keys())
                - set(required_coords + additonal_coords)
            )
            == 0
        )


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
# @pytest.mark.parametrize("lonname", ["lon", "longitude"])
# @pytest.mark.parametrize("latname", ["lat", "latitude"])
def test_rename_cmip6(xname, yname, zname):
    xlen, ylen, zlen = (10, 5, 6)
    ds = create_test_ds(xname, yname, zname, xlen, ylen, zlen)
    # TODO: Build the bounds into this.
    # Eventually I can use a universal dict will all possible combos instead
    # of the lenghty beast I am using now.
    universal_dict = {
        "test_id": {
            "x": ["i", "x", "lon"],
            "y": ["j", "y", "lat"],
            "lev": ["lev", "olev", "olevel", "deptht", "deptht"],
            "lon": ["lon", "longitude"],
            "lat": ["lat", "latitude"],
            # "lev_bounds": "lev_bounds",
            # "lon_bounds": "bounds_lon",
            # "lat_bounds": "bounds_lat",
            # "bnds": "axis_nbounds",
            # "vertex": None,
            # "time_bounds": "time_bnds",
        }
    }

    ds_renamed = rename_cmip6(ds, universal_dict)
    assert set(ds_renamed.dims) == set(["x", "y", "lev"])
    assert (set(ds_renamed.coords) - set(ds_renamed.dims)) == set(["lon", "lat"])
    assert xlen == len(ds_renamed.x)
    assert ylen == len(ds_renamed.y)
    assert zlen == len(ds_renamed.lev)


@pytest.mark.parametrize("source_id", ["test", "other"])
def test_rename_cmip6_unkown_name(source_id):
    xlen, ylen, zlen = (10, 5, 6)
    ds = create_test_ds("x", "y", "z", xlen, ylen, zlen)
    ds.attrs["source_id"] = source_id
    # input dictionary empty for source_id: `%s`

    # Test when there is no dict entry for the source_id
    universal_dict = {}
    with pytest.warns(
        UserWarning,
        match=f"No input dictionary entry for source_id: `{ds.attrs['source_id']}`",
    ):
        ds_renamed = rename_cmip6(ds, universal_dict)

    # and now if the entry is there but its empty itself
    # TODO: These can probably go as soon as I have implemented the single renaming dict
    universal_dict = {source_id: {}}
    with pytest.warns(
        UserWarning,
        match=f"input dictionary empty for source_id: `{ds.attrs['source_id']}`",
    ):
        ds_renamed = rename_cmip6(ds, universal_dict)


def test_promote_empty_dims():
    xlen, ylen, zlen = (10, 5, 6)
    ds = create_test_ds("x", "y", "z", xlen, ylen, zlen)
    ds = ds.drop(["x", "y", "z"])
    ds_promoted = promote_empty_dims(ds)
    assert set(["x", "y", "z"]).issubset(set(ds_promoted.coords))


def test_replace_x_y_nominal_lat_lon():
    x = np.linspace(0, 720, 10)
    y = np.linspace(-200, 140, 5)
    lon = xr.DataArray(np.linspace(0, 360, len(x)), coords=[("x", x)])
    lat = xr.DataArray(np.linspace(-90, 90, len(y)), coords=[("y", y)])
    llon = lon * xr.ones_like(lat)
    llat = xr.ones_like(lon) * lat

    data = np.random.rand(len(x), len(y))
    ds = xr.DataArray(data, coords=[("x", x), ("y", y)]).to_dataset(name="data")
    ds["lon"] = llon
    ds["lat"] = llat
    replaced_ds = replace_x_y_nominal_lat_lon(ds)
    print(replaced_ds.x.data)
    print(lon.data)
    np.testing.assert_allclose(replaced_ds.x, lon)
    np.testing.assert_allclose(replaced_ds.y, lat)
    # assert np.testing.assert_allclose(replaced_ds.y.data, lat.data)


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


@pytest.mark.parametrize("shift", [-70, -180, -360,])  # cant handle positive shifts yet
def test_correct_lon(shift):
    xlen, ylen, zlen = (40, 20, 6)
    ds = create_test_ds("x", "y", "lev", xlen, ylen, zlen)
    ds = ds.assign_coords(x=ds.x.data + shift)
    lon = ds["lon"].reset_coords(drop=True)
    ds = ds.assign_coords(lon=lon + shift)
    ds_lon_corrected = correct_lon(ds)
    assert ds_lon_corrected.x.min() >= 0
    assert ds_lon_corrected.x.max() <= 360
    assert ds_lon_corrected.lon.min() >= 0
    assert ds_lon_corrected.lat.max() <= 360
    assert all(ds_lon_corrected.x.diff("x") > 0)
