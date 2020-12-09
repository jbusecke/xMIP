# This module tests data directly from the pangeo google cloud storage.
# Tests are meant to be more high level and also serve to document known problems (see skip statements).
import pytest
import contextlib
import xarray as xr
import numpy as np
import intake
import fsspec
from cmip6_preprocessing.preprocessing import combined_preprocessing
from cmip6_preprocessing.grids import combine_staggered_grid

pytest.importorskip("gcsfs")


def col():
    return intake.open_esm_datastore(
        "https://raw.githubusercontent.com/NCAR/intake-esm-datastore/master/catalogs/pangeo-cmip6.json"
    )


def all_models():
    df = col().df
    all_models = df["source_id"].unique()
    all_models = np.sort(all_models)
    # For testing purposes, only put out a subset of models
    # return ["CanESM5", "GFDL-ESM4"]
    return all_models


def _diagnose_doubles(data):
    """displays non-unique entries in data"""
    _, idx = np.unique(data, return_index=True)
    missing = np.array([i for i in np.arange(len(data)) if i not in idx])
    if len(missing) > 0:
        missing_values = data[missing]
        print(f"Missing values Indicies[{missing}]/ Values[{missing_values}]")


def xfail_wrapper(models, fail_models):
    return [
        pytest.param(model, marks=pytest.mark.xfail(strict=True))
        if model in fail_models
        else model
        for model in models
    ]


zarr_kwargs = {
    "consolidated": True,
    "decode_times": False,
    # "decode_times": True,
    # "use_cftime": True,
}


def data(grid_label, experiment_id, variable_id, source_id):
    cat = col().search(
        source_id=source_id,
        experiment_id=experiment_id,
        variable_id=variable_id,
        # member_id="r1i1p1f1",
        table_id="Omon",
        grid_label=grid_label,
    )

    if len(cat.df["zstore"]) > 0:
        ddict = cat.to_dataset_dict(
            zarr_kwargs=zarr_kwargs,
            preprocess=combined_preprocessing,
            storage_options={"token": "anon"},
        )

        # @charlesbluca suggested this to make this work in GHA
        # https://github.com/jbusecke/cmip6_preprocessing/pull/62#issuecomment-741928365
        mm = fsspec.get_mapper(
            cat.df["zstore"][0]
        )  # think you can pass in storage options here as well?
        ds_raw = xr.open_zarr(mm, **zarr_kwargs)

        ds_manual = combined_preprocessing(ds_raw)

        _, ds = ddict.popitem()
    else:
        ds = None

    print(ds)
    return ds, cat


print(f"\n\n\n\n$$$$$$$ All available models: {all_models()}$$$$$$$\n\n\n\n")

# These are too many tests. Perhaps I could load all the data first and then
# test each dict item?

grid_labels = ["gn"]
experiment_ids = ["historical"]
variable_ids = ["thetao"]


@pytest.mark.parametrize("grid_label", grid_labels)
@pytest.mark.parametrize("experiment_id", experiment_ids)
@pytest.mark.parametrize("variable_id", variable_ids)
@pytest.mark.parametrize(
    "source_id",
    xfail_wrapper(
        all_models(),
        [
            "IPSL-CM6A-LR",  # There is a file not found error for thetao. Need to investigate. # some models have read errors? Indicates that there is a problem with the catalog?
            "NorESM2-MM",  # gives concat error with undecoded time (duplicate values)
            "CESM2-FV2",  # has duplicate values in the x
            #! This should be fixed already
            "AWI-CM-1-1-MR",  # these unstructured AWI outputs are not currently supported.
            "AWI-ESM-1-1-LR",
        ],
    ),
)
def test_check_dim_coord_values(grid_label, experiment_id, variable_id, source_id):
    # there must be a better way to build this at the class level and then tear it down again
    # I can probably get this done with fixtures, but I dont know how atm
    ds, cat = data(grid_label, experiment_id, variable_id, source_id)

    if ds is None:
        pytest.skip(
            f"No data found for {source_id}|{variable_id}|{experiment_id}|{grid_label}"
        )

    ##### Check for dim duplicates
    # check all dims for duplicates
    # for di in ds.dims:
    # for now only test a subset of the dims. TODO: Add the bounds once they
    # are cleaned up.
    for di in ["x", "y", "lev", "time"]:
        if di in ds.dims:
            _diagnose_doubles(ds[di].load().data)
            assert len(ds[di]) == len(np.unique(ds[di]))
            if di != "time":  # these tests do not make sense for decoded time
                assert ~np.all(np.isnan(ds[di]))
                assert np.all(ds[di].diff(di) >= 0)

    assert ds.lon.min().load() >= 0
    assert ds.lon.max().load() <= 360
    if "lon_bounds" in ds.variables:
        assert ds.lon_bounds.min().load() >= 0
        assert ds.lon_bounds.max().load() <= 360
    assert ds.lat.min().load() >= -90
    assert ds.lat.max().load() <= 90
    # make sure lon and lat are 2d
    assert len(ds.lon.shape) == 2
    assert len(ds.lat.shape) == 2


@pytest.mark.parametrize("grid_label", grid_labels)
@pytest.mark.parametrize("experiment_id", experiment_ids)
@pytest.mark.parametrize("variable_id", variable_ids)
@pytest.mark.parametrize(
    "source_id",
    xfail_wrapper(
        all_models(),
        [
            "IPSL-CM6A-LR",  # There is a file not found error for thetao. Need to investigate. # some models have read errors? Indicates that there is a problem with the catalog?
            "NorESM2-MM",  # gives concat error with undecoded time (duplicate values)
            "CESM2-FV2",  # has duplicate values in the x
            #! This should be fixed already
            "AWI-CM-1-1-MR",  # these unstructured AWI outputs are not currently supported.
            "AWI-ESM-1-1-LR",
        ],
    ),
)
def test_check_bounds_verticies(grid_label, experiment_id, variable_id, source_id):

    ds, cat = data(grid_label, experiment_id, variable_id, source_id)

    if ds is None:
        pytest.skip(
            f"No data found for {source_id}|{variable_id}|{experiment_id}|{grid_label}"
        )

    if "vertex" in ds.dims:
        np.testing.assert_allclose(ds.vertex.data, np.arange(4))

    ####Check for existing bounds and verticies
    for co in ["lon_bounds", "lat_bounds", "lon_verticies", "lat_verticies"]:
        assert co in ds.coords
        # make sure that all other dims are eliminated from the bounds.
        assert (set(ds[co].dims) - set(["bnds", "vertex"])) == set(["x", "y"])

    #### Check the order of the vertex
    # Ill only check these south of the Arctic for now. Up there
    # things are still weird.

    test_ds = ds.sel(y=slice(-40, 40))

    vertex_lon_diff1 = test_ds.lon_verticies.isel(
        vertex=3
    ) - test_ds.lon_verticies.isel(vertex=0)
    vertex_lon_diff2 = test_ds.lon_verticies.isel(
        vertex=2
    ) - test_ds.lon_verticies.isel(vertex=1)
    vertex_lat_diff1 = test_ds.lat_verticies.isel(
        vertex=1
    ) - test_ds.lat_verticies.isel(vertex=0)
    vertex_lat_diff2 = test_ds.lat_verticies.isel(
        vertex=2
    ) - test_ds.lat_verticies.isel(vertex=3)
    for vertex_diff in [vertex_lon_diff1, vertex_lon_diff2]:
        assert (vertex_diff <= 0).sum() <= (3 * len(vertex_diff.y))
        # allowing for a few rows to be negative

    for vertex_diff in [vertex_lat_diff1, vertex_lat_diff2]:
        assert (vertex_diff <= 0).sum() <= (5 * len(vertex_diff.x))
        # allowing for a few rows to be negative
    # This is just to make sure that not the majority of values is negative or zero.

    # Same for the bounds:
    lon_diffs = test_ds.lon_bounds.diff("bnds")
    lat_diffs = test_ds.lat_bounds.diff("bnds")

    assert (lon_diffs <= 0).sum() <= (5 * len(lon_diffs.y))
    assert (lat_diffs <= 0).sum() <= (5 * len(lat_diffs.y))


@pytest.mark.parametrize("grid_label", grid_labels)
@pytest.mark.parametrize("experiment_id", experiment_ids)
@pytest.mark.parametrize("variable_id", variable_ids)
@pytest.mark.parametrize(
    "source_id",
    xfail_wrapper(
        all_models(),
        [
            "IPSL-CM6A-LR",  # There is a file not found error for thetao. Need to investigate. # some models have read errors? Indicates that there is a problem with the catalog?
            "NorESM2-MM",  # gives concat error with undecoded time (duplicate values)
            "CESM2-FV2",  # has duplicate values in the x
            #! This should be fixed already
            "AWI-CM-1-1-MR",  # these unstructured AWI outputs are not currently supported.
            "AWI-ESM-1-1-LR",
        ],
    ),
)
def test_check_grid(grid_label, experiment_id, variable_id, source_id):

    ds, cat = data(grid_label, experiment_id, variable_id, source_id)

    if ds is None:
        pytest.skip(
            f"No data found for {source_id}|{variable_id}|{experiment_id}|{grid_label}"
        )

    # This is just a rudimentary test to see if the creation works
    staggered_grid, ds_staggered = combine_staggered_grid(ds, recalculate_metrics=True)

    print(ds_staggered)

    assert ds_staggered is not None
    #
    if "lev" in ds_staggered.dims:
        assert "bnds" in ds_staggered.lev_bounds.dims

    for axis in ["X", "Y"]:
        for metric in ["_t", "_gx", "_gy", "_gxgy"]:
            assert f"d{axis.lower()}{metric}" in list(ds_staggered.coords)
    # TODO: Include actual test to combine variables
