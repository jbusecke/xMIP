# This module tests data directly from the pangeo google cloud storage.
# Tests are meant to be more high level and also serve to document known problems (see skip statements).
import pytest
import xarray as xr
import numpy as np
from cmip6_preprocessing.tests.cloud_test_utils import (
    full_specs,
    xfail_wrapper,
    all_models,
    data,
    diagnose_doubles,
)
from cmip6_preprocessing.preprocessing import combined_preprocessing
from cmip6_preprocessing.grids import combine_staggered_grid

pytest.importorskip("gcsfs")

print(f"\n\n\n\n$$$$$$$ All available models: {all_models()}$$$$$$$\n\n\n\n")

############################### Specific Bound Coords Test ###############################
expected_failures = [
    ("AWI-ESM-1-1-LR", "thetao", "historical", "gn"),
    ("AWI-ESM-1-1-MR", "thetao", "historical", "gn"),
    ("AWI-ESM-1-1-MR", "thetao", "ssp585", "gn"),
    ("AWI-CM-1-1-MR", "thetao", "historical", "gn"),
    ("AWI-CM-1-1-MR", "thetao", "ssp585", "gn"),
    ("CESM2-FV2", "thetao", "historical", "gn"),
    ("FGOALS-f3-L", "thetao", "historical", "gn"),
    ("FGOALS-f3-L", "thetao", "ssp585", "gn"),
    ("FGOALS-g3", "thetao", "ssp585", "gn"),
    ("NorESM2-MM", "thetao", "historical", "gn"),
    ("IPSL-CM6A-LR", "thetao", "historical", "gn"),
    ("IPSL-CM6A-LR", "o2", "historical", "gn"),
]


@pytest.mark.parametrize(
    "source_id,variable_id,experiment_id,grid_label",
    xfail_wrapper(full_specs(), expected_failures),
)
def test_check_bounds_verticies(source_id, variable_id, experiment_id, grid_label):

    ds, cat = data(source_id, variable_id, experiment_id, grid_label, True)

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
