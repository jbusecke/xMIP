# This module tests data directly from the pangeo google cloud storage.
# Tests are meant to be more high level and also serve to document known problems (see skip statements).
import pytest
import xarray as xr
import numpy as np
from cmip6_preprocessing.tests.cloud_test_utils import (
    combine_specs,
    xfail_wrapper,
    xfail_wrapper_single,
    all_models,
    data,
    diagnose_doubles,
)
from cmip6_preprocessing.preprocessing import combined_preprocessing
from cmip6_preprocessing.grids import combine_staggered_grid

pytest.importorskip("gcsfs")

# does this work?
def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".

    for name in ["vi", "gl", "ei"]:

        option_value = getattr(metafunc.config.option, name)

        if isinstance(option_value, str):
            option_value = [option_value]

        if name in metafunc.fixturenames and option_value is not None:
            metafunc.parametrize(name, option_value)


@pytest.fixture(params=all_models())
def source_id(request):
    return request.param


print(f"\n\n\n\n$$$$$$$ All available models: {all_models()}$$$$$$$\n\n\n\n")

## Combine the input parameters according to command line input

########################### Most basic test #########################
expected_failures = [
    ("AWI-ESM-1-1-LR", "thetao", "historical", "gn"),
    ("AWI-ESM-1-1-LR", "thetao", "ssp585", "gn"),
    ("AWI-CM-1-1-MR", "thetao", "historical", "gn"),
    ("AWI-CM-1-1-MR", "thetao", "ssp585", "gn"),
    # TODO: would be nice to have a "*" matching...
    ("CESM2-FV2", "thetao", "historical", "gn"),  # this should fail
    # (
    #     "GFDL-CM4",
    #     "thetao",
    #     "historical",
    #     "gn",
    # ),  # this should not fail and should trigger an xfail
    ("CESM2-FV2", "thetao", "ssp585", "gn"),
]


@pytest.fixture()
def spec(source_id, vi, ei, gl):
    spec = (source_id, vi, ei, gl)
    if spec in expected_failures:
        # pytest.xfail()
        # This is not very satisfying, since it does not allow me
        # to check with strict true (and I might thus miss cases that
        # actually work but are marked as failing).
        # It would be nicer if I can mark a set of parameters as strictly xfailing

        # This is what I tried, and it doesnt work...
        # spec = tuple(pytest)
        pytest.param(*spec, marks=pytest.mark.xfail(strict=True))
        # spec = pytest.param(spec, marks=pytest.mark.xfail(strict=True))
        # spec = tuple(
        #     [pytest.param(s, marks=pytest.mark.xfail(strict=True)) for s in spec]
        # )
    return spec


def test_check_dim_coord_values_wo_intake(
    spec,
):
    source_id, variable_id, experiment_id, grid_label = spec
    # there must be a better way to build this at the class level and then tear it down again
    # I can probably get this done with fixtures, but I dont know how atm
    ds, _ = data(source_id, variable_id, experiment_id, grid_label, False)

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
            diagnose_doubles(ds[di].load().data)
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


# expected_failures = [
#     ("AWI-ESM-1-1-LR", "thetao", "historical", "gn"),
#     ("AWI-ESM-1-1-LR", "thetao", "ssp585", "gn"),
#     ("AWI-CM-1-1-MR", "thetao", "historical", "gn"),
#     ("AWI-CM-1-1-MR", "thetao", "ssp585", "gn"),
#     # TODO: would be nice to have a "*" matching...
#     ("CESM2-FV2", "thetao", "historical", "gn"),
#     ("CESM2-FV2", "thetao", "ssp585", "gn"),
#     (
#         "IPSL-CM6A-LR",
#         "thetao",
#         "historical",
#         "gn",
#     ),  # IPSL has an issue with `lev` dims concatting
#     ("IPSL-CM6A-LR", "o2", "historical", "gn"),
#     ("NorESM2-MM", "thetao", "historical", "gn"),
#     ("NorESM2-MM", "thetao", "historical", "gr"),
# ]


# @pytest.mark.parametrize(
#     "source_id,variable_id,experiment_id,grid_label",
#     xfail_wrapper(full_specs, expected_failures),
# )
# def test_check_dim_coord_values(source_id, variable_id, experiment_id, grid_label):
#     # there must be a better way to build this at the class level and then tear it down again
#     # I can probably get this done with fixtures, but I dont know how atm
#     ds, cat = data(source_id, variable_id, experiment_id, grid_label, True)

#     if ds is None:
#         pytest.skip(
#             f"No data found for {source_id}|{variable_id}|{experiment_id}|{grid_label}"
#         )

#     ##### Check for dim duplicates
#     # check all dims for duplicates
#     # for di in ds.dims:
#     # for now only test a subset of the dims. TODO: Add the bounds once they
#     # are cleaned up.
#     for di in ["x", "y", "lev", "time"]:
#         if di in ds.dims:
#             diagnose_doubles(ds[di].load().data)
#             assert len(ds[di]) == len(np.unique(ds[di]))
#             if di != "time":  # these tests do not make sense for decoded time
#                 assert ~np.all(np.isnan(ds[di]))
#                 assert np.all(ds[di].diff(di) >= 0)

#     assert ds.lon.min().load() >= 0
#     assert ds.lon.max().load() <= 360
#     if "lon_bounds" in ds.variables:
#         assert ds.lon_bounds.min().load() >= 0
#         assert ds.lon_bounds.max().load() <= 360
#     assert ds.lat.min().load() >= -90
#     assert ds.lat.max().load() <= 90
#     # make sure lon and lat are 2d
#     assert len(ds.lon.shape) == 2
#     assert len(ds.lat.shape) == 2


# ############################### Specific Bound Coords Test ###############################
# expected_failures = [
#     ("AWI-ESM-1-1-LR", "thetao", "historical", "gn"),
#     ("AWI-ESM-1-1-MR", "thetao", "historical", "gn"),
#     ("AWI-ESM-1-1-MR", "thetao", "ssp585", "gn"),
#     ("AWI-CM-1-1-MR", "thetao", "historical", "gn"),
#     ("AWI-CM-1-1-MR", "thetao", "ssp585", "gn"),
#     ("CESM2-FV2", "thetao", "historical", "gn"),
#     ("FGOALS-f3-L", "thetao", "historical", "gn"),
#     ("FGOALS-f3-L", "thetao", "ssp585", "gn"),
#     ("FGOALS-g3", "thetao", "historical", "gn"),
#     ("FGOALS-g3", "thetao", "ssp585", "gn"),
#     ("NorESM2-MM", "thetao", "historical", "gn"),
#     ("NorESM2-MM", "thetao", "historical", "gr"),
#     ("IPSL-CM6A-LR", "thetao", "historical", "gn"),
#     ("IPSL-CM6A-LR", "o2", "historical", "gn"),
# ]


# @pytest.mark.parametrize(
#     "source_id,variable_id,experiment_id,grid_label",
#     xfail_wrapper(full_specs, expected_failures),
# )
# def test_check_bounds_verticies(source_id, variable_id, experiment_id, grid_label):

#     ds, cat = data(source_id, variable_id, experiment_id, grid_label, True)

#     if ds is None:
#         pytest.skip(
#             f"No data found for {source_id}|{variable_id}|{experiment_id}|{grid_label}"
#         )

#     if "vertex" in ds.dims:
#         np.testing.assert_allclose(ds.vertex.data, np.arange(4))

#     ####Check for existing bounds and verticies
#     for co in ["lon_bounds", "lat_bounds", "lon_verticies", "lat_verticies"]:
#         assert co in ds.coords
#         # make sure that all other dims are eliminated from the bounds.
#         assert (set(ds[co].dims) - set(["bnds", "vertex"])) == set(["x", "y"])

#     #### Check the order of the vertex
#     # Ill only check these south of the Arctic for now. Up there
#     # things are still weird.

#     test_ds = ds.sel(y=slice(-40, 40))

#     vertex_lon_diff1 = test_ds.lon_verticies.isel(
#         vertex=3
#     ) - test_ds.lon_verticies.isel(vertex=0)
#     vertex_lon_diff2 = test_ds.lon_verticies.isel(
#         vertex=2
#     ) - test_ds.lon_verticies.isel(vertex=1)
#     vertex_lat_diff1 = test_ds.lat_verticies.isel(
#         vertex=1
#     ) - test_ds.lat_verticies.isel(vertex=0)
#     vertex_lat_diff2 = test_ds.lat_verticies.isel(
#         vertex=2
#     ) - test_ds.lat_verticies.isel(vertex=3)
#     for vertex_diff in [vertex_lon_diff1, vertex_lon_diff2]:
#         assert (vertex_diff <= 0).sum() <= (3 * len(vertex_diff.y))
#         # allowing for a few rows to be negative

#     for vertex_diff in [vertex_lat_diff1, vertex_lat_diff2]:
#         assert (vertex_diff <= 0).sum() <= (5 * len(vertex_diff.x))
#         # allowing for a few rows to be negative
#     # This is just to make sure that not the majority of values is negative or zero.

#     # Same for the bounds:
#     lon_diffs = test_ds.lon_bounds.diff("bnds")
#     lat_diffs = test_ds.lat_bounds.diff("bnds")

#     assert (lon_diffs <= 0).sum() <= (5 * len(lon_diffs.y))
#     assert (lat_diffs <= 0).sum() <= (5 * len(lat_diffs.y))


# ################################# xgcm grid specific tests ########################################
# expected_failures = [
#     ("AWI-ESM-1-1-LR", "thetao", "historical", "gn"),
#     ("AWI-ESM-1-1-MR", "thetao", "historical", "gn"),
#     ("AWI-ESM-1-1-MR", "thetao", "ssp585", "gn"),
#     ("AWI-CM-1-1-MR", "thetao", "historical", "gn"),
#     ("AWI-CM-1-1-MR", "thetao", "ssp585", "gn"),
#     ("CESM2-FV2", "thetao", "historical", "gn"),
#     ("CMCC-CM2-SR5", "thetao", "historical", "gn"),
#     ("CMCC-CM2-SR5", "thetao", "ssp585", "gn"),
#     ("FGOALS-f3-L", "thetao", "historical", "gn"),
#     ("FGOALS-f3-L", "thetao", "ssp585", "gn"),
#     ("FGOALS-g3", "thetao", "historical", "gn"),
#     ("FGOALS-g3", "thetao", "ssp585", "gn"),
#     ("MPI-ESM-1-2-HAM", "thetao", "historical", "gn"),
#     ("MPI-ESM-1-2-HAM", "o2", "historical", "gn"),
#     ("NorESM2-MM", "thetao", "historical", "gn"),
#     ("NorESM2-MM", "thetao", "historical", "gr"),
#     ("IPSL-CM6A-LR", "thetao", "historical", "gn"),
#     ("IPSL-CM6A-LR", "o2", "historical", "gn"),
# ]


# @pytest.mark.parametrize(
#     "source_id,variable_id,experiment_id,grid_label",
#     xfail_wrapper(full_specs, expected_failures),
# )
# def test_check_grid(source_id, variable_id, experiment_id, grid_label):

#     ds, cat = data(source_id, variable_id, experiment_id, grid_label, True)

#     if ds is None:
#         pytest.skip(
#             f"No data found for {source_id}|{variable_id}|{experiment_id}|{grid_label}"
#         )

#     # This is just a rudimentary test to see if the creation works
#     staggered_grid, ds_staggered = combine_staggered_grid(ds, recalculate_metrics=True)

#     print(ds_staggered)

#     assert ds_staggered is not None
#     #
#     if "lev" in ds_staggered.dims:
#         assert "bnds" in ds_staggered.lev_bounds.dims

#     for axis in ["X", "Y"]:
#         for metric in ["_t", "_gx", "_gy", "_gxgy"]:
#             assert f"d{axis.lower()}{metric}" in list(ds_staggered.coords)
#     # TODO: Include actual test to combine variables
