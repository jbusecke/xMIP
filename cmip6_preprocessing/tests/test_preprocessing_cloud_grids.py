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

pytest.skip("Deactivate sep test for now")
pytest.importorskip("gcsfs")

print(f"\n\n\n\n$$$$$$$ All available models: {all_models()}$$$$$$$\n\n\n\n")


################################# xgcm grid specific tests ########################################
expected_failures = [
    ("AWI-ESM-1-1-LR", "thetao", "historical", "gn"),
    ("AWI-ESM-1-1-MR", "thetao", "historical", "gn"),
    ("AWI-ESM-1-1-MR", "thetao", "ssp585", "gn"),
    ("AWI-CM-1-1-MR", "thetao", "historical", "gn"),
    ("AWI-CM-1-1-MR", "thetao", "ssp585", "gn"),
    ("CESM2-FV2", "thetao", "historical", "gn"),
    ("CMCC-CM2-SR5", "thetao", "historical", "gn"),
    ("CMCC-CM2-SR5", "thetao", "ssp585", "gn"),
    ("FGOALS-f3-L", "thetao", "historical", "gn"),
    ("FGOALS-f3-L", "thetao", "ssp585", "gn"),
    ("FGOALS-g3", "thetao", "ssp585", "gn"),
    ("MPI-ESM-1-2-HAM", "thetao", "historical", "gn"),
    ("MPI-ESM-1-2-HAM", "o2", "historical", "gn"),
    ("NorESM2-MM", "thetao", "historical", "gn"),
    ("IPSL-CM6A-LR", "thetao", "historical", "gn"),
    ("IPSL-CM6A-LR", "o2", "historical", "gn"),
]


@pytest.mark.parametrize(
    "source_id,variable_id,experiment_id,grid_label",
    xfail_wrapper(full_specs(), expected_failures),
)
def test_check_grid(source_id, variable_id, experiment_id, grid_label):

    ds, cat = data(source_id, variable_id, experiment_id, grid_label, True)

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
