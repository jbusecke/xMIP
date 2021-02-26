import numpy as np
import pytest
import xarray as xr

from cmip6_preprocessing.preprocessing import combined_preprocessing
from cmip6_preprocessing.regionmask import _default_merge_dict, merged_mask


regionmask = pytest.importorskip(
    "regionmask", minversion="0.5.0+dev"
)  # All tests get skipped if the version of regionmask is not > 0.5.0


@pytest.mark.parametrize("verbose", [True, False])
def test_merge_mask(verbose):
    x = np.linspace(0, 360, 720)
    y = np.linspace(-90, 90, 360)
    data = np.random.rand(len(x), len(y))
    ds = xr.DataArray(data, coords=[("x", x), ("y", y)]).to_dataset(name="data")
    ds["lon"] = ds["x"] * xr.ones_like(ds["y"])
    ds["lat"] = xr.ones_like(ds["x"]) * ds["y"]

    basins = regionmask.defined_regions.natural_earth.ocean_basins_50

    mask = merged_mask(basins, ds, verbose=verbose)

    # check if number of regions is correct
    mask_regions = np.unique(mask.data.flat)
    mask_regions = mask_regions[~np.isnan(mask_regions)]

    assert len(mask_regions) == len(_default_merge_dict().keys())

    # now a brief range check to make sure the pacific is stamped out correctly
    pac = ds.where(
        np.logical_or(np.logical_or(mask == 2, mask == 3), mask == 4), drop=True
    )
    assert pac.lon.min() > 95.0
    assert pac.lon.max() < 295.0

    # I shoud add a test for -180-180

    # How to use cloud data.
