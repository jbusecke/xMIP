# This module tests data directly from the pangeo google cloud storage.
# Tests are meant to be more high level and also serve to document known problems (see skip statements).
import pytest

pytest.skip("Deactivate sep test for now")
pytest.importorskip("gcsfs")

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

print(f"\n\n\n\n$$$$$$$ All available models: {all_models()}$$$$$$$\n\n\n\n")
